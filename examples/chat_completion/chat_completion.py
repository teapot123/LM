# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import inspect
import os
import sys
import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from tqdm import tqdm
import numpy as np

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens, create_batches
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.datasets import get_triviaqa_dataset_for_prediction, get_second_question_for_prediction, get_triviaqa_dataset_for_prediction_with_conf
from llama_recipes.configs import datasets



def extract_question_answer(outputs, tokenizer, infer_mode, use_chat_mode):
    """Extract question and answer from the output text."""
    questions, answers = [], []
    generated_tokens = outputs.sequences
    # print(f"sequences num: {len(generated_tokens)}")
    # print(f"sequences: {generated_tokens}")
    for response in generated_tokens:
        output_text = tokenizer.decode(response)
        if infer_mode == 'topk':
            if use_chat_mode or "[/INST]" in output_text:
                question = output_text.split('[/INST]')[0].split('The question is:')[1].strip()
                answer_conf = output_text.split('[/INST]')[1]
                answer = answer_conf.replace('\n', '\t').strip()
            else:
                question = output_text.split('\n\n')[2].strip()
                answer = output_text.split(question)[1].replace('\n\n', '\t').replace('\n', '\t').strip()       
        elif infer_mode == 'generate_recitation':
            texts = output_text.split('[/INST]')[0].split('\n\nQuestion 6: ')[-1]
            question = texts.split('\n\n')[0].strip()
            answer = output_text.split('[/INST]')[1].split('\n\n')[1].replace('\n', '\t').strip()
            if answer[0] == '"' and answer[-1] == '"':
                answer = answer[1:-1]
        questions.append(question)
        answers.append(answer)
    return questions, answers

def analyze_attention(outputs, tokenizer, questions, answers):
    generated_tokens = outputs.sequences
    attention_weights = outputs.attentions
    batch_attentions, token_ids = [], []
    # print(f"number of questions, answers, outputs: {len(questions)} {len(answers)} {len(generated_tokens)}")
    for i, response in enumerate(generated_tokens):
        output_text = tokenizer.decode(response)
        prompt_end_index = len(attention_weights[0][0][i][0]) + 1  # should be -1, but it seems there are two '\n'
        question = questions[i]
        answer = answers[i]
        if 'Guess:' in answer:
            answer = answer.split('Guess:')[1].strip()
        try:
            text_before_answer = output_text.split(question)[0] + question + output_text.split(question)[1].split(answer)[0]
        except ValueError:
            batch_attentions.append([])
            token_ids.append([])
            continue
        answer_start_index = len(tokenizer.tokenize(text_before_answer))
        answer_offset = answer_start_index - prompt_end_index
        # print(f'output length: {len(response)} output text: {tokenizer.tokenize(output_text)}')
        # print(f'answer: {answer}')
        # print(f'prompt end index: {prompt_end_index}  answer start index: {answer_start_index} answer offset: {answer_offset}')

        # print(f'attention weight matrix: {len(attention_weights[0][0][i][0])}x{len(attention_weights[0][0][i][0][0])}')
        answer_token_outputs = attention_weights[answer_offset]
        # print(f'attention weight size: {answer_token_outputs[0][i].size()}')
        attentions = torch.stack([x[i] for x in answer_token_outputs], dim=0)
        attentions = torch.squeeze(attentions) # 40 x 40 x (seq_len + 1)
        # attentions = attentions[:, :, prompt_end_index + 1:]
        # print(f'attentions size: {attentions.size()}')

        max_head_att = torch.mean(attentions, dim=1)
        max_layer_max_head_att = torch.mean(max_head_att, dim=0)
        assert len(max_layer_max_head_att) == answer_start_index - 1, f"head_att: {len(max_layer_max_head_att)} != answer_start_index {answer_start_index-1}"

        batch_attentions.append(max_layer_max_head_att)
        token_ids.append(response)
    return batch_attentions, token_ids


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    batch_size: int=4,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    partition: str='validation',
    prompt_file: str=None,
    read_data_func: str='get_triviaqa_dataset_for_prediction',
    output_file: str=None,
    att_file: str=None,
    logits_file: str=None,
    use_chat_mode: bool=False,
    infer_mode: str='topk',
    print_attention: bool=False,
    print_logits: bool=False,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=False, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)
        print(f"User dialog length:\n{len(dialogs)}")
        print("\n==================================\n")

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    elif not use_chat_mode:
        print("Using direct inference mode.")
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    # model = load_model(model_name, quantization)
    config = LlamaConfig.from_pretrained(model_name, output_attentions=print_attention, return_dict_in_generate=True)
    model = LlamaForCausalLM.from_pretrained(model_name,  device_map="auto", load_in_8bit=quantization, low_cpu_mem_usage=True, config=config)
    if peft_model:
        model = load_peft_model(model, peft_model)
        model = model.merge_and_unload()
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    if use_chat_mode:
        chats = format_tokens(dialogs, tokenizer)
    elif read_data_func == 'get_second_question_for_prediction':
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset_reorder']()
        train_data = get_second_question_for_prediction(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
    elif read_data_func == 'get_triviaqa_dataset_for_prediction':
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset']()
        train_data = get_triviaqa_dataset_for_prediction(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
    else:
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset']()
        train_data = get_triviaqa_dataset_for_prediction_with_conf(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
        

    chat_batches, attention_masks = create_batches(chats, batch_size=batch_size)

    with open(output_file, 'w') as fout:
        with torch.no_grad():
            if print_attention:
                f_att = open(att_file, 'w')
            elif print_logits:
                f_logits = open(logits_file, 'w')
            for idx, chat in tqdm(enumerate(chat_batches), total=len(chat_batches)):
                attention_mask = attention_masks[idx]
                tokens= torch.tensor(chat).long()
                tokens= tokens.to("cuda:0")
                attention_mask= attention_mask.to("cuda:0")
                outputs = model.generate(
                    tokens,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    output_attentions=print_attention,
                    output_scores=print_logits,
                    return_dict_in_generate=True,
                    **kwargs
                )
                questions, answers = extract_question_answer(outputs, tokenizer, infer_mode, use_chat_mode)
                for (question, answer) in zip(questions, answers):
                    fout.write(f"{question}\t{answer}\n")
                if print_attention:
                    batch_attentions, token_ids = analyze_attention(outputs, tokenizer, questions, answers)
                    # print(f"token ids: {token_ids}")
                    # print(f"attentions: {batch_attentions}")
                    for i, attention in enumerate(batch_attentions):
                        if attention == []:
                            continue
                        token_id = token_ids[i]
                        for j in range(len(attention)):
                            # print(f"token_id {token_id}")
                            if tokenizer.decode(token_id[j]) == '<unk>':
                                continue
                            f_att.write(f"{tokenizer.decode(token_id[j])} ({attention[j]:.3f}) ")
                        f_att.write(answers[i]+'\n')
                elif print_logits:
                    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                    generated_tokens = outputs.sequences
                    for i, output_score in enumerate(generated_tokens):
                        output_score = np.exp(transition_scores[i].detach().cpu().numpy())
                        token_id = generated_tokens[i]
                        output_offset = len(token_id) - len(output_score)
                        f_logits.write(f"{questions[i]}|||{answers[i]}|||")
                        for j in range(len(token_id)):
                            # print(f"token_id {token_id}")
                            if tokenizer.decode(token_id[j]) == '<unk>':
                                continue
                            if j >= output_offset:
                                f_logits.write(f"{tokenizer.decode(token_id[j])} ({output_score[j-output_offset]:.3f})\t")
                            # else:
                            #     f_logits.write(f"{tokenizer.decode(token_id[j])} ")
                        f_logits.write('\n')
                        

if __name__ == "__main__":
    fire.Fire(main)

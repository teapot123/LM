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

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens, create_batches
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.datasets import get_triviaqa_dataset_for_reordering, get_paraphrased_triviaqa_dataset, get_both_questions_for_correction, get_triviaqa_dataset_for_prediction_with_conf
from llama_recipes.configs import datasets

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    batch_size: int=4,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    read_data_func: str='get_triviaqa_dataset_for_prediction',
    prompt_file: str=None,
    output_file: str=None,
    partition: str=None,
    use_chat_mode: bool=False,
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
    config = LlamaConfig.from_pretrained(model_name)
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
    elif read_data_func == 'get_triviaqa_dataset_for_reordering':
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset']()
        train_data = get_triviaqa_dataset_for_reordering(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
    elif read_data_func == 'get_both_questions_for_correction':
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset_reorder']()
        train_data = get_both_questions_for_correction(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
    else:
        dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset']()
        train_data = get_triviaqa_dataset_for_prediction_with_conf(dataset_config, tokenizer, partition)
        chats = train_data['input_ids']
    

    chat_batches, attention_masks = create_batches(chats, batch_size=batch_size)

    with open(output_file, 'w') as fout:
        with torch.no_grad():
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
                    **kwargs
                )
                for output in outputs:
                    fout.write(tokenizer.decode(output).replace('\n','\t').strip()+'\n')
                

if __name__ == "__main__":
    fire.Fire(main)

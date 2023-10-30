# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import torch
from transformers import LlamaTokenizer
from tqdm import tqdm

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens, create_batches
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.inference.safety_utils import get_safety_checker


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    batch_size: int=4,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    output_file: str=None,
    mode: str='topk',
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
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

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialog length:\n{len(dialogs)}")
    print("\n==================================\n")


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)
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
    
    chats = format_tokens(dialogs, tokenizer)
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
                    output_attentions=True,
                    **kwargs
                )
                print(outputs)
                print('--------')
                print(outputs[0])
                for output in outputs:
                    output_text = tokenizer.decode(output, skip_special_tokens=True)
                    print(f"{output_text}\n")
                    if mode == 'topk':
                        question = output_text.split('[/INST]')[0].split('The question is:')[1].strip()
                        answer_conf = output_text.split('[/INST]')[1]
                        # try:
                        #     answer = answer_conf.split('Guess:')[1].split('\n')[0].split('Probability:')[0].strip()
                        #     conf = answer_conf.split('Probability:')[1].split('\n')[0].strip()
                        #     fout.write(f"{question}\t{answer}\t{conf}\n")
                        # except IndexError:
                        #     answer = answer_conf.replace('\n', ' ').strip()
                        #     fout.write(f"{question}\t{answer}\n")
                        answer = answer_conf.replace('\n', '\t').strip()
                        fout.write(f"{question}\t{answer}\n")
                    elif mode == 'generate_recitation':
                        texts = output_text.split('[/INST]')[0].split('\n\nQuestion 6: ')[-1]
                        question = texts.split('\n\n')[0].strip()
                        answer = output_text.split('[/INST]')[1].split('\n\n')[1].replace('\n', '\t').strip()
                        if answer[0] == '"' and answer[-1] == '"':
                            answer = answer[1:-1]
                        fout.write(f"{question}\t{answer}\n")



if __name__ == "__main__":
    fire.Fire(main)
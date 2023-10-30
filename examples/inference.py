# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


def main(
    model_name,
    num_generations_per_prompt: int=5,
    input_data: str=None,
    max_batch_size: int=6,
    peft_model: str=None,
    quantization: bool=True,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    system_prompt_file: str=None,
    prompt_file: str=None,
    output_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    
    if system_prompt_file is not None:
        assert os.path.exists(
            system_prompt_file
        ), f"Provided Prompt file does not exist {system_prompt_file}"
        with open(system_prompt_file, "r") as f:
            system_prompt = ''.join(f.readlines())

    input_data = datasets.load_from_disk(input_data)
    # input_data = [x for x in input_data for j in range(num_generations_per_prompt)]
    # print(input_data[0])
    print(len(input_data))
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
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
    tokenizer.pad_token = tokenizer.eos_token
    
    # input_data = input_data[0:10]
    batches = [input_data[i:i+max_batch_size] for i in range(0, len(input_data), max_batch_size)]
    # batches = [tokenizer(x, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt") for x in batch]
    # batches = [combine_tensors_with_padding(x) for x in batches]

    print(len(batches))
    
    batch = batches[0]
    print(batch)

    batch = {k: v.to("cuda") for k, v in batch.items() if torch.is_tensor(v)}
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    for output in outputs:
        print(output)
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Model output:\n{output_text}")

    
    # if output_file != None: 
    #     with open(output_file, 'w') as fout:
    #         for i, batch in enumerate(batches):
    #             batch = {k: v.to("cuda") for k, v in batch.items()}
    #             start = time.perf_counter()
    #             with torch.no_grad():
    #                 outputs = model.generate(
    #                     **batch,
    #                     max_new_tokens=max_new_tokens,
    #                     do_sample=do_sample,
    #                     top_p=top_p,
    #                     temperature=temperature,
    #                     min_length=min_length,
    #                     use_cache=use_cache,
    #                     top_k=top_k,
    #                     repetition_penalty=repetition_penalty,
    #                     length_penalty=length_penalty,
    #                     **kwargs 
    #                 )
    #             e2e_inference_time = (time.perf_counter()-start)*1000
    #             print(f"the inference time is {e2e_inference_time} ms")
                
    #             for j, output in enumerate(outputs):
    #                 ind = int((i * max_batch_size + j) / num_generations_per_prompt)
    #                 question = input_data[ind]['question'] 
    #                 output_text = tokenizer.decode(output, skip_special_tokens=True)
    #                 generated_answer = output_text.split(question)[-1].strip().split('\n')[0].split('Q:')[0].split('A:')[-1].strip()
    #                 # print(f"Question:\n{question}")
    #                 # print(f"Model output:\n{output_text}")
    #                 # print(f"generated_answer:\n{generated_answer}")
    #                 # print('-----------')
    #                 fout.write(question + '\t' + generated_answer + '\n')
            
    # else:
    #     for batch in batches:
    #         batch = {k: v.to("cuda") for k, v in batch.items()}
    #         start = time.perf_counter()
    #         with torch.no_grad():
    #             outputs = model.generate(
    #                 **batch,
    #                 max_new_tokens=max_new_tokens,
    #                 do_sample=do_sample,
    #                 top_p=top_p,
    #                 temperature=temperature,
    #                 min_length=min_length,
    #                 use_cache=use_cache,
    #                 top_k=top_k,
    #                 repetition_penalty=repetition_penalty,
    #                 length_penalty=length_penalty,
    #                 **kwargs 
    #             )
    #         e2e_inference_time = (time.perf_counter()-start)*1000
    #         print(f"the inference time is {e2e_inference_time} ms")
    #         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #         print(f"Model output:\n{output_text}")
   

if __name__ == "__main__":
    fire.Fire(main)
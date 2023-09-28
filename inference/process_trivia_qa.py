import argparse
import pathlib
import pickle
import json

import datasets
import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_data_num', type=int, default=1000)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--data_split', type=str)
    parser.add_argument('--down_sample', type=bool, default=False)
    parser.add_argument('--user_prompt_file', type=str, default=None)
    parser.add_argument('--system_prompt_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=[
        'topk', 'generate_recitation', 'with_recitation',
    ])
    args = parser.parse_args()

    

    print('Preprocessing dataset')
    if args.down_sample:
        val_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split=args.data_split)
        print('finish loading!')
        val_data = val_data.shuffle(seed=42)
        print('finish shuffling!')
        val_data = val_data.flatten_indices()
        print('finish flatten indices!')
        val_data = val_data.select(range(0,args.select_data_num))
    else:
        # model = LlamaForCausalLM.from_pretrained(args.model_name)
        # tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name)
        # tokenizer.add_special_tokens({"pad_token": "<PAD>",})
        if args.data_dir[-4:] == '.txt':
            with open(args.data_dir) as f:
                val_data = f.readlines()
        else:
            val_data = datasets.load_from_disk(args.data_dir)
        with open(args.system_prompt_file) as f:
            system_prompt = ''.join(f.readlines()).strip()
        with open(args.user_prompt_file) as f:
            user_prompt = ''.join(f.readlines())
        # few_shot_prompt = 'This is a bot that correctly answers questions. \n'
        # for sample in data_for_few_shot_prompt:
        #     few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + sample['answer']['value'] + ' '

    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def down_sample_data(batch):
        answers = [
            answer["aliases"]
            + answer["normalized_aliases"]
            + [answer['value']]
            + [answer['normalized_value']]
            for answer in batch["answer"]]
        answers = [list(set(answer)) for answer in answers]
        batch['answer'] = answers
        return batch

    
    def process_data_to_dialog_json(batch):
        
        # save as dialog json
        batch['dialogs'] = [[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt + question}
                              ] for question in batch["question"]]
        return batch
    
    def process_data_for_recitation(batch):
        
        # save as dialog json
        batch['dialogs'] = [[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt + ' ' + question + '\n\nParagraph 6:'}
                              ] for question in batch["question"]]
        return batch
    
    def process_data_w_recitation(texts):
        dialogs = []
        for text in texts:
            tmp = text.split('\t')
            question = tmp[0].strip()
            recitation = tmp[1].strip()
            dialogs.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt + recitation + ' ' + question}]
            )
        return dialogs

    if args.down_sample:
        val_data = val_data.map(down_sample_data,
                                batched=True,
                                batch_size=batch_size,
                                remove_columns=["search_results", "question_source", "entity_pages"])
        val_data.save_to_disk(f'../data/trivia_qa/{args.data_split}_{args.select_data_num}')
    else:
        if args.mode == 'topk':
            user_prompt_style=args.user_prompt_file.split('/')[-1].split('.')[0]
            # val_data_1.save_to_disk(f'{args.data_dir}_{user_prompt_style}')

            val_data_2 = val_data.map(process_data_to_dialog_json,
                                    batched = True,
                                    batch_size=batch_size)
            val_data_2 = val_data_2['dialogs']
        elif args.mode == 'generate_recitation':
            user_prompt_style='0recitation'
            val_data_2 = val_data.map(process_data_for_recitation,
                                    batched = True,
                                    batch_size=batch_size)
            val_data_2 = val_data_2['dialogs']
        elif args.mode == 'with_recitation':
            user_prompt_style='w_recitation'
            val_data_2 = process_data_w_recitation(val_data)
        if args.out_dir == None:
            outfilename = f'{args.data_dir}_{user_prompt_style}.json'
        else:
            outfilename = args.out_dir
        with open(outfilename, 'w') as fout:
            fout.write(json.dumps(val_data_2, indent = 4))

    

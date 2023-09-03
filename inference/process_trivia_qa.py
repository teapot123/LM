import argparse
import pathlib
import pickle

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
        model = LlamaForCausalLM.from_pretrained(args.model_name)
        tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name)
        tokenizer.add_special_tokens({"pad_token": "<PAD>",})
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
        answers = [answer["value"] for answer in batch["answer"]]
        batch['answer'] = answers
        return batch

    def process_data_to_model_inputs(batch):
        
        # tokenize the inputs and labels
        batch_with_prompt = [user_prompt + question for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=True, return_tensors="pt")
        outputs = tokenizer(batch['answer'], padding=False, truncation=False)
        print(inputs)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch
    
    def process_data_to_dialog_json(batch):
        
        # save as dialog json
        batch = [[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt + question}
                              ] for question in batch["question"]]

        return batch

    if args.down_sample:
        val_data = val_data.map(down_sample_data,
                                batched=True,
                                batch_size=batch_size,
                                remove_columns=["search_results", "question_source", "entity_pages"])
        val_data.save_to_disk(f'../data/trivia_qa/{args.data_split}_{args.select_data_num}')
    else:
        val_data_1 = val_data.map(process_data_to_model_inputs,
                                batched=True,
                                batch_size=batch_size,)
        val_data_1.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True)
        user_prompt_style=args.user_prompt_file.split('/')[-1].split('.')[0]
        val_data_1.save_to_disk(f'{args.data_dir}_{user_prompt_style}')

        val_data_2 = val_data.map(process_data_to_dialog_json,
                                  batched = True,
                                  batch_size=batch_size)
        val_data_2.to_json(f'{args.data_dir}_{user_prompt_style}.json')

    


import copy
import json
import os

import torch
import itertools
import random
from torch.utils.data import Dataset
import datasets
from llama_recipes.inference.chat_utils import format_tokens

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt="You are an intelligent, honest, and harmless assistant. Your direct, concise responses contain only the minimum words needed to convey the answer."


user_prompt = """Provide your best guess that it is correct for the following question. Give ONLY the guess, no other words or explanation. For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>

The question is: """

user_prompt_with_conf = """Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the guess and probability, no other words or explanation. For example:

Guess: <most likely guess, as short as possible; not a complete sentence, just the guess!>
Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>

The question is: """

def get_triviaqa_dataset(dataset_config, tokenizer, partition):
    if partition == 'train':
        # filename = 'train_10000'
        filename = 'train_138384'
    else:
        filename = 'validation_1000'
    # with open(dataset_config.system_prompt_path) as f:
    #     system_prompt = ''.join(f.readlines()).strip()
    # with open(dataset_config.user_prompt_path) as f:
    #     user_prompt = ''.join(f.readlines())
    def process_data(batch):  
        # save as dialog json
        prompt = tokenizer.encode(tokenizer.bos_token + user_prompt + batch["question"], add_special_tokens=False)
        output = tokenizer.encode('\n\n' + batch["answer"] + tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt + output,
            "attention_mask" : [1] * (len(prompt) + len(output)),
            "labels": [-100] * len(prompt) + output,
            }
        return sample
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, remove_columns=list(datafile.features))
    return dataset


def get_triviaqa_dataset_for_prediction(dataset_config, tokenizer, partition):
    if partition == 'train':
        # filename = 'train_10000'
        filename = 'train_138384'
    else:
        filename = 'validation_1000'
    # with open(dataset_config.system_prompt_path) as f:
    #     system_prompt = ''.join(f.readlines()).strip()
    # with open(dataset_config.user_prompt_path) as f:
    #     user_prompt = ''.join(f.readlines())
    def process_data(batch):  
        # save as dialog json
        txt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt+batch['question']} {E_INST}"
        prompt = tokenizer.encode(tokenizer.bos_token + txt, add_special_tokens=False)
        sample = {
            "input_ids": prompt,
            "answers": batch["answers"]
            }
        return sample
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, remove_columns=list(datafile.features))
    return dataset


def get_triviaqa_dataset_for_prediction_with_conf(dataset_config, tokenizer, partition):
    if partition == 'train':
        # filename = 'train_10000'
        filename = 'train_138384'
    else:
        filename = 'validation_1000'
    # with open(dataset_config.system_prompt_path) as f:
    #     system_prompt = ''.join(f.readlines()).strip()
    # with open(dataset_config.user_prompt_path) as f:
    #     user_prompt = ''.join(f.readlines())
    def process_data(batch):  
        # save as dialog json
        txt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt_with_conf+batch['question']} {E_INST}"
        prompt = tokenizer.encode(tokenizer.bos_token + txt, add_special_tokens=False)
        sample = {
            "input_ids": prompt,
            "answers": batch["answers"]
            }
        return sample
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, remove_columns=list(datafile.features))
    return dataset


def get_paraphrased_triviaqa_dataset(dataset_config, tokenizer, partition):
    
    def create_new_questions(question):
        questions = []
        for i in range(5):
            new_question = f"""Paraphrase the question "{question}", and make sure the new question has the same and complete meaning with the original question."""
            questions.append(new_question)
        return questions

    if partition == 'train':
        filename = 'train_10000'
        # filename = 'train_138384'
    else:
        filename = 'validation_1000'
    
    def process_data(batch):  
        # find noun phrases in question
        new_samples = {"input_ids": []}
        for i, bat in enumerate(batch):
            new_question = create_new_questions(batch["question"][i])
            # print(f"new question: {new_question}")
            for q in new_question:
                # print(q)
                encoded_prompt = tokenizer.encode(tokenizer.bos_token + q, add_special_tokens=False)
                new_samples["input_ids"].append(encoded_prompt)
        return new_samples
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, batched=True, batch_size = 4, remove_columns=list(datafile.features))
    return dataset
    

def get_paraphrased_triviaqa_dataset(dataset_config, tokenizer, partition):
    
    def create_new_questions(question):
        questions = []
        for i in range(5):
            new_question = f"""Paraphrase the question "{question}", and make sure the new question has the same and complete meaning with the original question."""
            questions.append(new_question)
        return questions

    if partition == 'train':
        filename = 'train_10000'
        # filename = 'train_138384'
    else:
        filename = 'validation_1000'
    
    def process_data(batch):  
        # find noun phrases in question
        new_samples = {"input_ids": []}
        for i, bat in enumerate(batch):
            new_question = create_new_questions(batch["question"][i])
            # print(f"new question: {new_question}")
            for q in new_question:
                # print(q)
                encoded_prompt = tokenizer.encode(tokenizer.bos_token + q, add_special_tokens=False)
                new_samples["input_ids"].append(encoded_prompt)
        return new_samples
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, batched=True, batch_size = 4, remove_columns=list(datafile.features))
    return dataset


def get_triviaqa_dataset_for_reordering(dataset_config, tokenizer, partition):
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.tree import Tree

    # Ensure the necessary NLTK data is downloaded
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('punkt')
    nltk.download('wordnet')

    def get_synsets(word):
        """Get synsets for a word."""
        return wn.synsets(word)

    def is_noun(tag):
        """Check if a tag is a noun."""
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']

    def get_noun_phrases(sentence):
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        # Get POS tags for the tokens
        tagged = pos_tag(tokens)
        # Parse the sentence to identify noun phrases
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        # Find noun phrases
        noun_phrases = [' '.join(leaf[0] for leaf in t.leaves()) for t in tree if isinstance(t, Tree) and t.label() == 'NP']
        return noun_phrases
    
    def reorder_list(input_list):
        """
        Takes a list as input and generates up to 10 unique randomly reordered lists.
        """
        unique_input_len = len(set(input_list))
        max_list_num = 10 if unique_input_len > 3 else len(set(itertools.permutations(input_list)))
        unique_lists = set()
        while len(unique_lists) < max_list_num:
            # Shuffle the list and convert to tuple for set storage (lists can't be set elements)
            shuffled_list = input_list.copy()
            random.shuffle(shuffled_list)
            unique_lists.add(tuple(shuffled_list))
            # print(unique_lists)
        # Convert the tuples back to lists for output
        return [list(unique_list) for unique_list in unique_lists]
    
    def create_new_questions(question, original_list, reordered_list):
        questions = []
        for np in reordered_list:
            new_question = f"""Reorganize the question "{question}" by reordering the phrases {", ".join([f'"{p}"' for p in original_list])} into the following order: {", ".join([f'"{p}"' for p in np])}. Make sure the reordered question has the same and complete meaning with the original question."""
            questions.append(new_question)
        return questions

    if partition == 'train':
        filename = 'train_10000'
        # filename = 'train_138384'
    else:
        filename = 'validation_1000'
    
    def process_data(batch):  
        # find noun phrases in question
        new_samples = {"input_ids": []}
        noun_phrases = [get_noun_phrases(q) for q in batch["question"]]
        # print(noun_phrases)
        for i, np in enumerate(noun_phrases):
            if len(np) >= 2:
                reordered_phrases = reorder_list(np)
                # print(f"reordered list: {reordered_phrases}")
                new_question = create_new_questions(batch["question"][i], np, reordered_phrases)
                # print(f"new question: {new_question}")
                for q in new_question:
                    # print(q)
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + q, add_special_tokens=False)
                    new_samples["input_ids"].append(encoded_prompt)
            else:
                continue
            # print(new_samples["input_ids"])
            # print("------------")
        return new_samples
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, batched=True, batch_size = 4, remove_columns=list(datafile.features))
    return dataset


def get_triviaqa_dataset_for_pruning(dataset_config, tokenizer, partition):
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.tree import Tree

    # Ensure the necessary NLTK data is downloaded
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('punkt')
    nltk.download('wordnet')

    def get_noun_phrases(sentence):
        # Tokenize the sentence
        tokens = word_tokenize(sentence)
        # Get POS tags for the tokens
        tagged = pos_tag(tokens)
        # Parse the sentence to identify noun phrases
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
        cp = nltk.RegexpParser(grammar)
        tree = cp.parse(tagged)
        # Find noun phrases
        noun_phrases = [' '.join(leaf[0] for leaf in t.leaves()) for t in tree if isinstance(t, Tree) and t.label() == 'NP']
        return noun_phrases
    
    def reorder_list(input_list):
        """
        Takes a list as input and generates up to 10 unique randomly reordered lists.
        """
        unique_input_len = len(set(input_list))
        max_list_num = 10 if unique_input_len > 3 else len(set(itertools.permutations(input_list)))
        unique_lists = set()
        while len(unique_lists) < max_list_num:
            # Shuffle the list and convert to tuple for set storage (lists can't be set elements)
            shuffled_list = input_list.copy()
            random.shuffle(shuffled_list)
            unique_lists.add(tuple(shuffled_list))
            # print(unique_lists)
        # Convert the tuples back to lists for output
        return [list(unique_list) for unique_list in unique_lists]
    
    def create_new_questions(question, original_list, reordered_list):
        questions = []
        for np in reordered_list:
            new_question = f"""Reorganize the question "{question}" by reordering the phrases {", ".join([f'"{p}"' for p in original_list])} into the following order: {", ".join([f'"{p}"' for p in np])}. Make sure it has the same meaning with the original question."""
            questions.append(new_question)
        return questions

    if partition == 'train':
        filename = 'train_10000'
        # filename = 'train_138384'
    else:
        filename = 'validation_1000'
    
    def process_data(batch):  
        # find noun phrases in question
        new_samples = {"input_ids": []}
        noun_phrases = [get_noun_phrases(q) for q in batch["question"]]
        # print(noun_phrases)
        for i, np in enumerate(noun_phrases):
            if len(np) >= 2:
                reordered_phrases = reorder_list(np)
                # print(f"reordered list: {reordered_phrases}")
                new_question = create_new_questions(batch["question"][i], np, reordered_phrases)
                # print(f"new question: {new_question}")
                for q in new_question:
                    # print(q)
                    encoded_prompt = tokenizer.encode(tokenizer.bos_token + q, add_special_tokens=False)
                    new_samples["input_ids"].append(encoded_prompt)
            else:
                continue
            # print(new_samples["input_ids"])
            # print("------------")
        return new_samples
    datafile = datasets.load_from_disk(os.path.join(dataset_config.data_path, filename))
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, batched=True, batch_size = 4, remove_columns=list(datafile.features))
    return dataset


def get_second_question_for_prediction(dataset_config, tokenizer, partition):
    if partition == 'train':
        filename = 'post_modified_train_questions1.txt'
    else:
        filename = 'post_modified_val_questions1.txt'
    def process_data(batch):  
        prompt = tokenizer.encode(tokenizer.bos_token + user_prompt + batch["text"].split('\t')[1], add_special_tokens=False)
        sample = {
            "input_ids": prompt,
            }
        return sample
    datafile = datasets.load_dataset('text', data_files = os.path.join(dataset_config.data_path, filename))['train']
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, remove_columns=list(datafile.features))
    return dataset

def get_both_questions_for_correction(dataset_config, tokenizer, partition):
    if partition == 'train':
        filename = 'post_reorder_train_questions1.txt'
    else:
        filename = 'post_reorder_val_questions1.txt'
    def process_data(batch):
        question1 = batch["text"].split('\t')[0]
        question2 = batch["text"].split('\t')[1]
        prompt = tokenizer.encode(tokenizer.bos_token + f"Question 1: {question1}\n\nQuestion 2: {question2}\n\nSlightly modify Question 2 to have similar meaning with Question 1 but different wordings:", add_special_tokens=False)
        sample = {
            "input_ids": prompt,
            }
        return sample
    datafile = datasets.load_dataset('text', data_files = os.path.join(dataset_config.data_path, filename))['train']
    # print(f'first data piece: {datafile[0]}\n{datafile[1]}\n{datafile[2]}\n{datafile[3]}')
    dataset = datafile.map(process_data, remove_columns=list(datafile.features))
    return dataset
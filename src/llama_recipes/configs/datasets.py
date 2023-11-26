# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class triviaqa_dataset:
    dataset: str = "triviaqa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/shared/data2/jiaxinh3/Calibration/LM/data/trivia_qa"
    user_prompt_path: str = "/shared/data2/jiaxinh3/Calibration/LM/data/prompts/direct_answer.txt"

@dataclass
class triviaqa_dataset_reorder:
    dataset: str = "triviaqa_dataset_reorder"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/shared/data2/jiaxinh3/Calibration/LM/generated/trivia_qa"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
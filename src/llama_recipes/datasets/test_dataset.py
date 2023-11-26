from llama_recipes.datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_triviaqa_dataset,
    get_triviaqa_dataset_for_prediction,
    get_triviaqa_dataset_for_reordering,
    get_second_question_for_prediction,
    get_both_questions_for_correction
)
model_name = 'meta-llama/Llama-2-7b-chat-hf'
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
import inspect
from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
dataset_config = {k:v for k, v in inspect.getmembers(datasets)}['triviaqa_dataset_reorder']()
train_data = get_both_questions_for_correction(dataset_config, tokenizer, 'val')
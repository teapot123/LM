python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
    --batch_size 4 \
    --prompt_file ../data/trivia_qa/validation_1000_verb_1s_top1.json \
    --system_prompt_file ../data/prompts/system_prompt.txt \
    --output_file ../generated/trivia_qa/gen_val_1000_verb_1s_top1.txt 
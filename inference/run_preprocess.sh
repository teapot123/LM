mkdir ../data/trivia_qa
python process_trivia_qa.py --down_sample True --data_split validation --select_data_num 1000

huggingface-cli login --token "$(< ../t.txt)"
python process_trivia_qa.py --data_dir ../data/trivia_qa/validation_1000 --user_prompt_file ../data/prompts/verb_1s_top1.txt --system_prompt_file ../data/prompts/system_prompt.txt
mkdir ../data/trivia_qa
python process_trivia_qa.py --down_sample True --data_split validation --select_data_num 1000
python process_trivia_qa.py --data_dir ../data/trivia_qa/validation_1000 --user_prompt_file ../data/prompts/verb_1s_top1.txt
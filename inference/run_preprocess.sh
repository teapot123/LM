#mkdir ../data/trivia_qa
# python process_trivia_qa.py --down_sample True --data_split validation --select_data_num 1000

# huggingface-cli login --token "$(< ../t.txt)"
python process_trivia_qa.py --data_dir ../data/trivia_qa/validation_1000 --user_prompt_file ../data/prompts/rec_1s_top1.txt --system_prompt_file ../data/prompts/system_prompt.txt --mode topk

# generate recitation for each question
# python process_trivia_qa.py --data_dir ../data/trivia_qa/validation_1000 --user_prompt_file ../data/trivia_qa/recitation_prompt.txt --system_prompt_file ../data/prompts/system_prompt.txt --mode generate_recitation

# generate answer and conf with recitation
# python process_trivia_qa.py --data_dir ../generated/trivia_qa/validation_1000_1recitation_0.3.txt --user_prompt_file ../data/prompts/verb_1s_top1.txt --system_prompt_file ../data/prompts/system_prompt.txt --mode with_recitation --out_dir ../data/trivia_qa/validation_1000_w_recitation.json

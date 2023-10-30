# topk

TEMP=0.3
python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
	--quantization True --batch_size 8 --temperature ${TEMP} \
	--prompt_file ../data/trivia_qa/validation_1000_verb_1s_top1.json \
	--output_file ../generated/trivia_qa/tmp.txt

# generate recitation

# TEMP=0.3
# python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
# 	--mode generate_recitation \
# 	--quantization True --batch_size 8 --temperature ${TEMP} \
# 	--prompt_file ../data/trivia_qa/validation_1000_0recitation.json \
# 	--output_file ../generated/trivia_qa/validation_1000_1recitation_${TEMP}.txt


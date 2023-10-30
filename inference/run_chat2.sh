# topk

TEMP=0.3
python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
	--quantization True --batch_size 4 --temperature ${TEMP} \
    --max_new_tokens 512 \
	--prompt_file ../data/trivia_qa/validation_1000_rec_1s_top4.json \
	--output_file ../generated/trivia_qa/gen_val_1000_rec_1s_top4_${TEMP}.txt

# generate recitation

# TEMP=0.3
# python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
# 	--mode generate_recitation \
# 	--quantization True --batch_size 8 --temperature ${TEMP} \
# 	--prompt_file ../data/trivia_qa/validation_1000_0recitation.json \
# 	--output_file ../generated/trivia_qa/validation_1000_1recitation_${TEMP}.txt


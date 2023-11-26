# topk

# TEMP=0.3
# python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
# 	--quantization True --batch_size 4 --temperature ${TEMP} \
#     --max_new_tokens 512 \
# 	--prompt_file ../data/trivia_qa/validation_1000_rec_1s_top4.json \
# 	--output_file ../generated/trivia_qa/gen_val_1000_rec_1s_top4_${TEMP}.txt

# generate recitation

# TEMP=0.3
# python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
# 	--mode generate_recitation \
# 	--quantization True --batch_size 8 --temperature ${TEMP} \
# 	--prompt_file ../data/trivia_qa/validation_1000_0recitation.json \
# 	--output_file ../generated/trivia_qa/validation_1000_1recitation_${TEMP}.txt


python examples/chat_completion/chat_completion.py --model_name meta-llama/Llama-2-70b-chat-hf \
	--batch_size 4 --do_sample False --quantization\
	--output_file generated/trivia_qa/top1_pretrained70b.txt \
	--print_logits --logits_file generated/trivia_qa/top1_pretrained70b_logits.txt \
	--read_data_func get_triviaqa_dataset_for_prediction \

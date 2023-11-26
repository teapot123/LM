
# pre-trained model
# TEMP=0.7
# python examples/chat_completion/direct_inference.py --model_name meta-llama/Llama-2-7b-chat-hf \
# 	--batch_size 8 --do_sample False  \
# 	--output_file generated/trivia_qa/modified_reorder_val_questions1.txt \
#     --partition validation

# python examples/chat_completion/direct_inference.py --model_name meta-llama/Llama-2-7b-chat-hf \
# 	--batch_size 8 --do_sample True --temperature ${TEMP} \
# 	--output_file generated/trivia_qa/reorder_train_questions1.txt \
#     --partition train

#fine-tuned model with peft
while true; do
	python examples/chat_completion/direct_inference.py --model_name meta-llama/Llama-2-7b-chat-hf \
		--peft_model output/lora_fixmlp_bz64 --batch_size 8 --do_sample False --partition validation\
		--output_file generated/trivia_qa/final_outputs/top1_lora_fixmlp_7b_w_conf.txt \
		--read_data_func get_triviaqa_dataset_for_prediction_with_conf
done

# post processing the re-ordered sentences
# python examples/chat_completion/post_process_reordered_sentences.py \
#     --dataset trivia_qa \
# 	--output_file generated/trivia_qa/post_modified_val_questions1.txt \
#     --input_file generated/trivia_qa/modified_reorder_val_questions1.txt


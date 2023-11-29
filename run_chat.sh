# topk
# chat mode
# TEMP=0.5
# python examples/chat_completion/chat_completion.py --model_name meta-llama/Llama-2-7b-chat-hf \
# 	--use_chat_mode True \
# 	--quantization True --batch_size 8 \
# 	--prompt_file data/trivia_qa/validation_1000_verb_1s_top1.json \
# 	--output_file generated/trivia_qa/tmp.txt


# inference mode
# pre-trained model
# TEMP=0.3
python examples/chat_completion/chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
	--batch_size 8 --do_sample False \
	--output_file generated/trivia_qa/top1_pretrained13b_w_conf.txt \
	--print_logits --logits_file generated/trivia_qa/top1_pretrained13b_w_conf_logits.txt \
	--read_data_func get_triviaqa_dataset_for_prediction \



# prediction on re-ordered sentences

# TEMP=0.3
# python examples/chat_completion/chat_completion.py --model_name meta-llama/Llama-2-7b-chat-hf \
# 	--batch_size 4 --do_sample False \
# 	--output_file generated/trivia_qa/modifyQ1_val_top1_pretrained7b.txt \
# 	--print_attention --att_file generated/trivia_qa/modifyQ1_val_top1_pretrained7b_att.txt \
# 	--partition validation \
# 	--read_data_func get_second_question_for_prediction \


# full fine-tuned model	
# python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path ./ --consolidated_model_path ./converted/ --HF_model_path_or_name meta-llama/Llama-2-7b-chat-hf

# python examples/chat_completion/chat_completion.py --model_name PATH/to/save/FSDP/model/fine-tuned-meta-llama/Llama-2-7b-chat-hf/converted \
# 	--batch_size 8 --do_sample True --temperature ${TEMP} \
# 	--output_file generated/trivia_qa/direct_answer_fft7b.txt \


#fine-tuned model with peft
# TEMP=0.3
# python examples/chat_completion/chat_completion.py --model_name meta-llama/Llama-2-7b-chat-hf \
# 	--peft_model output/lora_fixmlp_bz64 --batch_size 8 --do_sample True \
# 	--output_file generated/trivia_qa/top1_lora_fixmlp_7b.txt \
# 	--temperature ${TEMP} --print_attention --att_file generated/trivia_qa/top1_lora_fixmlp_7b_att.txt 






# generate recitation

# TEMP=0.3
# python chat_completion.py --model_name meta-llama/Llama-2-13b-chat-hf \
# 	--mode generate_recitation \
# 	--quantization True --batch_size 8 --temperature ${TEMP} \
# 	--prompt_file ../data/trivia_qa/validation_1000_0recitation.json \
# 	--output_file ../generated/trivia_qa/validation_1000_1recitation_${TEMP}.txt


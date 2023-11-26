#use lora
# torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/top1_lora_bz64 --batch_size_training 64

# use full finetuning
# torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/fft_whole_triviaqa_bz16 --batch_size_training 16

# use lora fix mlp
# torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --fix_mlp --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/lora_fixmlp_bz64 --batch_size_training 64 --num_epochs 3



export CUDA_VISIBLE_DEVICES=5,6


while true; do
    torchrun --nnodes 1 --nproc_per_node 2  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/finetune_whole_triviaqa2 --batch_size_training 16
done
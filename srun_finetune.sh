#!/bin/bash
# SLURM Resource Parameters
sbatch -N 1                     # number of nodes
sbatch --ntasks-per-node=1      # Number of commands that can be managed by SLURM at the same time.
#sbatch --cpus-per-task=<CPUs>   # number of CPUs required per task, since we always run a single task, this
                                 # is the total CPUs that get allocated.
#sbatch -t 0-00:05               # Runtime in D-HH:MM
sbatch -p gpuA100x4             # Partition to submit to. titan-gpu/dgx1-gpu/a100-gpu-shared
sbatch --gres=gpu:4             # Number of gpus
#sbatch --mem=100                # Memory pool for all cores in MBs (see also --mem-per-cpu)
sbatch --job-name=testing_finetune
#sbatch -o job_%j.out            # File to which STDOUT will be written. %j is job id
#sbatch -e job_%j.err            # File to which STDERR will be written
sbatch --mail-type=BEGIN          # Type of email notification- BEGIN,END,FAIL,ALL
sbatch --mail-user=jiaxinh3@illinois.com  # Email to which notifications will be sent

# After requested resources are allocated, run your program (for example in a docker container)
srun --account bcaq nvidia-docker run -v /home/jhuang6:/home/jhuang6 deep_learning python torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/test_model



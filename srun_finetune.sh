!/bin/bash
# SLURM Resource Parameters
#SBATCH -N 1                     # number of nodes
#SBATCH --ntasks-per-node=1      # Number of commands that can be managed by SLURM at the same time.
#SBATCH --cpus-per-task=<CPUs>   # number of CPUs required per task, since we always run a single task, this
                                 # is the total CPUs that get allocated.
#SBATCH -t 0-00:05               # Runtime in D-HH:MM
#SBATCH -p titan-gpu             # Partition to submit to. titan-gpu/dgx1-gpu/a100-gpu-shared
#SABTCH --gres=gpu:2             # Number of gpus
#SBATCH --mem=100                # Memory pool for all cores in MBs (see also --mem-per-cpu)
#SBATCH --job-name=testing_slurm
#SBATCH -o job_%j.out            # File to which STDOUT will be written. %j is job id
#SBATCH -e job_%j.err            # File to which STDERR will be written
#SBATCH --mail-type=END          # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ajk@123.com  # Email to which notifications will be sent

# After requested resources are allocated, run your program (for example in a docker container)
srun nvidia-docker run -v /home/jhuang6:/home/jhuang6 deep_learning python torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name meta-llama/Llama-2-7b-chat-hf --fsdp_config.pure_bf16 --output_dir output/test_model



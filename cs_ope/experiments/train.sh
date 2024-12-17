#!/bin/bash
#SBATCH --job-name=Atari_para
#SBATCH --partition=gpulong  
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=40000M
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --account=ailab    


source /home/tnguye11/anaconda3/bin/activate cs_ope

module load cuda/12.4

commands=(
    # "python experiment_learning.py --preset=heart_disease --num_trials=10 --ration_size=0.7 "
    "python experiment_learning.py --preset=heart_disease --num_trials=10 --ration_size=0.4 "
    # "python experiment_learning.py --preset=satimage --num_trials=10 --ration_size=0.05"
)


for i in "${!commands[@]}"; do
   srun --exclusive -N1 -n1 ${commands[i]} &
done

wait

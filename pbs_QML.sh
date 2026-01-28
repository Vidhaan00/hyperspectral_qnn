#PBS -N HSI_QML
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l place=scatter:shared
#PBS -q gpu
#PBS -m abe
#PBS -o output.log
#PBS -e error.log

export CUDA_VISIBLE_DEVICES=0


cd ${PBS_O_WORKDIR}

# Initialize Conda
source /apps/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate deeplearning

date

python3 main.py

date


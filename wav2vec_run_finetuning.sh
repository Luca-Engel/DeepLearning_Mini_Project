#!/bin/bash -l
#SBATCH --chdir /scratch/izar/dkopp/DeepLearning_Project
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 40G
#SBATCH --time 02:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552

date
module load gcc python


source ~/venvs/dl_project/bin/activate

pip install --upgrade pip
pip3 install -r ./requirements.txt

pip3 install torchaudio


pip3 install evaluate
pip3 install datasets
pip3 install accelerate -U

echo "Install Dependencies complete"
echo "Running the code"

python3 ./wav2vec_cluster_run.py


echo "Run completed on $(hostname)"
sleep 2
date
#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/ainlp22/learning-nlp/
#SBATCH -t 0-00:08 # Runtime in D-HH:MM
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 4096 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o /hhome/ainlp22/learning-nlp/hello.out # File to which STDOUT will be written
#SBATCH -e /hhome/ainlp22/learning-nlp/error.err # File to which STDERR will be written

python3 asho_project.py

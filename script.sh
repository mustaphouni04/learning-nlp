#!/bin/bash
#SBATCH -n 4                       # Number of CPU cores
#SBATCH -N 1                       # All cores on one node
#SBATCH -D /hhome/ainlp22/learning-nlp/
#SBATCH -t 0-05:00                 # Runtime in D-HH:MM
#SBATCH -p dcca40                  # Partition
#SBATCH --mem 40096                # Memory in MB
#SBATCH --gres gpu:1               # 1 GPU
#SBATCH -o /hhome/ainlp22/learning-nlp/hello.out
#SBATCH -e /hhome/ainlp22/learning-nlp/error.err

python3 asho_project.py

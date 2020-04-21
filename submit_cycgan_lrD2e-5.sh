#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J lrD_2e-5
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
# request 8GB of system-memory
#BSUB -R "rusage[mem=8GB]"
#BSUB -u s182154@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
#BSUB -o ../log/log-%J-%I.out
#BSUB -e ../log/log-%J-%I.err
# -- end of LSF options --

###sh setup-python3.sh
module load python3/3.7.6
module load cuda/10.0
module load cudnn/v7.6.5.32-prod-cuda-10.0
pip3 install --user tensorflow-gpu==2.0.0
pip3 install --user tensorboardX
pip3 install --user matplotlib
pip3 install --user acoustics
pip3 install --user progressbar
pip3 install --user tensorflow-addons==0.6.0 
pip3 install --user pydot
pip3 install --user graphviz



python3 train_cyclegan.py train ../working_cyclegan/lrD_lower_new_data --data_dir ../working_cyclegan/data --data_slice_len 65536 --data_sample_rate 16000 --total_epochs 100 --sample_interval 1 --save_interval 1 --D_learning_rate 0.00002 --loadIRs


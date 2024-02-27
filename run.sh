#!/bin/bash
#SBATCH --job-name="contrast"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=2
##SBATCH --mem=32G
#SBATCH --time=120:15:00
###SBATCH --nodelist=eris
#SBATCH --mem-per-gpu=32G

source /etc/profile.d/conda.sh
source /etc/profile.d/cuda.sh

conda activate ECAPA

# train a model :
#python run.py -m train --cfg cfg/experiment_settings.cfg --checkpoint 0
#python run.py -m test --cfg cfg/contrastive_vox.cfg --checkpoint 0
# python run.py -m test --cfg cfg/contrastive.cfg --checkpoint 875
# python run.py -m train --cfg cfg/resnet.cfg --checkpoint 0
# test a model :
#python run.py -m train --cfg cfg/resnet.cfg --checkpoint 98200

#python run.py -m test --cfg cfg/resnet.cfg --checkpoint 3475

#python trainRESNETModel.py --lr 0.000171 --save_path exps/exp2 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp1/model/pretrain.model

#python trainRESNETModel.py --save_path exps/exp2 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/

#python trainRESNETModel.py --lr 0.000166 --save_path exps/exp4 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp3/model/pretrain.model

#python trainRESNETModel.py --lr 0.000033 --save_path exps/exp5 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp4/model/pretrain.model

#python trainRESNETModel.py --lr 0.000033 --save_path exps/exp5 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp5/model/pretrain.model --eval

python trainRESNETModel.py --save_path exps/exp6 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp6/model/pretrain.model

#python trainRESNETModel.py  --save_path exps/exp_test_norm --lr 0.000694 --train_list ./lists/train_list.txt  --train_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/voxceleb2/ --eval_list ./lists/veri_test2.txt --eval_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/TEST/wav/ --musan_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/musan_split --rir_path /local_disk/atlantia/laboinfo/matrouf/VOXCELEB_2023/voxceleb_trainer/data/RIRS_NOISES/simulated_rirs/ --initial_model exps/exp_test_norm/model/pretrain.model


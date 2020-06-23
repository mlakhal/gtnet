'''Novel-View Human Action Synthesis

'''

#!/bin/bash

python train.py \
   --save_path='../data/models/GTNet' \
   --dataset_dict_path='data/ntu_nvs.json' \
   --pix_dir='../data/pix' \
   --flow_dir='../data/flow' \
   --dep_dir='../data/dep' \
   --seg_dir='../data/seg' \
   --mask_dir='../data/faces' \
   --logs_dir='logs/GTNet' \
   --model_name='GTNet' \
   --image_width=112 \
   --image_height=112 \
   --train_batch_size=1 \
   --num_workers=8 \
   --max_epoch=50 \
   --start_epoch=0 \
   --timesteps=8 \
   --split='train' \
   --save_every=5 \
   --print_every=5 \
   --log_every=1 \
   --dataset_name='NTU'

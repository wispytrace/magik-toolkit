#!/usr/bin/env bash

model_root_path="./models/4bit_20220627_clip2.0"
#640_face_4bit_new"
log_dir="$model_root_path/logs"
log="$log_dir/log"


python3 -u train.py \
  --datasets \
  /home/ssd/sfyan/data/wider_face_add_lm_10_10 \
  --validation_dataset \
  /home/ssd/sfyan/data/wider_face_add_lm_10_10 \
  --net \
  slim \
  --num_epochs \
  210 \
  --milestones \
  "90,150,190" \
  --lr \
  1e-2 \
  --batch_size \
  24 \
  --input_size \
  320 \
  --checkpoint_folder \
  ${model_root_path} \
  --log_dir \
  ${log_dir} \
  --num_workers \
  4 \
  --cuda_index \
  0 \
  #--pretrained_ssd \
  #"./models/sfyan_8bit_1126/slim-Epoch-208-Loss-3.0398501616937144.pth" \
 
 
  
  # 2>&1 | tee "$log"





  

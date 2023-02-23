#singlel_gpu_train
GPUS=1
PORT=60032
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT train.py \
    --data data/coco.yaml \
    --cfg models/yolov5.yaml \
    --weights '' \
    --batch-size 64 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/yolov5n \
    --epochs 300  \
    --img-size 416 \
    --device 0 \
    #--adam \

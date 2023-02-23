export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
GPUS=6

#32bit:
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
    --data data/coco-person.yaml\
    --cfg models/yolov5s.yaml \
    --weights ' ' \
    --batch-size 132 \
    --hyp data/hyp.scratch.yaml \
    --project ./runs/train/yolov5s-person-32bit \
    --epochs 300 \
    --device 0,1,2,3,4,5


#8bit:
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights './runs/train/yolov5s-person-32bit/weights/best.pt' \
#     --batch-size 132 \
#     --hyp data/hyp.scratch-8bit.yaml \
#     --project ./runs/train/yolov5s-person-8bit \
#     --epochs 300 \
#     --device 0,1,2,3,4,5


#4bit:
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=60051 train.py \
#     --data data/coco-person.yaml\
#     --cfg models/yolov5s.yaml \
#     --weights './runs/train/yolov5s-person-8bit/weights/best.pt' \
#     --batch-size 132 \
#     --hyp data/hyp.scratch-4bit.yaml \
#     --project ./runs/train/yolov5s-person-4bit \
#     --epochs 300 \
#     --device 0,1,2,3,4,5
#     --adam

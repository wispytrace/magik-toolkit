#multi_gpu_train
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=ens

NODES=2
GPUS=4
IP_ADDR="193.169.1.211"
PORT=60050
NODE_RANK=0
PER_GPU_BATCH_SIZE=8
BATCH_SIZE=$(($PER_GPU_BATCH_SIZE*$GPUS*$NODES))


python3 -m torch.distributed.launch --nnodes=$NODES --nproc_per_node=$GPUS --node_rank=$NODE_RANK --master_addr=$IP_ADDR --master_port=$PORT train.py \
    --data data/coco.yaml \
    --cfg models/yolov5.yaml \
    --weights '' \
    --batch-size $BATCH_SIZE \
    --hyp data/hyp.lowbit.yaml \
    --project ./runs/train/yolov5n \
    --epochs 300  \
    --img-size 416 \
    #--adam \

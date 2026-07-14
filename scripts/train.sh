swanlab login -k "Your Swanlab Key" || { echo "Swanlab login failed,"; }

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_NET_GDR_LEVEL=SYS
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1
#export TOKENIZERS_PARALLELISM=false
#NUM_PROCESSES=$((${PET_NNODES} * ${PET_NPROC_PER_NODE}))
#accelerate launch --config_file=configs/accelerate_config.yaml \
#  --main_process_ip=${PET_MASTER_ADDR} \
#  --main_process_port=${PET_MASTER_PORT} \
#  --num_machines=${PET_NNODES} \
#  --num_processes=${NUM_PROCESSES} \
#  --machine_rank=${PET_NODE_RANK} \
#  train.py --config configs/train_config.yaml

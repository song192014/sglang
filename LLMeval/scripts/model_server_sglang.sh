# 确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true

# 临时关闭aclnn，避免host oom
export DISABLE_L2_CACHE=1
# 降低内存碎片，避免不稳定oom
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

# CANN
install_path = /usr/local/Ascend
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

# 服务器基础配置信息
model_path="/home/mdoel"
model_name="Qwen/Qwen25-32B"
num_gpus=8
max_model_len=131072
mem_fraction_static=0.8


# sglang在线服务启动命令
python -m sglang.launch_server \
    --model $model_path \
    --served-model-name $model_name \
    --trust-remote-code \
    --device npu \
    --attention-backend ascend \
    --sampling-backend ascned \
    --tensor-parallel-size $num_gpus \
    --mem-fraction-static $mem_fraction_static \
    --context-length $max_model_len \
    --port 8090 \
    --tp-size 8 \
    --schedule-conservativeness 2 \
    --cuda-graph-max-bs 64 \
    --disable-radix-cache \
    2>&1 | tee log_sglang_server.log

    # 关闭图模式
    # --disable-cuda-graph \
    # 2>&1 | tee log_sglang_server_nograph.log
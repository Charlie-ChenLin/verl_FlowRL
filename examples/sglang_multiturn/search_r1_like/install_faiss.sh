# 1) 备份当前环境（克隆一份）
conda create -n verl050_faiss_clone --clone verl050

# 2) 激活克隆环境
conda activate verl050_faiss_clone

# 3) 卸载旧 faiss（避免混用）
pip uninstall -y faiss-gpu-cu12 faiss-gpu faiss-cpu faiss || true

# 4) 编译安装带 sm90 的 faiss-gpu
git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.9.0  # 或你信任的版本
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(which python)
cmake --build build -j 16
pip install --no-build-isolation build/faiss/python

# 5) 验证
python - <<'PY'
import faiss, torch
print("faiss GPUs:", faiss.get_num_gpus())
print("torch cuda:", torch.randn(1, device="cuda").device)
PY

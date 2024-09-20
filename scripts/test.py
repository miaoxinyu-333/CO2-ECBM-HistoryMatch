import torch

# 查看 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 查看 CUDA 版本
print("CUDA version:", torch.version.cuda)

# 检查 CUDA 是否可用
print("Is CUDA available?", torch.cuda.is_available())

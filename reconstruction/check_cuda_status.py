# import torch
# print('PyTorch版本:', torch.__version__)
# print('CUDA版本:', torch.version.cuda)
# print('GPU是否可用:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     print('GPU设备名称:', torch.cuda.get_device_name(0))
#     print('CUDA能力:', torch.cuda.get_device_capability(0))

import torch
# 1. PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 2. PyTorch 内置的 CUDA 版本（核心！是 PyTorch 编译时用的 CUDA 版本）
print(f"PyTorch 编译的 CUDA 版本: {torch.version.cuda}")

# 3. 系统安装的 CUDA 版本（可选，仅参考）
print(f"系统 CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 算力: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch 支持的 GPU 算力列表: {torch.cuda.get_arch_list()}")


import os

# 设置环境变量以启用CUDA兼容性
os.environ['CUDA_DISABLE_COMPATIBILITY_CHECK'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 现在导入rtx50_compat和torch
import rtx50_compat
import torch

print('设置环境变量后测试：')
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('GPU是否可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU设备名称:', torch.cuda.get_device_name(0))
    print('CUDA能力:', torch.cuda.get_device_capability(0))
    
    # 测试移动更大的张量
    print('\n测试移动更大的张量...')
    try:
        test_tensor = torch.randn(1, 1, 64, 64, 64)
        print(f'测试张量形状: {test_tensor.shape}')
        test_tensor_gpu = test_tensor.to('cuda')
        print('✅ 大张量成功移动到GPU')
        print(f'GPU张量形状: {test_tensor_gpu.shape}')
    except Exception as e:
        print(f'❌ 错误: {e}')
        import traceback
        traceback.print_exc()
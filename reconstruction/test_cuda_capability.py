import os

# 设置环境变量以启用CUDA兼容性
os.environ['CUDA_DISABLE_COMPATIBILITY_CHECK'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_CAPABILITY_NAME'] = 'sm_90'  # 欺骗PyTorch使用sm_90

# 现在导入rtx50_compat和torch
import rtx50_compat
import torch

print('设置CUDA_CAPABILITY_NAME后测试：')
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('GPU是否可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU设备名称:', torch.cuda.get_device_name(0))
    print('CUDA能力:', torch.cuda.get_device_capability(0))
    
    # 测试在GPU上执行操作
    print('\n测试在GPU上执行操作...')
    try:
        test_tensor = torch.randn(1, 1, 64, 64, 64)
        test_tensor_gpu = test_tensor.to('cuda')
        result = test_tensor_gpu / 255.0
        print('✅ 在GPU上执行除法操作成功')
        print(f'结果形状: {result.shape}')
    except Exception as e:
        print(f'❌ 在GPU上执行操作失败: {e}')
        import traceback
        traceback.print_exc()
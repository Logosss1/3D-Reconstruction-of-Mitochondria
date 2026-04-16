import os

# 设置环境变量以启用CUDA兼容性
os.environ['CUDA_DISABLE_COMPATIBILITY_CHECK'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 现在导入rtx50_compat和torch
import rtx50_compat
import torch

print('测试在GPU上执行操作...')
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('GPU是否可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU设备名称:', torch.cuda.get_device_name(0))
    print('CUDA能力:', torch.cuda.get_device_capability(0))
    
    # 测试1: 移动张量到GPU
    print('\n测试1: 移动张量到GPU')
    try:
        test_tensor = torch.randn(1, 1, 64, 64, 64)
        test_tensor_gpu = test_tensor.to('cuda')
        print('✅ 大张量成功移动到GPU')
    except Exception as e:
        print(f'❌ 移动张量失败: {e}')
    
    # 测试2: 在GPU上执行简单操作
    print('\n测试2: 在GPU上执行简单操作')
    try:
        test_tensor = torch.randn(1, 1, 64, 64, 64)
        test_tensor_gpu = test_tensor.to('cuda')
        result = test_tensor_gpu / 255.0
        print('✅ 在GPU上执行除法操作成功')
    except Exception as e:
        print(f'❌ 在GPU上执行操作失败: {e}')
    
    # 测试3: 在GPU上执行模型操作
    print('\n测试3: 在GPU上执行模型操作')
    try:
        from src.model import ConvONet
        model = ConvONet().to('cuda')
        input_tensor = torch.randn(1, 1, 64, 64, 64).to('cuda')
        points_tensor = torch.rand(1, 1000, 3).to('cuda')
        output = model(input_tensor, points_tensor)
        print('✅ 在GPU上执行模型前向传播成功')
        print(f'模型输出形状: {output.shape}')
    except Exception as e:
        print(f'❌ 在GPU上执行模型操作失败: {e}')
        import traceback
        traceback.print_exc()
import torch
print(torch.__version__)           # 检查版本，应该带有 cu124 或类似字样
print(torch.cuda.is_available())   # 应该是 True
print(torch.cuda.get_device_name(0)) # 应该显示 RTX 5080 ...

# 测试生成一个张量
x = torch.rand(5, 3).cuda()
print(x)
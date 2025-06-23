"""
测试torch.cat方法维度合并问题

结论：dim维度变化（合并），其他维度不变
"""


import torch

torch.manual_seed(42)

tensor1=torch.randn((3,4))
tensor2=torch.randn((3,4))

print(tensor1)
print(tensor2)
tensors=[tensor1,tensor2] #要合并的tensor元组或列表
dim0_res_tensor=torch.cat(tensors=tensors,dim=0)
print(dim0_res_tensor.shape) #torch.Size([6, 4])
print(dim0_res_tensor)

dim1_res_tensor=torch.cat(tensors=tensors,dim=1)
print(dim1_res_tensor.shape) #torch.Size([3, 8])
print(dim1_res_tensor)


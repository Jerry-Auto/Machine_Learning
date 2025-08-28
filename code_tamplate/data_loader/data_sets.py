from torch.utils.data import Dataset
import torch

""" 
自定义数据集,继承自Dataset,重写getitem和len方法 
    __init__:初始化数据路径、标签、预处理方法等。    
    __len__返回数据集样本总数。
    __getitem__根据索引返回单个样本(数据 + 标签）。
"""
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data          # 数据列表/数组
        self.labels = labels      # 标签列表/数组
        self.transform = transform  # 预处理函数（如torchvision.transforms）

    def __len__(self):
        return len(self.data)     # 必须返回整数

    def __getitem__(self, idx):
        x = self.data[idx]        # 获取样本
        y = self.labels[idx]      # 获取标签
        if self.transform:        # 应用预处理
            x = self.transform(x)
        return x, y               # 返回（数据，标签）元组

class CovidDataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)
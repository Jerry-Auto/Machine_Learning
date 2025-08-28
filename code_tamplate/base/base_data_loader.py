import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

""" 
初始化机会进行分割，同时返回训练集的DataLoader对象，
使用split_validation方法返回测试集的DataLoader对象 
"""
class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    validation_split:验证集比例
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        p_m=False
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory':p_m
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0: # 无需验证集
            return None, None

        idx_full = np.arange(self.n_samples)
        np.random.seed(0)# 固定随机种子确保可复现
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split   # 直接指定验证集样本数
        else:
            len_valid = int(self.n_samples * split) # 按比例计算

        valid_idx = idx_full[0:len_valid]  # 验证集索引
        train_idx = np.delete(idx_full, np.arange(0, len_valid)) # 训练集索引

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False  # 使用sampler后必须关闭shuffle
        self.n_samples = len(train_idx) # 更新训练集大小

        # 返回训练集和测试集
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None 
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

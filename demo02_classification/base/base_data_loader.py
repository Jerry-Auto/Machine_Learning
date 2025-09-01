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
    def __init__(self, dataset, batch_size, shuffle, K_folds, num_workers, 
                collate_fn=default_collate, sampler=None):  # 添加sampler参数
        self.shuffle = shuffle
        self.origin_shuffle = shuffle
        self.origin_dataset = dataset
        self.batch_idx = 0
        self.n_samples = len(dataset)
        p_m = False
        
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': p_m}
        
        if sampler is not None:
            # 如果提供了sampler，使用sampler模式
            self.init_kwargs['sampler'] = sampler
            self.init_kwargs['shuffle'] = False  # 使用sampler时必须关闭shuffle
        elif K_folds == 0:
            self.validation_split = 0.0
            self.init_kwargs['shuffle'] = self.origin_shuffle
        else:
            self.validation_split = 1 / K_folds
            self.sampler, self.valid_sampler = self._split_sampler()
            self.init_kwargs['sampler'] = self.sampler
            self.init_kwargs['shuffle'] = False    
        super().__init__(**self.init_kwargs)


    def _split_sampler(self,K_th_fold=None):

        if self.validation_split == 0.0: # 无需验证集
            return None, None
        if K_th_fold==None:
            K_th_fold=0
        idx_full = np.arange(self.n_samples)
        np.random.seed(0)# 固定随机种子确保可复现
        np.random.shuffle(idx_full)

        if isinstance(self.validation_split, int):
            assert self.validation_split > 0
            assert self.validation_split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = self.validation_split   # 直接指定验证集样本数
        else:
            len_valid = int(self.n_samples * self.validation_split) # 按比例计算
        # [num_valid_samples * fold:num_valid_samples * (fold + 1)]
        valid_idx = idx_full[len_valid*K_th_fold:len_valid*(K_th_fold+1)]  # 验证集索引
        train_idx = np.delete(idx_full, np.arange(len_valid*K_th_fold,len_valid*(K_th_fold+1))) # 训练集索引

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False  # 使用sampler后必须关闭shuffle
        # self.n_samples = len(train_idx) # 更新训练集大小

        # 返回训练集和测试集
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None 
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
  
    def K_fold_split(self,K_th_fold):
        """返回验证集和训练集的DataLoader"""
        # 获取当前fold的采样器
        train_sampler, valid_sampler = self._split_sampler(K_th_fold)
        # 创建新的DataLoader实例
        valid_loader = BaseDataLoader(
            dataset=self.origin_dataset,
            batch_size=self.init_kwargs['batch_size'],
            shuffle=False,  # 使用sampler时必须关闭shuffle
            K_folds=0,      # 关闭后续K折划分
            num_workers=self.init_kwargs['num_workers'],
            sampler=valid_sampler
        )
        train_loader = BaseDataLoader(
            dataset=self.origin_dataset,
            batch_size=self.init_kwargs['batch_size'],
            shuffle=False,
            K_folds=0,
            num_workers=self.init_kwargs['num_workers'],
            sampler=train_sampler
        )
        return valid_loader, train_loader  # 更符合直觉的返回顺序
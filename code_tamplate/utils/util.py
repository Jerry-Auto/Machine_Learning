import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from sklearn.feature_selection import SelectKBest, f_regression
	

# 递归创建目录（若不存在）
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
	

# 读取JSON文件并保持键顺序（返回OrderedDict），加载配置文件
def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
    
# 将数据写入JSON文件（保留缩进和原始键序），保存实验配置或结果
def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
	
# 创建无限循环的数据加载器
def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.average[key]
    
    #返回所有指标的均值
    def result(self):
        return dict(self._data.average)
    
    
    #使用模型预测未知的测试集
    def predict(test_loader, model, device):
        model.eval() # Set your model to evaluation mode.
        preds = []
        for x in tqdm(test_loader):
            x = x.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        return preds
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from abc import abstractmethod
""" 
基类，实现任意数据读取，自定义数据读取格式，自定义特征选取逻辑
 """
class BaseDataReader:
    """通用数据读取与特征选择基类（无数据划分）"""
    
    def __init__(self):
        self._raw_data = None
        self._feature_data = None
        
    def load_data(self):
        """加载并返回完整数据集"""
        self._raw_data = self._read_raw_data()
        self._feature_data=self._select_features()
        return self._feature_data
        # return self._preprocess(self._feature_data)
    
    @abstractmethod
    def _read_raw_data(self):
        """子类必须实现的具体数据读取逻辑"""
        raise NotImplementedError
    
    def _preprocess(self, data):
        """数据预处理钩子（可选重载）"""
        return data  # 默认不做处理

    @abstractmethod
    def _select_features(self):
        """子类必须实现的具体特征选择逻辑"""
        raise NotImplementedError


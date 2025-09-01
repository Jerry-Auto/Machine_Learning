import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from base import BaseDataReader


class CSVDataReader(BaseDataReader):
    """CSV数据读取与特征选择实现类
    
    功能：
    1. 自动识别表头
    2. 支持分类/回归任务的特征选择
    3. 保留特征名称元数据
    4. 支持保存/加载特征选择结果
    """

    def __init__(
        self,
        data_path: str,
        label_col: Optional[str] = None,
        header: bool = True,
        feature_select_mode: str = 'regression',
        k_features: Optional[int] = None
    ):
        """
        :param label_col: 标签列名（None表示无监督）
        :param header: 是否包含表头
        :param feature_select_mode: 'regression'/'classification'
        :param k_features: 选择top k特征（None时禁用）
        """
        super().__init__()
        self.data_path=data_path
        self.label_col = label_col
        self.has_header = header
        self.mode = feature_select_mode
        self.k = k_features
        self._feat_names = None
        self._label_name = None
        self._selector = None          # 存储训练好的特征选择器
        self._selected_feat_indices = None  # 存储选中的特征索引

    def _read_raw_data(self) -> pd.DataFrame:
        """读取CSV文件并保留列名"""
        df = pd.read_csv(self.data_path, header=0 if self.has_header else None)
        
        if self.has_header:
            self._feat_names = [col for col in df.columns if col != self.label_col]
            self._label_name = self.label_col
        else:
            self._feat_names = [f"feature_{i}" for i in range(df.shape[1]-1)] if self.label_col else None
            
        return df

    def _select_features(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """执行特征选择"""
        # 分离特征和标签
        X = self._raw_data.drop(self.label_col, axis=1).values if self.label_col else self._raw_data.values
        y = self._raw_data[self.label_col].values if self.label_col else None

        # 特征选择逻辑
        if self.k is not None and self.k < X.shape[1]:
            score_func = f_regression if self.mode == 'regression' else f_classif
            self._selector = SelectKBest(score_func=score_func, k=self.k)
            self._selector.fit(X, y)
            self._selected_feat_indices = self._selector.get_support(indices=True)
            X = self._selector.transform(X)
            
        return X, y

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """默认标准化预处理"""
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

    @property
    def feature_names(self) -> Optional[List[str]]:
        """获取选中的特征名称列表"""
        if self._feat_names is None or self._selected_feat_indices is None:
            return None
        return [self._feat_names[i] for i in self._selected_feat_indices]

    @property
    def label_name(self) -> Optional[str]:
        """获取标签列名称"""
        return self._label_name
    
    def save_selected_features(self, save_path: Union[str, Path]) -> None:
        """
        保存特征选择结果到文件
        Args:
            save_path: 保存路径
        Raises:
            ValueError: 如果尚未执行特征选择
        """
        if self._selected_feat_indices is None:
            raise ValueError("尚未执行特征选择，无法保存")
            
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'indices': self._selected_feat_indices.tolist(),
            'feature_names': self._feat_names,
            'label_name': self._label_name,
            'metadata': {
                'k_features': self.k,
                'mode': self.mode,
                'saved_at': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_selected_features(
        self, 
        load_path: Union[str, Path], 
        test_data_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        从文件加载特征选择结果并返回测试数据
        
        Args:
            load_path: 特征选择结果文件路径
            test_data_path: 测试数据文件路径（None时使用当前数据）
            
        Returns:
            Tuple: (测试特征矩阵, 测试标签) 标签可能为None
            
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式无效或数据不匹配
        """
        # 加载特征选择结果
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"特征文件不存在: {load_path}")
            
        try:
            with open(load_path) as f:
                data = json.load(f)
                
            self._selected_feat_indices = np.array(data['indices'])
            self._feat_names = data['feature_names']
            self._label_name = data.get('label_name')
            
            # 验证元数据一致性
            if 'metadata' in data:
                if data['metadata']['k_features'] != self.k:
                    print(f"警告: 加载的特征数({data['metadata']['k_features']})与当前设置({self.k})不一致")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"无效的特征文件格式: {str(e)}")

        # 加载测试数据
        test_data = self._read_raw_data() if test_data_path is None else pd.read_csv(test_data_path)
        
        # 应用特征选择
        X_test = test_data.drop(self.label_col, axis=1).values if self.label_col else test_data.values
        y_test = test_data[self.label_col].values if self.label_col else None
        
        if self._selected_feat_indices is not None:
            X_test = X_test[:, self._selected_feat_indices]
        
        return X_test, y_test

    def _validate_feature_selection(self) -> None:
        """验证特征选择结果的有效性"""
        if self._selected_feat_indices is None:
            raise RuntimeError("尚未执行特征选择")
        if len(self._selected_feat_indices) == 0:
            raise ValueError("特征选择结果为空")
        if self._feat_names and any(i >= len(self._feat_names) for i in self._selected_feat_indices):
            raise ValueError("特征索引超出范围")
        
class NPYDataReader(BaseDataReader):
    """ 分类器 """
    def __init__(self,
        data_path: str,
        label_path: str=None,
        feature_select_mode: str = 'classify'):
        self.data_path = Path(data_path)
        if label_path is not None:
            self.label_path=Path(label_path)
        else:
            self.label_path=None
        super().__init__()

    def _read_raw_data(self):
        data = np.load(self.data_path)
        if self.label_path is not None:
            label = np.load(self.label_path)
            return data,label
        else:
            return data

    def _select_features(self)-> Tuple[np.ndarray, Optional[np.ndarray]]:
        """子类必须实现的具体特征选择逻辑"""
        return self._raw_data
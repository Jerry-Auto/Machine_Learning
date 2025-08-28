from torchvision import datasets, transforms
from base import BaseDataLoader
from .data_reader import CSVDataReader
from .data_sets import CovidDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CSVDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self,data_dir, batch_size,label,k_feats,feats_path,mode="train",shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        if (mode=="train"):
            self.data_reader=CSVDataReader(data_dir,label_col=label,k_features=k_feats)
            x, y =self.data_reader.load_data()
            self.data_reader.save_selected_features(feats_path)
            self.dataset = CovidDataset(x,y)
            super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        elif(mode=="test"):
            self.test_data_reader=CSVDataReader(data_dir)
            x,y=self.test_data_reader.load_selected_features('features/train_features.json')
            self.dataset = CovidDataset(x)
            super().__init__(self.dataset, batch_size,False,0.0, num_workers)
    

    

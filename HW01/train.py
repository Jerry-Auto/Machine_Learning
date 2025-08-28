# %% [markdown]
# # Import packages

# %%
# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Matplotlib
import matplotlib.pyplot as plt

# Optuna
import optuna

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# %% [markdown]
# # Some Utility Functions
# 
# You do not need to modify this part.

# %%
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

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

# %% [markdown]
# # Dataset

# %%
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

# %% [markdown]
# # Neural Network Model
# 
# Try out different model architectures by modifying the class below. (You could tune config['layer'] to try)

# %%
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, config['layer'][0]),
            nn.ReLU(),
            nn.Linear(config['layer'][0], config['layer'][1]),
            nn.ReLU(),
            nn.Linear(config['layer'][1], 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

# %% [markdown]
# # Feature Selection
# 
# Choose features you deem useful by modifying the function below.

# %%
from sklearn.feature_selection import SelectKBest, f_regression

def select_feat(train_data, valid_data, test_data, no_select_all=True):
    '''Selects useful features to perform regression'''
    global config
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]#训练集与测试集的标签
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data #训练集与测试集的特征

    if not no_select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # Feature selection
        k = config['k']
        selector = SelectKBest(score_func=f_regression, k=k)
        result = selector.fit(train_data[:, :-1], train_data[:,-1])
        idx = np.argsort(result.scores_)[::-1]
        feat_idx = list(np.sort(idx[:k]))

    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid

# %% [markdown]
# # Training Loop

# %%
def trainer(train_loader, valid_loader, model, config, device):
    
    # Define your loss function, do not modify this.
    criterion = nn.MSELoss(reduction='mean') 
    
    # Define your optimization algorithm.
    if config['optim'] == 'SGD':
        if config['no_momentum']:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])     
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])     
    elif config['optim'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
    # Writer of tensoboard.
    writer = SummaryWriter() 

        
    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []
        
        # 如果你在kaggle上运行，可以注释掉大部分的打印函数，并将train_pbar注释掉，令 x,y in train_loader，因为kaggle上打印太多可能会报错。
        # tqdm is a package to visualize your training progress.
        #train_pbar = tqdm(train_loader, position=0, leave=True)
        #for x, y in train_pbar:
        for x, y in train_loader:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            #train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            #train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader: 
            x, y = x.to(device), y.to(device) 
            with torch.no_grad(): 
                pred = model(x)
                loss = criterion(pred, y) 

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)        
        
        #if epoch % 100 == 0:
        #    print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if not config['no_tensorboard']:
            writer.add_scalar('Loss/train', mean_train_loss, step)
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            
            # 一轮epcho中保存 K 折交叉验证中单折表现最好的模型
            if len(valid_scores):
                if best_loss < min(valid_scores):
                    torch.save(model.state_dict(), config['save_path']) # Save your best model
                    #print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
                    print('Saving model with loss {:.3f}...'.format(best_loss))
            else:
                torch.save(model.state_dict(), config['save_path']) # Save your best model
                #print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
                print('Saving model with loss {:.3f}...'.format(best_loss))
                
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('Best loss {:.3f}...'.format(best_loss))
            print('\n在'+early_stop_count+'次参数更新后，模型损失未下降，该fold训练结束！')
            break
    return best_loss

# %% [markdown]
# # Save predictions

# %%
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# %% [markdown]
# # Start training!
# 
# config contains hyper-parameters for training and the path to save your model.
# 
# `objective()` is used for automatic parameter tuning, but you could set `AUTO_TUNE_PARAM` `False` to avoid it.

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'k': 20,              # Select k features
    'layer': [20, 20],
    'optim': 'Adam',
    'momentum': 0.7,
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 1000,    # Number of epochs.
    'batch_size': 300,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'early_stop': 500,        # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt',  # Your model will be saved here.
    'no_select_all': True,    # Whether to use all features.
    'no_momentum': True,      # Whether to use momentum
    'no_normal': True,        # Whether to normalize data
    'no_k_cross': True,      # Whether to use K-fold cross validation，数据平均分成K份，按顺序拿出一份来做测试集，其余的做训练集进行K次训练，总共参数更新次数： K*epchos*batch_sizes
    'no_save': False,         # Whether to save model parameters
    'no_tensorboard': False,  # Whether to write tensorboard
} 

# 设置 k-fold 中的 k，这里是根据 valid_ratio 设定的
k = int(1 / config['valid_ratio'])

 # Set seed for reproducibility
same_seed(config['seed'])

training_data, test_data = pd.read_csv('./covid_train.csv').values, pd.read_csv('./covid_test.csv').values
    
num_valid_samples = len(training_data) // k
np.random.shuffle(training_data)
valid_scores = []  # 记录 valid_loss

def objective(trial):
    if trial != None:
        print('\nNew trial here')
        # 定义需要调优的超参数空间
        config['learning_rate'] = trial.suggest_float('lr', 1e-6, 1e-3)
        config['batch_size'] = trial.suggest_categorical('batch_size', [128])
        config['k'] = trial.suggest_int('k_feats', 16, 32)
        config['layer'][0] = config['k']
    
    # 打印所需的超参数
    print(f'''hyper-parameter: 
        epchos: {config['n_epochs']},
        optimizer: {config['optim']},
        lr: {config['learning_rate']}, 
        batch_size: {config['batch_size']}, 
        k: {config['k']}, 
        layer: {config['layer']}''')
    
    global valid_scores
    # 每次 trial 初始化 valid_scores，可以不初始化，通过 trial * k + fold 来访问当前 trial 的 valid_score，
    # 这样可以让 trainer() 保存 trials 中最好的模型参数，但这并不意味着该参数对应的 k-fold validation loss 最低。
    valid_scores = []

    for fold in range(k):
        # Data split
        valid_data = training_data[num_valid_samples * fold:
                                num_valid_samples * (fold + 1)]
        train_data = np.concatenate((
            training_data[:num_valid_samples * fold],
            training_data[num_valid_samples * (fold + 1):]))

        # Normalization
        if not config['no_normal']:
            train_mean = np.mean(train_data[:, 35:-1], axis=0)  # 前 35 列为 one-hot vector，我并没有对他们做 normalization，可以自行设置
            train_std = np.std(train_data[:, 35:-1], axis=0)
            train_data[:, 35:-1] -= train_mean
            train_data[:, 35:-1] /= train_std
            valid_data[:, 35:-1] -= train_mean
            valid_data[:, 35:-1] /= train_std
            test_data[:, 35:] -= train_mean
            test_data[:, 35:] /= train_std

        x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['no_select_all'])
        
        train_dataset, valid_dataset, test_dataset = CovidDataset(x_train, y_train), \
                                                CovidDataset(x_valid, y_valid), \
                                                CovidDataset(x_test)

        # Pytorch data loader loads pytorch dataset into batches.
        p_m=False;
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=p_m)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=p_m)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=p_m)
        
        model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
        valid_score = trainer(train_loader, valid_loader, model, config, device)
        valid_scores.append(valid_score)
        
        if not config['no_k_cross']:
            break
            
        if valid_score > 2:
            print(f'在第{fold+1}折上欠拟合') # 提前终止，减少计算资源
            break       
        
    print(f'valid_scores: {valid_scores}')
    
    if trial != None:
        return np.average(valid_scores)
    else:
        return x_test, test_loader




AUTO_TUNE_PARAM = False  # Whether to tune parameters automatically

if AUTO_TUNE_PARAM:
    # 使用Optuna库进行超参数搜索
    n_trials = 10  # 设置试验数量
    print(f'AUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}\nn_trials: {n_trials}')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # 输出最优的超参数组合和性能指标
    print('Best hyperparameters: {}'.format(study.best_params))
    print('Best performance: {:.4f}'.format(study.best_value))
else:
    # 注意，只有非自动调参时才进行了predict，节省一下计算资源
    print(f'You could set AUTO_TUNE_PARAM True to tune parameters automatically.\nAUTO_TUNE_PARAM: {AUTO_TUNE_PARAM}')
    x_test, test_loader = objective(None)
    model = My_Model(input_dim=x_test.shape[1]).to(device)
    model.load_state_dict(torch.load(config['save_path']))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'submission.csv')


# %% [markdown]
# # Plot learning curves with `tensorboard` (optional)
# 
# `tensorboard` is a tool that allows you to visualize your training progress.
# 
# If this block does not display your learning curve, please wait for few minutes, and re-run this block. It might take some time to load your logging information. 

# %%
# %reload_ext tensorboard
# %tensorboard --logdir=./runs/

# %%




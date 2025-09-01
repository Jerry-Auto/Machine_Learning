import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    len_epoch代表一个epoch有多少个batch_size,此时dataloader无限循环，长度为无穷大
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        epoch_length=self.data_loader.n_samples/self.data_loader.batch_size
        self.log_step = int(np.sqrt(epoch_length))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()#切换到训练模式
        #每个epoch的评价指标都要清零
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            #tensorboard记录当前训练速度，步长为一个batch_size，即一次参数更新
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            #记录当前batch的评价指标值
            self.train_metrics.update('loss', loss.item())
            #记录除了loss评价指标之外的其他评价指标
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                #输出示意：Train Epoch: 5 [128/1000 (13%)] Loss: 0.324512
                # self.logger.debug('Train Epoch: {:^3} {} Loss: {:.6f}'.format(
                #     epoch,
                #     self._progress(batch_idx),
                #     loss.item()))
                self.writer.add_scalar('Loss/train', loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        #返回指标的均值
        log = self.train_metrics.result()


        #如果有验证集
        if self.do_validation:
            #返回所有指标的均值
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        
        #如果有学习率自适应调节器
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log
        # 返回值示意  
        #{
        #     # 训练指标（来自train_metrics）
        #     'loss': float,               # 训练集平均损失
        #     'accuracy': float,          # 训练集准确率 
        #     'f1_score': float,          # 其他自定义指标...
            
        #     # 验证指标（来自_valid_epoch，前缀为val_）
        #     'val_loss': float,          # 验证集平均损失
        #     'val_accuracy': float,      # 验证集准确率
            
        #     # 系统信息（可选）
        #     'epoch': int,               # 当前epoch数
        #     'lr': float                 # 当前学习率（如果记录）
        # }

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        #切换为验证模式
        self.model.eval()
        #每个epoch的评价指标都要清零
        self.valid_metrics.reset()
        #关闭梯度
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                #tensorboard记录当前训练速度，步长为一个batch_size，即一次参数更新
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                
                #不输出日志，只保存tensorboard数据
                self.writer.add_scalar('Loss/eval', loss.item(), (epoch - 1) * self.len_epoch + batch_idx)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p.detach().cpu().float(), bins='auto')

            #返回所有指标的均值
        return self.valid_metrics.result()
    
    #返回三个数值[96/1000 (10%)]，最后一位表示当前batch在当前epoch的进度
    def _progress(self, batch_idx):
        """返回居中格式的进度信息
        Returns:
            str: 格式如 [  900  /  2709  ( 33% )]
        """
        # 计算当前进度和总量
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx + 1  # 从1开始计数
            total = self.len_epoch
        # 计算百分比
        percent = int(100.0 * current / total)
        # 居中格式输出
        return "[ {:^5} / {:^5} ( {:^3}% ) ]".format(
            current,
            total,
            percent)

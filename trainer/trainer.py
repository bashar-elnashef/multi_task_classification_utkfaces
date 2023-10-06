import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.tasks = self.config ['loss']['tasks']
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[f'loss_{tsk}' for tsk in self.tasks], 
                *[f'{m.__name__}_{tsk}' for tsk in self.tasks for m in self.metric_ftns], writer=self.writer)

        self.valid_metrics = MetricTracker('loss', *[f'loss_{tsk}' for tsk in self.tasks],  
                *[f'{m.__name__}_{tsk}' for tsk in self.tasks for m in self.metric_ftns], writer=self.writer)

        self.sig = nn.Sigmoid()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)
            age_tar = target[self.tasks[0]].to(self.device)
            gen_tar = target[self.tasks[1]].to(self.device)
            rac_tar = target[self.tasks[2]].to(self.device)
            true = [age_tar, gen_tar.unsqueeze(1).float(), rac_tar]

            self.optimizer.zero_grad()
            age_pred, gen_pred, rac_pred = self.model(data)
            pred = [age_pred, self.sig(gen_pred), rac_pred]

            loss_age, loss_gender, loss_race = self.criterion(pred, true)
            loss = loss_age + loss_gender + loss_race
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for idx, tsk in enumerate(self.tasks):
                self.train_metrics.update(f'loss_{tsk}', locals()[f'loss_{tsk}'].item())
                for met in self.metric_ftns:
                    self.train_metrics.update(f'{met.__name__}_{tsk}', met(pred[idx], true[idx]))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{f'val_{k}': v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                age_tar = target[self.tasks[0]].to(self.device)
                gen_tar = target[self.tasks[1]].to(self.device)
                rac_tar = target[self.tasks[2]].to(self.device)
                true = [age_tar, gen_tar.unsqueeze(1).float(), rac_tar]

                age_pred, gen_pred, rac_pred = self.model(data)

                pred = [age_pred, self.sig(gen_pred), rac_pred]
                loss_age, loss_gender, loss_race = self.criterion(pred, true)
                loss = loss_age + loss_gender + loss_race

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for idx, tsk in enumerate(self.tasks):
                    self.valid_metrics.update(f'loss_{tsk}', locals()[f'loss_{tsk}'].item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(f'{met.__name__}_{tsk}', met(pred[idx], true[idx]))

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

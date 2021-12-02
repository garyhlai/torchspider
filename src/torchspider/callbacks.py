import sys
import numpy as np
import torch
from fastai.basics import GetAttr, store_attr
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb


class Callback(GetAttr):
    _default = 'learner'


class TestFreezing(Callback):
    def before_train_epoch(self):
        self.begin_params = []
        check_first_n = 50
        current_n = 0
        for param in self.learner.model.parameters():
            self.begin_params.append(param)
            current_n += 1
            if current_n >= check_first_n:
                break

    def after_epoch(self):
        self.after_params = []
        check_first_n = 50
        current_n = 0
        for param in self.learner.model.parameters():
            self.after_params.append(param)
            current_n += 1
            if current_n >= check_first_n:
                break


class TrackLoss(Callback):
    '''
    We always track epoch train loss, but we validate periodically during the epoch, so the 
    valid loss is named current_valid_loss. 
    '''

    def __init__(self, save_model=False, path=None, model_name=None):
        if save_model:
            if not path or not model_name:
                raise ValueError(
                    "you want to save but path or model name is None")
        store_attr('save_model, path, model_name', self)
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float("inf")

    def before_epoch(self):
        self.epoch_train_loss = []

    def after_train_loss(self):
        self.epoch_train_loss.append(float(self.loss))

    def before_validate(self):
        self.current_valid_loss = []

    def after_valid_loss(self):
        self.current_valid_loss.append(float(self.loss))

    def after_validate(self):
        # valid_losses store the loss for every step / batch
        self.valid_losses += self.current_valid_loss
        # to evaluate the model, we average all the batches in valid_ds
        self.cur_model_val_loss = np.mean(self.current_valid_loss)
        if self.cur_model_val_loss < self.best_valid_loss:
            self.best_valid_loss = self.cur_model_val_loss
            tqdm.write(
                f">>>>> best valid loss updated: {self.best_valid_loss}\r")
            if self.save_model:
                torch.save(self.model.state_dict(),
                           f"{self.path}/{self.model_name}_best_valid.pth")
            else:
                tqdm.write("!!! Warning: model is not saving \r")

    def after_epoch(self):
        self.train_losses += self.epoch_train_loss

        if self.train_losses and self.valid_losses:
            print("*" * 90)
            print(f"epoch {self.epoch} done | epoch train loss: {np.mean(self.epoch_train_loss)} | current valid loss: {np.mean(self.current_valid_loss)}")
            print("*" * 90)
            print()
        else:
            tqdm.write("Warning: one of the losses is None")
        if self.save_model:
            if self.epoch > 1:  # don't save the first epoch to save memory
                torch.save(self.model.state_dict(),
                           f"{self.path}/{self.model_name}_epoch{self.epoch}.pth")


class CudaCallback(Callback):
    def __init__(self, device="cuda"):
        self.device = device

    def before_fit(self):
        self.model.to(self.device)
        self.get_pred.to(self.device)

    def before_batch(self):
        self.learner.batch_x = ({k: v.to(self.device) for k, v in self.batch_x[0].items()},
                                {k: v.to(self.device) for k, v in self.batch_x[1].items()})

        self.learner.batch_y = self.batch_y.to(self.device)

    def before_train_batch(self):
        self.before_batch()

    def before_valid_batch(self):
        self.before_batch()


class LrRecorder(Callback):
    def __init__(self, param_group_length):
        self.lrs = [[] for _ in range(param_group_length)]

    def before_train_batch(self):
        for i, param_group in enumerate(self.learner.optimizer.param_groups):
            self.lrs[i].append(param_group['lr'])

    def plot_lrs(self):
        for i in range(len(self.lrs)):
            if i == 0:
                label = 'tail group'
            elif i == len(self.lrs)-1:
                label = 'head group'
            else:
                label = f"group {i-1}"
            plt.plot(self.lrs[i], label=label)
        plt.legend(loc="upper right")


class Scheduler(Callback):
    def after_train_batch(self):
        self.scheduler.step()


class ProgressBar(Callback):
    def before_epoch(self):
        self.learner.pbar = tqdm(total=len(
            self.dls.train_dl), position=0, desc=f"epoch {self.epoch}", leave=True, file=sys.stdout)

    def after_train_batch(self):
        self.learner.pbar.update(1)

    def after_epoch(self):
        self.learner.pbar.close()

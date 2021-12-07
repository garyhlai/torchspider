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


class SaveModel(Callback):
    '''
    Save model (1) after each epoch, (2) after the best validation loss
    Make sure this callback is the last one in the list of callbacks
    '''

    def __init__(self, path, model_name):
        store_attr('path, model_name', self)

    def after_validate(self):
        if self.learner.is_updated_best_valid_loss:
            torch.save(self.model.state_dict(),
                       f"{self.path}/{self.model_name}_best_valid.pth")

    def after_epoch(self):
        if self.epoch > 1:  # don't save the first epoch to save memory
            torch.save(self.model.state_dict(),
                       f"{self.path}/{self.model_name}_epoch{self.epoch}.pth")


class TrackLoss(Callback):
    '''
    We always track epoch train loss, but we validate periodically during the epoch, so the
    valid loss is named interval_valid_loss as opposed to epoch_valid_loss.
    '''

    def __init__(self):
        # store the loss for every step / batch
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float("inf")

    def before_epoch(self):
        self.epoch_train_loss = []

    def after_train_loss(self):
        self.epoch_train_loss.append(float(self.loss))

    def before_validate(self):
        self.interval_valid_loss = []
        self.correct_count = 0

    def after_valid_loss(self):
        self.interval_valid_loss.append(float(self.loss))
        # get batch correct count
        pred = torch.argmax(self.learner.pred, 1)
        self.correct_count += pred.eq(self.batch_y).sum(
        ).item()

    def after_validate(self):
        # update valid loss
        self.valid_losses += self.interval_valid_loss
        self.learner.is_updated_best_valid_loss = self.update_best_valid_loss_maybe(
            np.mean(self.interval_valid_loss))
        # update accuracy
        self.acc = self.correct_count / len(self.dls.valid_dl.dataset)

    def after_epoch(self):
        self.train_losses += self.epoch_train_loss
        self.log_current_loss()

    def log_current_loss(self):
        if self.train_losses and self.valid_losses:
            print("*" * 90)
            print(f"epoch {self.epoch} done | avg epoch train loss: {np.mean(self.epoch_train_loss)} | avg current valid loss: {np.mean(self.interval_valid_loss)} | acc: {self.acc}")
            print("*" * 90)
            print()
        else:
            tqdm.write("Warning: one of the losses is None")

    def update_best_valid_loss_maybe(self, cur_model_val_loss):
        # update best valid loss if applicable
        if cur_model_val_loss < self.best_valid_loss:
            self.best_valid_loss = cur_model_val_loss
            tqdm.write(
                f">>>>> best valid loss updated: {self.best_valid_loss}\r")
            return True
        return False


class WandbTrackAndSave(TrackLoss):
    """
    Example: 
        ```
        WandbTrackAndSave("hello", {"learning_rate": self.lr})
        ```
    """

    def __init__(self, project, config, path, model_name):
        super().__init__()
        store_attr('project, config, path, model_name', self)
        self.best_valid_path = f"{self.path}/{self.model_name}_best_valid.pth"

    def before_fit(self):
        self.wandb_run = wandb.init(project=self.project,
                                    config=self.config)
        wandb.watch(self.model)  # track gradients

    def after_train_loss(self):
        step_train_loss = float(self.loss)
        self.epoch_train_loss.append(step_train_loss)
        wandb.log({"train_loss": step_train_loss})

    def after_valid_loss(self):
        step_valid_loss = float(self.loss)
        self.interval_valid_loss.append(step_valid_loss)
        wandb.log({"valid_loss": step_valid_loss})

    def after_validate(self):
        super().after_validate()
        if self.learner.is_updated_best_valid_loss:
            torch.save(self.model.state_dict(),
                       self.best_valid_path)

    def after_fit(self):
        save_learner_path = "."
        if self.save_learner:
            self.save(save_learner_path)
            wandb.save(f"{save_learner_path}/learner.pkl")
        self.wandb_run.finish()


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

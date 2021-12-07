from typing import Set, List, Dict
import torch
from torch.cuda import amp
from fastai.basics import store_attr, noop
import dill
from .utils import find_incremental_filename
from transformers import AdamW
from .callbacks import ProgressBar


"""
Order of lifecycle:

- before_fit
    - before_epoch
            - before_train_batch
                - before_train_loss
                - after_train_loss
            - after_train_batch
        - before_validate (interval)
            - before_valid_batch
                - before_valid_loss
                - after_valid_loss
            - after_valid_batch
        - after_validate (interval)
    - after_epoch
- after_fit
"""


class Learner:
    def __init__(self, model, dls, config, valid_interval=None, cbs=[], get_pred=None, save_learner=True):
        store_attr(
            'model, dls, config, valid_interval, cbs, get_pred, save_learner', self)
        self.lr = config.lr
        self.optimizer = self.get_optimizer(config.optimizer)
        self.loss_func = self.get_loss_func(config.loss_func)

        if valid_interval is None:
            self.valid_interval = len(dls.train_dl)

        self.cbs = [ProgressBar()] + self.cbs
        for cb in self.cbs:
            cb.learner = self
        self.cb_dict = {type(cb).__name__: cb for cb in self.cbs}

    def get_loss_func(self, loss_func):
        if loss_func == 'cross_entropy_loss':
            return torch.nn.functional.cross_entropy
        elif loss_func == 'binary_cross_entropy_with_logits':
            return torch.nn.functional.binary_cross_entropy_with_logits
        elif loss_func == 'mse':
            return torch.nn.MSELoss()
        elif loss_func == 'bce':
            return torch.nn.BCELoss()
        else:
            raise ValueError(f"loss_func {loss_func} not supported")

    def get_optimizer(self, optimizer: str):
        if optimizer == 'adamw':
            return AdamW(self.model.parameters(), lr=self.lr)
        elif optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"optimizer {optimizer} not supported")

    def get_batch_x_y(self, batch):
        # support huggingface dict
        if isinstance(batch, dict):
            batch_y = batch['label']
            del batch['label']
            batch_x = batch
        else:
            batch_x, batch_y = batch
        return batch_x, batch_y

    def fit(self, epochs):
        self.epochs = epochs
        self('before_fit')
        for epoch in range(1, epochs+1):

            self.epoch = epoch
            self('before_epoch')
            self.model.train()
            for i, batch in enumerate(self.dls.train_dl):
                self.batch_x, self.batch_y = self.get_batch_x_y(batch)
                self.train_batch()
                if i % self.valid_interval == 0:
                    self.validate_interval()

            self('after_epoch')
        self('after_fit')

    def validate_interval(self):
        self('before_validate')
        self.model.eval()

        for batch in self.dls.valid_dl:
            self.batch_x, self.batch_y = self.get_batch_x_y(batch)
            self.validate_batch()

        self('after_validate')
        self.model.train()

    def train_batch(self):
        self('before_train_batch')
        # forward
        if self.get_pred == None:
            self.pred = self.model(self.batch_x)
        else:
            self.out = self.model(self.batch_x)
            self.pred = self.get_pred(self.out)
        self('before_train_loss')
        self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_train_loss')
        # backward
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self('after_train_batch')

    def validate_batch(self):
        self('before_valid_batch')
        # forward
        with torch.no_grad():
            if self.get_pred == None:
                self.pred = self.model(self.batch_x)
            else:
                self.out = self.model(self.batch_x)
                self.pred = self.get_pred(self.out)
            self('before_valid_loss')
            self.loss = self.loss_func(self.pred, self.batch_y)
            self('after_valid_loss')
        self('after_valid_batch')

    def __call__(self, name):
        for cb in self.cbs:
            getattr(cb, name, noop)()

    def save(self, path):
        """
        Save callback dict to `path`
        """
        export = {}
        for cb_name, cb in self.cb_dict.items():
            # export everything except for reference to learner
            export[cb_name] = {key: value for key,
                               value in cb.__dict__.items() if key != 'learner'}
        # save
        with open("learner.pkl", 'wb') as learner_file:
            dill.dump(export, learner_file)
            print("save successful!")


# TODO: update MixedPrecisionCallback to match the updated Learner above
class MixedPrecisionLearner(Learner):
    def fit(self, epochs):
        self.epochs = epochs
        self.scaler = amp.GradScaler()

        self('before_fit')
        for epoch in range(1, epochs+1):
            self.epoch = epoch
            self('before_epoch')
            self.model.train()
            for i, batch in enumerate(self.dls.train_dl):
                self.batch_x, self.batch_y = batch
                self.train_batch()
                if i % self.valid_interval == 0:
                    self.validate_interval()
            self('after_epoch')
        self('after_fit')

    def train_batch(self):
        self('before_train_batch')
        # forward
        with amp.autocast():
            self.pred = self.get_pred(self.batch_x, self.model)
            self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_train_loss')
        # backward
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self('after_train_batch')

    def validate_batch(self):
        self('before_valid_batch')
        # forward
        with torch.no_grad():
            with amp.autocast():
                self.pred = self.get_pred(self.batch_x, self.model)
                self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_valid_loss')

class DataLoaderGroup():
    def __init__(self, train_dl, valid_dl):
        self.train_dl, self.valid_dl = train_dl, valid_dl

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset
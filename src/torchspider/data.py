class DataLoaderGroup():
    def __init__(self, train_dl, valid_dl, test_dl=None):
        self.train_dl, self.valid_dl, self.test_dl = train_dl, valid_dl, test_dl

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

    @property
    def test_ds(self):
        return self.test_dl.dataset

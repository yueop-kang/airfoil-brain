import numpy as np

class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_val_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, loss_val):

        score = -loss_val

        if self.best_score is None:
            self.best_score = score
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
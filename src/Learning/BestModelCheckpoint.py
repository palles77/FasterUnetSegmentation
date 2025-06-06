import torch

class BestModelCheckpoint:
    def __init__(self, model, filepath, monitor='val_loss', mode='min'):
        self.model = model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if self.mode == 'min':
            if current_value < self.best:
                self.best = current_value
                torch.save(self.model.state_dict(), self.filepath)
        else:
            if current_value > self.best:
                self.best = current_value
                torch.save(self.model.state_dict(), self.filepath)
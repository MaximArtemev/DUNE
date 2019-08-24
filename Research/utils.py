import torch




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, ema_decay=0.99, name=None):
        self.name = name
        self.reset()
        self.ema_decay = ema_decay

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.ema = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.ema == 0:
            self.ema = val
        else:
            self.ema = self.ema_decay * self.ema + (1-self.ema_decay) * val

        return self

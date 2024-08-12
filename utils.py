
from collections import OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LogCollector(object):


    def __init__(self):

        self.meters = OrderedDict()

    def update(self, k, v, n=0):

        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):

        s = ''
        for i, (k, v) in enumerate(self.meters.items()): 
            if i > 0:
                s += '  '
            if(k == 'lr'):
                v = '{:.3e}'.format(v.val)
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):

        for k, v in self.meters.items(): 
            tb_logger.log_value(prefix + k, v.val, step=step)

import random
import time
import datetime
import sys
import json
import torch
import numpy as np

class Logger():
    """训练过程中打印并保存 metric
    """
    def __init__(self, n_epochs, batches_epoch):
        
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.metrics = {}
        self.metric_list = {}


    def log(self, metrics=None):
        """metrics: {"name1": value1, "name2": value2}
        """
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, metric_name in enumerate(metrics.keys()):
            if metric_name not in self.metrics:
                self.metrics[metric_name] = metrics[metric_name].item()
            else:
                self.metrics[metric_name] += metrics[metric_name].item()

            if (i+1) == len(metrics.keys()):
                sys.stdout.write('%s: %.4f -- ' % (metric_name, self.metrics[metric_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (metric_name, self.metrics[metric_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # End of epoch, save metrics for plot
        if (self.batch % self.batches_epoch) == 0:
            for metric_name, value in self.metrics.items():
                if metric_name not in self.metric_list:
                    self.metric_list[metric_name] = [value / self.batch]
                else:
                    self.metric_list[metric_name].append(value / self.batch)

                # Reset metrics for next epoch
                self.metrics[metric_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

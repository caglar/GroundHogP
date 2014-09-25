import numpy, time
import itertools
import sys, os
import pickle
import cPickle as pkl
import gzip

import theano


class MnistIterator(object):
    def __init__(self,
                path=None,
                use_infinite_loop=True,
                output_format = None,
                batch_size = 1,
                mode='train'):
        self.path = path
        self.use_infinite_loop = use_infinite_loop
        self.output_format = output_format
        self.mode = mode
        self.bs = batch_size

        self.next_offset = -1

        if path == None:
            path = '/data/lisa/data/mnist/mnist.pkl.gz'
        dfile = gzip.open(path)
        mnist = pkl.load(dfile)
        dfile.close()
        if mode == 'train':
            self.data_x = mnist[0][0]
            self.data_y = mnist[0][1]
        elif mode == 'valid':
            self.data_x = mnist[1][0]
            self.data_y = mnist[1][1]
        else:
            self.data_x = mnist[2][0]
            self.data_y = mnist[2][1]

        self.pos = 0

    def __iter__(self):
        return self

    def get_length(self):
        return len(self.data_x)

    def start(self, start_offset):
        logger.debug("Not supported")
        self.next_offset = -1

    def next(self):
        if self.pos >= len(self.data_x) and self.use_infinite_loop:
            #print 'Restarting iterator'
            self.pos = 0
        elif self.pos >= len(self.data_x):
            self.pos = 0
            raise StopIteration

        if self.bs != "full":
            self.pos += self.bs
            if not self.output_format:
                return self.data_x[self.pos-self.bs:self.pos],\
                        self.data_y[self.pos-self.bs:self.pos]
            else:
                return self.output_format(self.data_x[self.pos-self.bs:self.pos],
                                          self.data_y[self.pos-self.bs:self.pos])
        else:
            self.pos = len(self.data_x)
            if not self.output_format:
                return self.data_x, self.data_y
            else:
                return self.output_format(self.data_x,
                                          self.data_y)



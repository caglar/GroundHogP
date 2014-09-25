"""
Data iterator for text datasets that are used for language modeling.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy as np
import logging

logger = logging.getLogger(__name__)

class NParityIterator(object):
    def __init__(self,
                 batch_size,
                 stop,
                 start=0,
                 max_iters=0,
                 use_hints=False,
                 path=None):

        assert type(path) == str, "Target language file path should be a string."
        self.path = path
        self.batch_size = batch_size
        self.data = self.load_file()

        self.X = np.asarray(self.data[0].tolist(), dtype=np.int8)
        self.y = np.asarray(self.data[1].tolist(), dtype=np.int8)

        self.data_len = self.X.shape[0]

        if stop is None:
            stop = self.data_len

        self.batch_size = batch_size
        self.start = start
        self.stop = stop
        self.use_hints = use_hints
        self.next_offset = -1
        self.niters = 0
        self.max_iters = max_iters

        self.cnt = 0
        self.offset = 0

        if self.use_hints:
            self.hints = self.X.sum(1)

    def load_file(self):
        mmap_mode = None
        data = np.load(self.path)
        return data

    def start(self, start_offset):
        logger.debug("Not supported")
        self.next_offset = -1

    def __iter__(self):
        return self

    def next(self):
        inc_offset = self.offset + self.batch_size
        hints = None
        if inc_offset > self.stop:
            len_diff = inc_offset - self.stop
            data_part1 = self.X[self.offset:self.stop]
            y_part1 = self.y[self.offset:self.stop]
            data_part2 = self.X[self.start:len_diff]
            y_part2 = self.y[self.start:len_diff]

            if self.use_hints:
                hints_part1 = self.hints[self.offset:self.stop]
                hints_part2 = self.hints[self.start:len_diff]
                hints = np.concatenate((hints_part1, hints_part2), axis=0)

            data = np.concatenate((data_part1, data_part2), axis=0)
            y = np.concatenate((y_part1, y_part2), axis=0)
            self.offset = self.start + len_diff
            self.niters += 1
        else:
            data = self.X[self.offset:inc_offset]
            y = self.y[self.offset:inc_offset]
            if self.use_hints:
                hints  = self.hints[self.offset:inc_offset]

            self.offset = inc_offset

        ret_dict = {"x": data, "y": y}

        if self.use_hints:
            ret_dict["hints"] = hints

        if self.niters > self.max_iters:
            self.offset = 0
            self.niters = 0
            raise StopIteration

        return ret_dict


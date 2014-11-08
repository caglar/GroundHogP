"""
Data iterator for text datasets that are used for language modeling.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Caglar Gulcehre "
               "KyungHyun Cho ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import logging
import theano
import cPickle as pkl

np = numpy

logger = logging.getLogger(__name__)

#Get patches from a single image according to patch_size
def get_patches(img, patch_size=(8, 8)):
    if img.ndim == 1:
        dim = np.sqrt(img.shape[0])
        img = img.reshape((dim, dim))

    img_rows = img.shape[0]
    img_cols = img.shape[1]
    patches = None

    for i in xrange(img_rows / patch_size[0]):
        for j in xrange(img_cols / patch_size[1]):
            patch = img[i * patch_size[0]: (i + 1)* patch_size[0], j * patch_size[0]: (j + 1) * patch_size[1]]
            patch = patch.flatten()
            if patches is None:
                patches = patch
            else:
                if patches.ndim != patch.ndim:
                    patches = np.vstack((patches, [patch]))
                else:
                    patches = np.vstack(([patches], [patch]))
    return patches


#Get patches from the dataset for each image
def get_dataset_patches(data, patch_size=(8, 8)):
    data_patches = []
    for x in data:
        data_patches.append(get_patches(x, patch_size))
    return np.asarray(data_patches, dtype=theano.config.floatX)


class PentominoIterator(object):

    def __init__(self,
                 pdir=None,
                 use_infinite_loop=True,
                 output_format=None,
                 batch_size=1,
                 names=None,
                 start=None,
                 stop=None,
                 use_binary=True,
                 mode='train'):

        self.pdir = pdir
        self.use_infinite_loop = use_infinite_loop
        self.output_format = output_format
        self.mode = mode
        self.bs = batch_size
        self.next_offset = -1
        self.dtype = theano.config.floatX
        self.img_size = 64*64
        self.next_offset = -1
        self.pos = 0

        if pdir is None:
            self.pdir = '/data/lisa/data/pentomino/datasets/'

        if type(names) is not list:
            names = [names]

        if names is None:
            raise ValueError("Dataset names is not None.")

        X, y, hints = self._load_data(self.pdir, names)

        if start is None:
            start = 0

        if stop is None:
            stop = y.shape[0]

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.img_size
        n_multi_classes = 10
        self.n_multi_classes = n_multi_classes
        self.use_binary = use_binary
        self.X = X[start:stop]
        self.y = y[start:stop]
        self.hints = hints

        if self.use_binary:
            self.y = self._binarize_labels(self.y)

    def _binarize_labels(self, y):
        last_lbl = self.n_multi_classes
        binarized_lbls = []
        for label in y:
            if label == last_lbl:
                binarized_lbls.append(0)
            else:
                binarized_lbls.append(1)
        return np.asarray(binarized_lbls, dtype="uint8")

    def _load_data(self, pdir, names):
        X = None
        y = None
        hints = None

        for name in names:
            if name.endswith("npy"):
                data = np.load(pdir + name)
                if X is None:
                    X = data[0]
                else:
                    X = np.vstack((X, data[0]))
                if y is None:
                    y = data[1]
                else:
                    y = np.vstack((y, data[1]))

                if data.shape[0] == 3:
                    if hints is None:
                        hints = data[2]
                    else:
                        hints = np.vstack((hints, data[2]))
            else:
                data = pkl.load(open(pdir + name, "rb"))
                if X is None:
                    X = data[0]
                else:
                    X = np.vstack((X, data[0]))
                if y is None:
                    y = data[1]
                else:
                    y = np.vstack((y, data[1]))

                if data.shape[0] == 3:
                    if hints is None:
                        hints = data[2]
                    else:
                        hints = np.vstack((hints, data[2]))

        if hints is not None:
            hints = np.asarray(hints.tolist(), dtype=theano.config.floatX)

        return (np.asarray(X.tolist(), dtype=self.dtype), np.asarray(y.tolist(), dtype="uint8"), hints)

    def __iter__(self):
        return self

    def get_length(self):
        return len(self.X)

    def start(self, start_offset):
        logger.debug("Not supported")
        self.next_offset = -1

    def next(self):
        if self.pos >= len(self.X) and self.use_infinite_loop:
            # Restart the iterator
            self.pos = 0
        elif self.pos >= len(self.X):
            self.pos = 0
            raise StopIteration

        if self.bs != "full":
            self.pos += self.bs
            if not self.output_format:
                return (self.X[self.pos-self.bs: self.pos],
                        self.y[self.pos - self.bs: self.pos])
            else:
                return self.output_format(self.X[self.pos-self.bs:self.pos],
                                          self.y[self.pos-self.bs:self.pos])
        else:
            self.pos = len(self.X)
            if not self.output_format:
                return self.X, self.y
            else:
                return self.output_format(self.X,
                                          self.y)


class PentominoTensorIterator(PentominoIterator):
    def __init__(self,
                 pdir=None,
                 use_infinite_loop=True,
                 output_format=None,
                 batch_size=1,
                 kern_size=8*8,
                 names=None,
                 start=None,
                 stop=None,
                 use_binary=True,
                 mode='train'):

        super(PentominoTensorIterator, self).__init__(pdir=pdir,
                                                       use_infinite_loop=use_infinite_loop,
                                                       output_format=output_format,
                                                       batch_size=batch_size,
                                                       names=names,
                                                       start=start,
                                                       stop=stop,
                                                       use_binary=use_binary,
                                                       mode=mode)

        self.kern_size = kern_size
        self.nkerns = self.img_size / kern_size
        self.X = get_dataset_patches(self.X)


class PentominoTensorHintsIterator(PentominoTensorIterator):

    def __init__(self,
                 pdir=None,
                 use_infinite_loop=True,
                 output_format=None,
                 batch_size=1,
                 kern_size=8*8,
                 names=None,
                 start=None,
                 stop=None,
                 hints_type=1, # 1: Global, 2: Local hints
                 use_binary=True,
                 mode='train'):

        self.hints_type = hints_type
        super(PentominoTensorHintsIterator, self).__init__(pdir=pdir,
                                                     use_infinite_loop=use_infinite_loop,
                                                     output_format=output_format,
                                                     batch_size=batch_size,
                                                     names=names,
                                                     start=start,
                                                     stop=stop,
                                                     kern_size=kern_size,
                                                     use_binary=use_binary,
                                                     mode=mode)
        self.hints = self.hints[start:stop]

    def _get_binary_targets(self, targets):
        n_elems = len(targets)
        bin_hints = np.zeros((self.n_multi_classes + 1))
        i = 0
        #import ipdb; ipdb.set_trace()
        for z in targets:
            bin_hints[z] = 1
        return bin_hints

    def _grab_hints(self, hints):
        bin_hints = []
        assert hints is not None, "Hints should not be empty!"
        if self.hints_type == 1:

            for hint in hints:
                indxs = np.where(hint!=-1)[0]
                filtered_hint = np.unique(hint[indxs])
                bin_hint = self._get_binary_targets(filtered_hint)
                bin_hints.append(bin_hint.tolist())

            #bin_hints = self._get_binary_targets(bin_hints)
        else:
            for hint in hints:
                hint = hint + 1
                indxs = np.where(hint!=0)[0]
                hint[indxs] = 1
                bin_hints.append(hint.tolist())

        return np.asarray(bin_hints, dtype=theano.config.floatX)

    def next(self):
        if self.pos >= len(self.X) and self.use_infinite_loop:
            # Restart the iterator
            self.pos = 0
        elif self.pos >= len(self.X):
            self.pos = 0
            raise StopIteration

        if self.bs != "full":
            self.pos += self.bs
            hints = self._grab_hints(self.hints[self.pos - self.bs: self.pos])
            if not self.output_format:
                return (self.X[self.pos-self.bs: self.pos],
                        self.y[self.pos - self.bs: self.pos],
                        hints)
            else:
                hints = self._grab_hints(self.hints[self.pos - self.bs: self.pos])
                return self.output_format(self.X[self.pos-self.bs:self.pos],
                                          self.y[self.pos-self.bs:self.pos],
                                          hints)
        else:
            self.pos = len(self.X)
            hints = self._grab_hints(self.hints)
            if not self.output_format:
                return self.X, self.y, hints
            else:
                return self.output_format(self.X,
                                          self.y,
                                          hints)



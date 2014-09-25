'''
This is a test of the deep RNN

'''
from groundhog.datasets.NParity_dataset import NParityIterator

from groundhog.trainer.SGD import SGD
from groundhog.mainLoop import MainLoop

from groundhog.layers import MultiLayer, \
                             SoftmaxLayer, LastState, \
                             DropOp, UnaryOp, Operator, Shift, \
                             GaussianNoise, \
                             MultiSoftmaxClassificationLayer

from groundhog.layers import maxpool, maxpool_ntimes, minpool, minpool_ntimes, \
                             last, last_ntimes, \
                             tanh, sigmoid, rectifier, hard_sigmoid, hard_tanh

from groundhog.models import Classification_Model

import groundhog.utils as utils
from theano.sandbox.scan import scan

import numpy
import math
import theano
import theano.tensor as TT

import cPickle as pkl

theano.config.allow_gc = True

def get_data(state):

    new_format = lambda x, y: {'x': x, 'y': y}
    out_format = lambda x, y: new_format(x, y)
    path = "/data/lisa/exp/caglargul/codes/python/nbit_parity_data/par_fil_npar_2_nsamp_4_det.npy"

    if state["bs"] == "full":
        train_data = NParityIterator(batch_size = state['bs'],
                                     start=0,
                                     stop=90000,
                                     max_iters=200,
                                     path=path)

        valid_data = NParityIterator(batch_size = state['bs'],
                                     start=90000,
                                     stop=95000,
                                     max_iters=1,
                                     path=path)

        test_data = NParityIterator(batch_size = state['bs'],
                                    start=95000,
                                    stop=100000,
                                    max_iters=1,
                                    path=path)
    else:
        train_data = NParityIterator(batch_size = int(state['bs']),
                                     start=0,
                                     stop=4,
                                     max_iters=200,
                                     path=path)
        valid_data = NParityIterator(batch_size = int(state['bs']),
                                     start=0,
                                     stop=4,
                                     max_iters=1,
                                     path=path)
        test_data = None

        """
        test_data = NParityIterator(batch_size = int(state['bs']),
                                    start=95000,
                                    stop=100000,
                                    max_iters=1,
                                    path=path)
        """
    return train_data, valid_data, test_data


rect = 'lambda x:x*(x>0)'
htanh = 'lambda x:x*(x>-1)*(x<1)'

def maxout(x):
    shape = x.shape
    if x.ndim == 1:
        shape1 = TT.cast(shape[0] / state['maxout_part'], 'int64')
        shape2 = TT.cast(state['maxout_part'], 'int64')
        x = x.reshape([shape1, shape2])
        x = x.max(1)
    else:
        shape1 = TT.cast(shape[1] / state['maxout_part'], 'int64')
        shape2 = TT.cast(state['maxout_part'], 'int64')
        x = x.reshape([shape[0], shape1, shape2])
        x = x.max(2)
    return x

def powerup(x, p=None, c=None):
    eps = 1e-10
    p_activ = TT.nnet.softplus(p) + 1
    shape = x.shape

    if x.ndim == 1:
        shape1 = TT.cast(shape[0] / state['maxout_part'], 'int64')
        shape2 = TT.cast(state['maxout_part'], 'int64')
        x_pooled = TT.maximum(abs(x - c), eps).reshape([shape1, shape2])**p_activ.dimshuffle(0, 'x')
        x = x_pooled.sum(axis=1)
        x = x**(numpy.float32(1.0) / p_activ)
    elif x.ndim == 2:
        shape1 = TT.cast(shape[1] / state['maxout_part'], 'int64')
        shape2 = TT.cast(state['maxout_part'], 'int64')
        x_pooled = TT.maximum(abs(x - c.dimshuffle('x',0)), eps).reshape([shape[0], shape1, shape2])**p_activ.dimshuffle('x', 0, 'x')
        x = x_pooled.sum(axis=2)
        x = x**(numpy.float32(1.0) / p_activ.dimshuffle('x',0))
    return x

def jobman(state, channel):
    # load dataset
    state['nouts'] = 2
    state['nins'] = 2
    rng = numpy.random.RandomState(state['seed'])
    train_data, valid_data, test_data = get_data(state)

    ########### Training graph #####################
    ## 1. Inputs
    x_train = TT.matrix('x', dtype='float32')
    x_valid = TT.matrix('x', dtype='float32')

    y = TT.lvector('y')
    hints = TT.lvector('hints')

    # 2. Layers and Operators
    bs = state['bs']
    n_pieces = 1

    if state['activ'] == 'powerup' or state['activ'] == 'maxout':
        n_pieces = state['maxout_part']

    mlp = MultiLayer(rng,
                     n_in=state['nins'],
                     n_hids=eval(state['dim']),
                     activation=eval(state['activ']),
                     init_fn=state['weight_init_fn'],
                     scale=state['weight_scale'],
                     learn_bias=True,
                     name='mlp')

    if state['activ'] == 'powerup' or state['activ'] == 'maxout':
        pendim = eval(state['dim'])[-1] / state['maxout_part']
    else:
        pendim = eval(state['dim'])[-1]

    print("pendim ", pendim)

    output_layer = SoftmaxLayer(rng,
                                pendim,
                                state['nouts'],
                                bias_scale=state['out_bias_scale'],
                                scale=state['weight_scale'],
                                sparsity=state['out_sparse'],
                                init_fn=state['weight_init_fn'],
                                sum_over_time=False,
                                name='out')

    def update_lr(obj, cost):
        stp = obj.step

        if isinstance(obj.state['lr_start'], int) and stp > obj.state['lr_start']:
            time = float(stp - obj.state['lr_start'])
            if obj.state['lr_adapt_exp']:
                if time % obj.state['lr_tau'] == 0 and stp > 0:
                    new_lr = obj.lr * obj.state['lr_beta']
                else:
                    new_lr = obj.lr
            else:
                new_lr = obj.state['clr']/(1 + time / obj.state['lr_beta'])
            obj.lr = new_lr

    if state['lr_adapt']:
        output_layer.add_schedule(update_lr)

    output_sto = output_layer(mlp(x_train))
    train_model = output_sto.train(target=y) / TT.cast(y.shape[0], 'float32')

    valid_model = output_layer(mlp(x_valid,
                                   use_noise=False),
                                   use_noise=False).validate(target=y)

    valid_fn = theano.function([x_valid, y],
                               [valid_model.cost,
                                valid_model.model_output],
                                name = 'valid_fn',
                                on_unused_input='warn')

    model = Classification_Model(cost_layer = train_model,
                                 valid_fn = valid_fn,
                                 sample_fn = None,
                                 clean_before_noise_fn = False,
                                 rng = rng)

    if state['activ'] == 'powerup' and state['exclude_powers']:
        # do not apply momentum to p and c
        model.momentum_exclude = [x['p'] for x in mlp.cc_params] + \
                [x['c'] for x in mlp.cc_params]

    algo = SGD(model, state, train_data)
    hooks = []
    main = MainLoop(train_data,
                    valid_data,
                    test_data,
                    model,
                    algo, state, channel,
                    reset = state['reset'], hooks = hooks)

    if state['reload']:
        main.load()
    main.main()


if __name__=='__main__':
    state = {}

    state['nclasses'] = 2
    state['reload'] = False
    state['dim'] = '[5]' #5000
    state['activ'] = 'lambda x: TT.tanh(x)'
    state['bias'] = 0.
    state['exclude_powers'] = False
    state['maxout_part'] = 1.

    state['weight_init_fn'] = 'sample_weights_classic'
    state['weight_scale'] = 0.01

    state['lr'] = .15
    state['minlr'] = 1e-8
    state['moment'] = .42
    state['switch'] = 500 * 5

    state['cutoff'] = 0.
    state['cutoff_rescale_length'] = 0.
    state["cpu_subspace"] = True

    state['lr_adapt'] = True
    state['lr_adapt_exp'] = False
    state['lr_beta'] = 500. * 100. # 0.99
    state['lr_start'] = 0
    state['lr_tau'] = 500.

    state['patience'] = -1
    state['divide_lr'] = 1.#2.
    state['cost_threshold'] = 1.0002

    state['max_norm'] = 0.

    state['bs']  = 1
    state['reset'] = -1

    state['loopIters'] = 500 * 500
    state['timeStop'] = 24*60*7
    state['minerr'] = -1

    state['seed'] = 123

    state['trainFreq'] = 1
    state['validFreq'] = 10
    state['hookFreq'] =  20
    state['saveFreq'] = 100

    state['profile'] = 0
    state['out_sparse'] = -1
    state['out_bias_scale'] = -0.5
    state['prefix'] = 'model_wmlp_'
    state['overwrite'] = 1

    jobman(state, None)

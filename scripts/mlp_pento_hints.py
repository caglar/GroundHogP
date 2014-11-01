'''
This is a test of the deep RNN

'''
from groundhog.datasets.NParity_dataset import NParityIterator
from groundhog.datasets.Pentomino_dataset import PentominoTensorIterator

from groundhog.trainer.SGD_hessapprox3 import SGD
#from groundhog.trainer.vsgd import SGD

from groundhog.mainLoop import MainLoop

from groundhog.layers import MultiLayer, \
                             SoftmaxLayer, LastState, \
                             DropOp, UnaryOp, Operator, Shift, \
                             GaussianNoise, \
                             ParallelOutputLayer

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

    names = [ "pento64x64_80k_64patches_seed_735128712_64patches.npy" ]

    # "pento64x64_40k_64patches_seed_975168712_64patches.npy"]

    if state["bs"] == "full":
        train_data = NParityIterator(batch_size = state['bs'],
                                     start=0,
                                     stop=90000,
                                     max_iters=20000,
                                     names=names,
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
        print "Loading the training set..."
        train_data = PentominoIterator(batch_size = int(state['bs']),
                                       start=0,
                                       names=names,
                                       stop=70000,
                                       output_format=out_format)

        print "Loading the validation set..."
        """
        valid_data = PentominoIterator(batch_size = int(state['bs']),
                                       start=0,
                                       stop=70000,
                                       names=names,
                                       use_infinite_loop=False,
                                       output_format=out_format,
                                       mode="valid")
        """
        print "Loading the test set..."
        valid_data = PentominoIterator(batch_size = int(state['bs']),
                                       start=70000,
                                       stop=80000,
                                       names=names,
                                       use_infinite_loop=False,
                                       output_format=out_format,
                                       mode="valid")

        #valid_data = train_data
        test_data = None

    return train_data, valid_data, test_data


def get_data(state):

    new_format = lambda x, y: {'x': x, 'y': y}
    out_format = lambda x, y: new_format(x, y)

    names = [ "pento64x64_80k_64patches_seed_735128712_64patches.npy" ]

    # "pento64x64_40k_64patches_seed_975168712_64patches.npy"]

    print "Loading the training set..."
    train_data = PentominoTensorIterator(batch_size = int(state['bs']),
                                         start=0,
                                         names=names,
                                         stop=70000,
                                         output_format=out_format)

    print "Loading the validation set..."
    """
    valid_data = PentominoIterator(batch_size = int(state['bs']),
                                   start=0,
                                   stop=70000,
                                   names=names,
                                   use_infinite_loop=False,
                                   output_format=out_format,
                                   mode="valid")
    """


    valid_data = PentominoTensorIterator(batch_size = int(state['bs']),
                                         start=70000,
                                         stop=80000,
                                         names=names,
                                         use_infinite_loop=False,
                                         output_format=out_format,
                                         mode="valid")

    print "Loading the test set..."

    #valid_data = train_data
    test_data = None

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
    state['nins'] = 64*64

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

    #dropper = DropOp(dropout=state['dropout'], rng=rng)

    if "nlayers" in state:
        nlayers = state['nlayers']
    else:
        nlayers = 1

    layers = []
    mlp = MultiLayer(rng,
                     n_in=state['nins'],
                     dropout=state['dropout'],
                     n_hids=eval(state['dim']),
                     activation=eval(state['activ']),
                     init_fn=state['weight_init_fn'],
                     scale=state['weight_scale'],
                     learn_bias=True,
                     name='mlp_layer_%d' % 0)

    layers.append(mlp)
    for i in xrange(1, nlayers):
        mlp = MultiLayer(rng,
                         n_in=eval(state['dim']),
                         n_hids=eval(state['dim']),
                         activation=eval(state['activ']),
                         #dropout=state['dropout'],
                         init_fn=state['weight_init_fn'],
                         scale=state['weight_scale'],
                         learn_bias=True,
                         name='mlp_layer_%d' % i)

        layers.append(mlp)

    if state['activ'] == 'powerup' or state['activ'] == 'maxout':
        pendim = eval(state['dim']) / state['maxout_part']
    else:
        pendim = eval(state['dim'])

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
                new_lr = obj.state['clr'] / (1 + time / obj.state['lr_beta'])
            obj.lr = new_lr

    if state['lr_adapt']:
        output_layer.add_schedule(update_lr)

    mlayers = []
    mlp_layer_out = layers[0](x_train)

    for i in xrange(1, nlayers):
        mlp_layer_out = layers[i](mlp_layer_out)

    output_sto = output_layer(mlp_layer_out)
    train_model = output_sto.train(target=y) / TT.cast(y.shape[0], 'float32')

    mlp_layer_out = layers[0](x_valid)

    for i in xrange(1, nlayers):
        mlp_layer_out = layers[i](mlp_layer_out, use_noise=False)

    valid_model = output_layer(mlp_layer_out,
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
    state['dim'] = '800' #5000
    state['activ'] = 'lambda x: TT.maximum(x, 0)'
    state['bias'] = 0.
    state['exclude_powers'] = False
    state['maxout_part'] = 1.

    state['nlayers'] = 2
    state['weight_init_fn'] = 'sample_weights_orth'
    state['weight_scale'] = 0.01

    state['lr'] = .15
    state['minlr'] = 1e-8

    state['switch'] = 100

    state['cutoff'] = 0.
    state['cutoff_rescale_length'] = 0.

    state['lr_adapt'] = False
    state['lr_adapt_exp'] = False
    state['lr_beta'] = 500. * 100. # 0.99
    state['lr_start'] = 0
    state['lr_tau'] = 500.

    state['patience'] = -1
    state['divide_lr'] = 1.#2.
    state['cost_threshold'] = 1.0002

    state['max_norm'] = 0.

    state['bs']  = 1000
    state['reset'] = -1

    state['loopIters'] = 10000*500
    state['timeStop'] = 24*60*100
    state['minerr'] = -1

    state['seed'] = 123
    state['correction'] = 1.0
    state['trainFreq'] = 100
    state['validFreq'] = 300
    state['hookFreq'] =  100
    state['saveFreq'] = 60

    state['out_sparse'] = -1
    state['out_bias_scale'] = -0.5
    state['prefix'] = 'model_wmlp_pento_'
    state['overwrite'] = 1
    state['dropout'] = 0.5

    jobman(state, None)

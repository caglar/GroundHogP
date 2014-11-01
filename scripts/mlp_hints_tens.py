'''
This is a test of the deep RNN

'''
from groundhog.datasets.NParity_dataset import NParityIterator
from groundhog.datasets.Pentomino_dataset import PentominoTensorIterator
from groundhog.datasets.Pentomino_dataset import PentominoTensorHintsIterator

#from groundhog.trainer.SGD_hessapprox3 import SGD
from groundhog.trainer.SGD_adadelta import SGD

from groundhog.mainLoop import MainLoop

from groundhog.layers import MultiLayer, MultiTensorLayer, \
                             SoftmaxLayer, LastState, SigmoidLayer, \
                             DropOp, UnaryOp, Operator, Shift, \
                             GaussianNoise, ParallelOutputLayer

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
np = numpy
import cPickle as pkl

theano.config.allow_gc = True

def get_data(state):

    new_format = lambda x, y, z: {'x': x, 'y': y, 'hints': z}
    out_format = lambda x, y, z: new_format(x, y, z)

    path = "/data/lisa/exp/caglargul/codes/python/nbit_parity_data/par_fil_npar_20_nsamp_100000_det2.npy"
    names = [ "pento64x64_80k_64patches_seed_735128712_64patches.npy" ]

    if state["bs"] == "full":
        train_data = NParityIterator(batch_size = int(state['bs']),
                                     start=0,
                                     stop=90000,
                                     max_iters=2000000,
                                     use_hints=True,
                                     path=path)

        valid_data = NParityIterator(batch_size = int(state['bs']),
                                     start=90000,
                                     stop=95000,
                                     max_iters=1,
                                     use_hints=True,
                                     names=names)

        test_data = NParityIterator(batch_size = int(state['bs']),
                                    start=95000,
                                    stop=100000,
                                    max_iters=1,
                                    use_hints=True,
                                    path=path)
    else:
        train_data = PentominoTensorHintsIterator(batch_size = int(state['bs']),
                                                  start=0,
                                                  stop=80000,
                                                  use_infinite_loop=True,
                                                  output_format=out_format,
                                                  names=names)

        valid_data = PentominoTensorHintsIterator(batch_size = int(state['bs']),
                                                  start=0,
                                                  stop=80000,
                                                  output_format=out_format,
                                                  use_infinite_loop=False,
                                                  names=names)
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
        x_pooled = TT.maximum(abs(x - c.dimshuffle('x', 0)), eps).reshape([shape[0], shape1, shape2])**p_activ.dimshuffle('x', 0, 'x')
        x = x_pooled.sum(axis=2)
        x = x**(numpy.float32(1.0) / p_activ.dimshuffle('x', 0))

    return x

def jobman(state, channel):
    # load dataset
    state['nins'] = 64
    rng = numpy.random.RandomState(state['seed'])
    train_data, valid_data, test_data = get_data(state)

    ########### Training graph #####################
    ## 1. Inputs
    x_train = TT.tensor3('x', dtype='float32')
    x_valid = TT.tensor3('x', dtype='float32')

    y = TT.lvector('y')
    y_valid = TT.lvector('y')
    hints = TT.matrix('hints', dtype="float32")
    hints_valid = TT.matrix('hints', dtype="float32")

    if "debug" in   state and state["debug"]:
        theano.config.compute_test_value = 'warn'
        x_train.tag.test_value = np.random.uniform(-1, 1, (state['bs'], 64, 64)).astype("float32")
        x_valid.tag.test_value = np.random.uniform(-1, 1, (state['bs'], 64, 64)).astype("float32")
        y.tag.test_value = np.zeros((state['bs'],)).astype('int64')
        y_valid.tag.test_value = np.zeros((state['bs'],)).astype('int64')
        hints.tag.test_value = np.zeros((state['bs'], state['nouts'][1])).astype("float32")
        hints_valid.tag.test_value = np.zeros((state['bs'], state['nouts'][1])).astype("float32")

    # 2. Layers and Operators
    bs = state['bs']
    n_pieces = 1

    if state['activ'] == 'powerup' or state['activ'] == 'maxout':
        n_pieces = state['maxout_part']

    ff_layers = []
    nin = state['nins']

    mlp = MultiTensorLayer(rng,
                           n_in=nin,
                           n_hids=eval(state['dim']),
                           activation=eval(state['activ']),
                           init_fn=state['weight_init_fn'],
                           scale=state['weight_scale'],
                           learn_bias=True,
                           name='mlp_%d')

    nout2 = eval(state['dim'])[-1]*64
    mlp2 = MultiLayer(rng,
                            n_in=nout2,
                            n_hids=eval(state['dim'])[-1],
                            activation=eval(state['activ']),
                            init_fn=state['weight_init_fn'],
                            scale=state['weight_scale'],
                            learn_bias=True,
                            name='mlp_%d')


    if state['activ'] == 'powerup' or state['activ'] == 'maxout':
        pendim = eval(state['dim'])[-1] / state['maxout_part']
    else:
        pendim = eval(state['dim'])[-1]

    print("pendim ", pendim)
    output_layer1 = SoftmaxLayer(rng,
                                n_in=pendim,
                                n_out=state['nouts'][0],
                                bias_scale=state['out_bias_scale'],
                                scale=state['weight_scale'],
                                sparsity=state['out_sparse'],
                                init_fn=state['weight_init_fn'],
                                sum_over_time=True,
                                name='out_1')

    output_layer2 = SigmoidLayer(rng,
                                n_in=pendim,
                                n_out=state['nouts'][1],
                                bias_scale=state['out_bias_scale'],
                                scale=state['weight_scale'],
                                sparsity=state['out_sparse'],
                                init_fn=state['weight_init_fn'],
                                sum_over_time=True,
                                name='out_1')


    output_layer = ParallelOutputLayer([output_layer1, output_layer2],
                                       cost_scales=state['cost_scales'])


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

    mlp_out = mlp(x_train)
    mlp_out = mlp2(mlp_out.out.reshape((-1, 64*pendim)))

    """
    for i in xrange(state['nlayers']):
        mlp_out = ff_layers[i](mlp_out)
    """

    targets=[y, hints]
    output_sto = output_layer(mlp_out,
                              targets=targets)


    train_model = output_sto.train(targets=targets) / TT.cast(y.shape[0], 'float32')

    mlp_valid_out = mlp(x_valid)

    """
    for i in xrange(state['nlayers']):
        mlp_valid_out = ff_layers[i](mlp_valid_out,
                                     use_noise=False)
    """

    mlp_valid_out = mlp2(mlp_valid_out.out.reshape((-1, 64*pendim)))

    targets_valid = [y_valid, hints_valid]

    valid_model = output_layer(mlp_valid_out,
                               targets=targets_valid,
                               use_noise=False).validate(targets=targets_valid)

    valid_fn = theano.function([x_valid, y_valid, hints_valid],
                               [valid_model.cost,
                                valid_model.out[0],
                                valid_model.out[1]],
                                name = 'valid_fn',
                                on_unused_input='warn')

    model = Classification_Model(cost_layer = train_model,
                                 valid_fn = valid_fn,
                                 sample_fn = None,
                                 clean_before_noise_fn = False,
                                 rng = rng)
    model.inputs += [hints]

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

    state['nclasses'] = 3
    state['reload'] = False
    state['dim'] = '[240, 240, 240]' #5000
    state['debug'] = False

    state['activ'] = 'lambda x: TT.maximum(x, 0)'
    state['bias'] = 0.
    state['exclude_powers'] = False
    state['maxout_part'] = 1.
    state['nlayers'] = 2

    #state['momentum'] = 1
    state['weight_init_fn'] = 'sample_weights_orth'
    state['weight_scale'] = 0.01

    state['lr'] = .15
    state['minlr'] = 1e-8
    state['moment'] = .55
    state['momentum'] = 1.

    state['switch'] = 50

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

    state['bs']  = 1000
    state['reset'] = -1

    state['loopIters'] = 50000 * 100
    state['timeStop'] = 24*60*7
    state['minerr'] = -1

    state['seed'] = 123

    state['trainFreq'] = 2 #100
    state['validFreq'] = 1000 #1000
    state['hookFreq'] =  2 #1000
    state['saveFreq'] = 500

    state['profile'] = 0
    state['out_sparse'] = -1
    state['out_bias_scale'] = -0.5
    state['prefix'] = 'model_wmlp_'
    state['overwrite'] = 1

    state["nouts"] = [2, 11]

    state["cost_scales"] = [0.4, 0.6]

    jobman(state, None)

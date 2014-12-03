"""
Test of the classical LM model for language modelling
"""

from groundhog.datasets.LM_dataset import LMIterator
from groundhog.trainer.SGD_rmsprop2 import SGD
from groundhog.mainLoop import MainLoop

from groundhog.layers import MultiLayer, \
                             RecurrentMultiLayer, \
                             SoftmaxLayer, \
                             RecurrentLayer

from groundhog.models import LM_Model
from theano.sandbox.scan import scan

import numpy
import theano
import theano.tensor as TT

theano.config.allow_gc = True

rect = lambda x: TT.maximum(0, x)

def get_data(state):

    def out_format (x, y, r):
        return {'x':x, 'y' :y}# 'reset': r}

    def out_format_valid (x, y, r):
        return {'x':x, 'y' :y, 'reset': r}

    train_data = LMIterator(batch_size=state['bs'],
                            path="/data/lisa/data/PennTreebankCorpus/pentree_char_and_word.npz",
                            stop=-1,
                            seq_len=state['seqlen'],
                            mode="train",
                            chunks=state["chunks"],
                            shift=1,
                            output_format = out_format,
                            can_fit=True)

    valid_data = LMIterator(batch_size=state['val_bs'],
                            path="/data/lisa/data/PennTreebankCorpus/pentree_char_and_word.npz",
                            stop=-1,
                            use_infinite_loop=False,
                            seq_len=state['seqlen'],
                            mode="valid",
                            chunks=state["chunks"],
                            shift=1,
                            output_format = out_format_valid,
                            can_fit=True)

    test_data = LMIterator(batch_size=state['val_bs'],
                           path="/data/lisa/data/PennTreebankCorpus/pentree_char_and_word.npz",
                           stop=-1,
                           use_infinite_loop=False,
                           allow_short_sequences=True,
                           seq_len=state['seqlen'],
                           mode="test",
                           chunks=state["chunks"],
                           shift = 1,
                           output_format = out_format_valid,
                           can_fit=True)

    return train_data, valid_data, test_data

def jobman(state, channel):

    if state['debug']:
        theano.config.compute_test_value = 'warn'

    # load dataset
    rng = numpy.random.RandomState(state['seed'])
    train_data, valid_data, test_data = get_data(state)

    if state['chunks'] == 'words':
        state['nin'] = 10000
        state['nout'] = 10000
    else:
        state['nin'] = 50
        state['nout'] = 50

    ########### Training graph #####################
    # 2. Layers and Operators
    bs = state['bs']

    emb_words = MultiLayer(
        rng,
        n_in=state['nin'],
        n_hids=state['nhids'],
        activation='lambda x:x',
        init_fn='sample_weights_classic',
        weight_noise=state['weight_noise'],
        scale=.01,
        sparsity=-1,
        bias_scale=0,
        name='emb_words')


    rec = RecurrentLayer(rng,
                         state['nhids'],
                         activation = 'TT.nnet.sigmoid',
                         bias_scale=0,
                         scale=0.01,
                         sparsity=-1,
                         init_fn="sample_weights_orth",
                         weight_noise = state['weight_noise'],
                         name = 'rec')

    output_layer = SoftmaxLayer(rng,
                                n_in=state['nhids'],
                                n_out=state['nout'],
                                scale=.01,
                                init_fn="sample_weights_classic",
                                weight_noise=state['weight_noise'],
                                sparsity=-1,
                                sum_over_time=True,
                                use_nce=False,
                                name='out')
    nsteps = 0
    if state['bs'] == 1:
        tx = TT.lvector('x')
        ty = TT.lvector('y')
        nsteps = state['seqlen']
    else:
        tx = TT.lmatrix('x')
        ty = TT.lmatrix('y')
        nsteps = state['seqlen']
        if state['debug']:
            tx.tag.test_value = numpy.random.random_integers(0, 10,
            size=(state['bs'], state['seqlen']))
            ty.tag.test_value = numpy.random.random_integers(0, 10,
            size=(state['bs'], state['seqlen']))


    reset = TT.scalar('reset')

    if state['bs'] > 1:
        h0 = TT.alloc(numpy.float32(0), state['bs'], state['nhids'])
    else:
        h0 = TT.alloc(numpy.float32(0), state['nhids'])

    emb_words_rep = emb_words(tx)

    # 3. Constructing the model
    rec_layer = rec(state_below = emb_words_rep(tx),
                    nsteps = nsteps,
                    init_state = h0,
                    batch_size = state['bs'])

    train_model = output_layer.train(state_below=rec_layer,
                                     target=ty,
                                     reg=None,
                                     mask=None,
                                     scale=numpy.float32(1. / state['seqlen']))
    vnsteps = 0
    if state['val_bs'] == 1:
        vx = TT.lvector('x')
        vy = TT.lvector('y')
        vnsteps = vy.shape[0]
    else:
        vx = TT.lmatrix('x')
        vy = TT.lmatrix('y')
        vnsteps = vy.shape[1]

    reset = TT.scalar('reset')

    if state['val_bs'] == 1:
        h0 = theano.shared(numpy.zeros((state['nhids'],)).astype("float32"))
    else:
        h0 = theano.shared(numpy.zeros((state['val_bs'], state['nhids'])).astype("float32"))

    rec_layer = rec(state_below=emb_words_rep(vx, use_noise=False),
                    nsteps = vnsteps,
                    init_state = h0*reset,
                    use_noise = False,
                    batch_size=state['val_bs'])

    nw_h0 = rec_layer.out[-1]
    valid_model = output_layer(rec_layer,
                               use_noise=False).validate(target=vy,
                                                         sum_over_time=True)

    valid_fn = theano.function([vx, vy, reset],
                               (valid_model.cost),
                               name='valid_fn',
                               updates=[(h0, nw_h0)])

    model = LM_Model(cost_layer = train_model,
                     weight_noise_amount=state['weight_noise_amount'],
                     valid_fn = valid_fn,
                     clean_before_noise_fn = False,
                     noise_fn = None,
                     rng = rng)

    algo = SGD(model, state, train_data)
    main = MainLoop(train_data,
                    valid_data,
                    test_data,
                    model, algo,
                    state, channel)

    if state['reload']:
        main.load()
    main.main()


if __name__=='__main__':
    state = {}

    state["chunks"] = "chars"
    state['reload'] = False
    state['noisy_ll'] = 0
    state['nhids'] = 500

    state['reseting'] = True
    state['lr'] = 0.06
    state['minlr'] = 0.01

    state['dictionary']= "/data/lisa/data/PennTreebankCorpus/dictionaries.npz"
    state['momentum'] = 1.0
    state['cutoff_rescale_length'] = False

    state['weight_noise'] = False
    state['weight_noise_amount'] = 0.1
    state['reseting'] = False
    state['bs']  = 200
    state['val_bs'] = 1
    state['reset'] = -1
    state['seqlen'] = 160

    state['loopIters'] = 15000000
    state['timeStop'] = 320 * 6000
    state['minerr'] = -1
    state['debug'] = False
    state['debug_state'] = False

    state['seed'] = 123
    state["correction"] = 1.0
    state["switch"] = 200
    state['rmsprop_decay'] = 0.95
    state['max_lr_scale'] = 50

    state['trainFreq'] = 50
    state['hookFreq'] = 0
    state['validFreq'] = 300
    state['saveFreq'] = 60

    state['prefix'] = 'penn_rnn2_corr_rmsprop0.5_v3'
    state['overwrite'] = 1
    state['correction'] = 1.0
    state['patience'] = 3
    state['divide_lr'] = 2.
    state['cost_threshold'] = 1.003

    jobman(state, None)
import numpy
import time
import logging

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const

logger = logging.getLogger(__name__)


class SGD(object):

    def __init__(self,
                 model,
                 state,
                 data):
        """
        This class implements the Nesterov Momentum in the formulation at the below paper:
            http://arxiv.org/pdf/1212.0901v2.pdf
        With rmsprop as well.
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
            :param rmsprop_decay:
                Decay parameter for rmsprop.
        """
        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))

        print("My state is: ")
        print(state)
        print("RMSPROPPPP!!!")

        if "rmsprop_decay" in state:
            self.rmsprop_decay = state["rmsprop_decay"]
        else:
            self.rmsprop_decay = 0.95

        if "adaeps" not in state:
            state["adaeps"] = 1e-6

        if "max_lr_scale" not in state:
            state["max_lr_scale"] = 50

        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]
        self.rms_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name="rms_%s" % p.name)
                   for p in model.params]
        self.mean_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name="meangs_%s" % p.name)
                   for p in model.params]
        lr = TT.scalar('lr')
        self.eps = state["adaeps"]

        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

        if 'profile' not in self.state:
            self.state['profile'] = 0

        ###################################
        # Step 1. Compile training function
        ###################################
        print('Constructing grad function')

        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(model.param_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))

        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        # Clip the momentum-applied gradient
        moment_gs = []
        rms_gs = []
        mean_gs = []

        if len(moment_gs) > 0:
            gs = moment_gs
        else:
            moment_gs = gs

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(moment_gs, self.model.params) if p not in self.model.params))

        store_gs = zip(self.gs, gs)
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]
        for p, g, mg, rms in zip(self.model.params, gs, self.mean_gs, self.rms_gs):
            print "Using rmsprop for param %s" % p.name
            mg_t = self.rmsprop_decay * mg + (1 - self.rmsprop_decay) * g
            r_t = self.rmsprop_decay * rms + (1 - self.rmsprop_decay) * g**2
            rms_t = TT.sqrt(r_t - mg_t**2 + self.eps)
            rms_gs.append(r_t)
            mean_gs.append(mg_t)
            new_lr = 1. / rms_t

            if 'max_lr_scale' in state:
                new_lr = TT.minimum(state['max_lr_scale']*lr, new_lr)
            moment_gs.append(g * new_lr)

        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []

            for g, p in zip(moment_gs, self.model.params):
                if p not in self.model.params:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1) * p,
                           tmpg))
                else:
                    _gs.append(g)

            gs = _gs

            moment_gs = gs

        store_rms = zip(self.rms_gs, rms_gs)
        store_mgs = zip(self.mean_gs, mean_gs)

        updates += store_rms
        updates += store_mgs

        print('Compiling grad function')
        st = time.time()
        self.train_fn = theano.function([], outs,
                                        name='train_function',
                                        updates = updates,
                                        givens = zip(model.inputs, loc_data),
                                        profile=self.state['profile'])

        print 'took', time.time() - st

        self.lr = numpy.float32(state['lr'])
        new_params = [p - self.lr * g for p, g in zip(model.params, moment_gs)]
        updates = zip(model.params, new_params)
        self.update_fn = theano.function([], [], name='update_function',
                                         allow_input_downcast=True,
                                         on_unused_input="warn",
                                         updates = updates,
                                         profile=self.state['profile'])

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                       ['cost',
                        'error',
                        'time_step',
                        'whole_time',
                        'lr']

    def __call__(self):

        batch = self.data.next()
        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)

        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)

        # Run the training function
        g_st = time.time()
        rvals = self.train_fn()

        for schedule in self.schedules:
            schedule(self, rvals[-1])

        self.update_fn()
        g_ed = time.time()

        self.state['lr'] = float(self.lr)
        cost = rvals[-1]

        if numpy.isnan(cost) or numpy.isinf(cost):
            raise Exception('Got NaN in the cost!')

        self.old_cost = cost
        whole_time = time.time() - self.step_timer

        if self.step % self.state['trainFreq'] == 0:

            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]

            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]

            msg += ' step time %s whole time %s lr %.2e'

            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr)]

            print msg % tuple(vals)

        self.step += 1

        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                    ('lr', float(self.lr)),
                    ('time_step', float(g_ed - g_st)),
                    ('whole_time', float(whole_time))] + zip(self.prop_names, rvals))

        return ret

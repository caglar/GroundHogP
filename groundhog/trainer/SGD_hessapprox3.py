"""
Stochastic Gradient Descent.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import time
import logging

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const
np = numpy

logger = logging.getLogger(__name__)

class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
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
        """

        if 'adarho' not in state:
            state['adarho'] = 0.96

        if 'adaeps' not in state:
            state['adaeps'] = 1e-6

        if 'moment' not in state:
            self.mom = -1
        elif state['moment'] > 0:
            self.mom = state['moment']

        if 'correction' not in state:
            self.correction = 0.
        elif state['correction'] > 0:
            self.correction = state['correction']

        f32 = np.float32
        eps = f32(1e-6)

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))

        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]

        self.gs_sqr = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name + "_sqr")
                   for p in model.params]

        self.gamma_denom = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name + "_denom")
                   for p in model.params]

        self.gamma_nume = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name + "_nume")
                   for p in model.params]

        self.taus = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(2.3),
                                name=p.name + "_taus")
                   for p in model.params]

        self.old_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(eps),
                                name=p.name+'_old_gs')
                    for p in model.params]

        self.curv_sqr = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(eps),
                                name=p.name+'_curv_sqr')
                    for p in model.params]

        self.ave_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(eps),
                                name=p.name+'_ave_gs')
                    for p in model.params]

        self.delta_sqr_gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(eps),
                                name=p.name+'_delta_sqr_gs')
                    for p in model.params]

        self.ave_delta = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX) + numpy.float32(eps),
                                name=p.name+'_ave_delta')
                    for p in model.params]

        self.stp_cnt = theano.shared(f32(0),
                                     name=p.name+'_stp_cnt')


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
        logger.debug('Constructing grad function')
        loc_data = self.gdata
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]

        #rval = theano.clone(model.param_grads + self.update_rules + \
        #                    self.prop_exprs + [model.train_cost],
        #                    replace=zip(model.inputs, loc_data))
        rval = model.param_grads + self.update_rules + \
                    self.prop_exprs + [model.train_cost]

        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
                        for x,p in zip(gs, self.model.params) if p not in self.model.exclude_params_for_norm))

        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []

            for g, p in zip(gs, self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g * c / norm_gs, g)
                    _gs.append(
                    TT.switch(notfinite, numpy.float32(.1)*p, tmpg))
                else:
                    _gs.append(g)

            gs = _gs

        normed_gs = [g/norm_gs for g in self.gs]
        store_gs = [(s, g) for s, g in zip(self.gs, gs)]
        updates = store_gs + [(s[0], r) for s, r in zip(model.updates, rules)]

        rho = np.float32(self.state['adarho'])
        eps = np.float32(self.state['adaeps'])

        # grad2
        ave_gs_lst = [ rho * agn2 + (f32(1) - rho) * g for agn2, g, tau in zip(self.ave_gs,
                                                                               normed_gs, self.taus)]
        gs_sqr_lst = [ rho * g_sqr + (f32(1) - rho) * TT.sqr(g) for g_sqr, g, tau in zip(self.gs_sqr,
                                                                                         normed_gs,
                                                                                         self.taus)]

        curve_sqr_lst = [ (1 - (1 / tau)) * curve_sqr + (1 / tau) * TT.sqr((oldg) - (g))\
                        for curve_sqr, oldg, g, tau in zip(self.curv_sqr, self.old_gs, normed_gs,
                                                           self.taus)]

        if state["full_gamma"]:
            gamma_nume_lst = [ ( 1 - (1 / tau)) * gamma_nume + (1 / tau) * TT.sqr((g - oldg) * (ag
                - oldg))\
                        for gamma_nume, ag, oldg, g, tau in zip(self.gamma_nume, ave_gs_lst, self.old_gs,
                                                                normed_gs, self.taus)]

            gamma_denom_lst = [ (1 - (1 / tau)) * gamma_denom + (1 / tau) * TT.sqr((g - ag) * (ag
                - oldg))\
                        for gamma_denom, ag, oldg, g, tau in zip(self.gamma_denom, ave_gs_lst, self.old_gs,
                                                                 normed_gs, self.taus)]
        else:
            gamma_nume_lst = [ ( 1 - (1 / tau)) * gamma_nume + (1 / tau) * TT.sqr((oldg - g))\
                        for gamma_nume, ag, oldg, g, tau in zip(self.gamma_nume, ave_gs_lst, self.old_gs,
                                                                normed_gs, self.taus)]

            gamma_denom_lst = [ (1 - (1 / tau)) * gamma_denom + (1 / tau) * TT.sqr((g - ag))\
                        for gamma_denom, ag, oldg, g, tau in zip(self.gamma_denom, ave_gs_lst, self.old_gs,
                                                                 normed_gs, self.taus)]

        taus_lst = [(f32(1) - (TT.sqr(ave_delta)) / (delta_sqr + eps)) * tau + f32(1) + eps\
                        for tau, ave_delta, delta_sqr in zip(self.taus, self.ave_delta,
                                                            self.delta_sqr_gs)]

        old_gs_up = [(og, g) for og, g in zip(self.old_gs, normed_gs)]
        updates += old_gs_up

        new_delta_sqr_gs_lst = []
        ave_delta_lst = []
        corrected_grads = []
        steps = []
        stp_cnts = [(self.stp_cnt, self.stp_cnt + 1)]

        for (delta_sqr, g, curve, old_g, ave_g,
             gamm_n, gamma_d, ave_delta, tau) in zip(self.delta_sqr_gs,
                                                     normed_gs, curve_sqr_lst, self.old_gs,
                                                     ave_gs_lst, gamma_nume_lst,
                                                     gamma_denom_lst,
                                                     self.ave_delta,
                                                     self.taus):

            gamma = self.correction * TT.sqrt(gamm_n) / TT.sqrt(gamma_d + eps)

            if "momentum" in state:
                gamma = gamma.clip(-state["momentum"], state["momentum"])

            corrected_grad = (g + gamma * ave_g) /\
                    (f32(1) + gamma)

            steplen = TT.sqrt(delta_sqr + numpy.float32(eps)) /\
                    TT.sqrt(curve + numpy.float32(eps))

            if 'switch' in self.state:
                delta_x_t = TT.switch(self.state['switch'] > self.stp_cnt, steplen * corrected_grad, g)
            else:
                delta_x_t = steplen * corrected_grad

            delta_x_t_fX = TT.cast(delta_x_t.clip(-8, 8), theano.config.floatX)
            steps.append(delta_x_t_fX)
            delta_x_t_sqr = TT.sqr(delta_x_t_fX)
            new_delta_sqr_gs_lst.append(delta_sqr * (f32(1) - (f32(1) / tau)) + (f32(1) / tau) * delta_x_t_sqr)
            ave_delta_lst.append(ave_delta * (f32(1) - (f32(1) / tau)) + (f32(1) / tau) * delta_x_t_fX)

        delta_x_t_up = zip(self.delta_sqr_gs, new_delta_sqr_gs_lst)
        ave_gs_up = zip(self.ave_gs, ave_gs_lst)
        curve_sqr_up = zip(self.curv_sqr, curve_sqr_lst)
        ave_delta_up = zip(self.ave_delta, ave_delta_lst)
        gamma_nume_up = zip(self.gamma_nume, gamma_nume_lst)
        gamma_denom_up = zip(self.gamma_denom, gamma_denom_lst)
        gs_sqr_up = zip(self.gs_sqr, gs_sqr_lst)

        new_taus_lst =  []
        for g, mg, mvg, d, md, mvd, tau in zip(self.gs,
                                               self.ave_gs,
                                               self.gs_sqr,
                                               steps,
                                               self.ave_delta,
                                               self.delta_sqr_gs,
                                               taus_lst):

            new_tau = TT.switch(TT.or_(abs(g - mg) > 2 * TT.sqrt(mvg - mg**2),
                                       abs(d - md) > 2 * TT.sqrt(mvd - md**2)),
                                       theano.shared(f32(2.1)), tau)

            new_tau = TT.maximum(new_tau, f32(1.3))
            new_taus_lst.append(new_tau)

        new_taus_up = zip(self.taus, new_taus_lst)

        updates += new_taus_up
        updates += ave_delta_up
        updates += gs_sqr_up
        updates += delta_x_t_up
        updates += ave_gs_up
        updates += curve_sqr_up
        updates += gamma_nume_up
        updates += gamma_denom_up

        logger.debug('Compiling grad function')
        st = time.time()
        self.train_fn = theano.function([], outs,
                                        name='train_function',
                                        updates = updates,
                                        givens = zip(model.inputs, loc_data))

        logger.debug('took {}'.format(time.time() - st))

        self.lr = numpy.float32(1.)
        new_params = []

        for p, stp in zip(model.params, steps):
            new_params_up = (p, p - TT.cast(stp, theano.config.floatX))
            new_params.append(new_params_up)

        updates += new_params #zip(model.params, new_params)

        self.update_fn = theano.function([], [], name='update_function',
                                         allow_input_downcast=True,
                                         givens = zip(model.inputs, loc_data),
                                         updates = updates)

        self.old_cost = 1e20
        self.schedules = model.get_schedules()

        self.return_names = self.prop_names + \
                    ['cost',
                     'error',
                     'time_step',
                     'whole_time',
                     'lr']

        self.prev_batch = None

    def __call__(self):
        batch = self.data.next()
        assert batch

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

        # Run the trianing function
        g_st = time.time()
        rvals = self.train_fn()

        for schedule in self.schedules:
            schedule(self, rvals[-1])

        self.update_fn()
        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
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
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret

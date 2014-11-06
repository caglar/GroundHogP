import numpy
import time

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from mt_lisa.utils.utils import print_time, print_mem, const


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

        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        eps_fd = theano.shared(numpy.float32(1e-8), name="eps_fd")

        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_g')
                   for p in model.params]
        self.mean_gs = [theano.shared(numpy.ones(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX)*1e-5,
                                name=p.name+'_mg')
                   for p in model.params]
        self.mean_vsg = [theano.shared(numpy.ones(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX)*1e-5,
                                name=p.name+'_mvg')
                   for p in model.params]
        self.mean_vsh = [theano.shared(numpy.ones(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX)*1e-5,
                                name=p.name+'_mvh')
                   for p in model.params]
        self.mean_hs = [theano.shared(numpy.ones(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX)*1e-5,
                                name=p.name+'_mh')
                   for p in model.params]
        self.time = [theano.shared(numpy.ones(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_t')
                   for p in model.params]

        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

        ###################################
        # Step 1. Compile training function
        ###################################
        print 'Constructing grad function'
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

        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(gs, self.model.params) if p not in self.model.myparams))
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.myparams:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p,
                           tmpg))
                else:
                    _gs.append(g)
            gs = _gs
        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        updates = store_gs + [(s, r) for s,r in zip(model.updates, rules)]
        print 'Compiling grad function'
        st = time.time()
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = updates,
            profile=0)
        print 'took', time.time() - st


        gs = theano.clone(model.param_grads,
                            replace=zip(model.inputs, loc_data)+
                             [(p, p + eps_fd) for p in zip(model.params)])
        norm_gs = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(gs, self.model.params) if p not in self.model.myparams))
        if 'cutoff' in state and state['cutoff'] > 0:
            c = numpy.float32(state['cutoff'])
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_fd_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.myparams:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p,
                           tmpg))
                else:
                    _gs.append(g)
            gs = _gs

        hs = [ abs((g - ng)/mg) for g, ng, mg in zip(self.gs, gs,
                                                     self.mean_gs)]
        new_time = [
            TT.switch( TT.or_(
                abs(g - mg) > 2 * TT.sqrt(mvg-mg**2),
                abs(h - mh) > 2 * TT.sqrt(mvh-mh**2)),
                t + 1, t) for
            g,mg,h,mh, mvg, mvh, t in zip(self.gs, self.mean_gs, hs,
                                          self.mean_hs, self.mean_vsg,
                                          self.mean_vsh, self.time)]


        new_mean_gs = [
            (1-1./(t+1e-5)) * mg + 1./(t+1e-5) *  g for t, mg, g in zip(new_time,
                                                          self.mean_gs,
                                                          self.gs)]

        new_mean_vsg = [
            (1-1./(t+1e-5))* mv + 1./(t+1e-5) * g**2 for t, mv, g in zip(new_time,
                                                           self.mean_vsg,
                                                           self.gs)]

        new_mean_hs = [
            (1-1./(t+1e-5)) * mh + 1./(t+1e-5) *  h for t, mh, h in zip(new_time,
                                                          self.mean_hs,
                                                          hs)]

        new_mean_vsh = [
            (1-1./(t+1e-5))* mv + 1./(t+1e-5) * h**2 for t, mv, h in zip(new_time,
                                                           self.mean_vsh,
                                                           hs)]

        new_time = [
            (1 - mg**2/(mv+1e-5)) * t + 1 for mg, mv, t in zip(new_mean_gs,
                                                        new_mean_vsg,
                                                        new_time)]

        new_updates = zip(self.mean_gs, new_mean_gs) + \
                zip(self.mean_vsg, new_mean_vsg) + \
                zip(self.mean_hs, new_mean_hs) + \
                zip(self.mean_vsh, new_mean_vsh) + \
                zip(self.time, new_time)

        self.second_step = theano.function(
            [], [], name='second_step',
            updates = new_updates)


        lr = TT.scalar('lr')
        self.lr = numpy.float32(state['lr'])
        new_params = [p - lr * g for p, g in zip(model.params, self.gs)]
        self.update_fn = theano.function(
            [lr], [], name='update_function',
            allow_input_downcast=True,
            updates = zip(model.params, new_params),
            profile=0)


        eta = [ ((TT.sqrt(lr)+mh)/ (mvh+TT.sqrt(lr))) *
               ((mg**2+TT.sqrt(lr))/(mvg+TT.sqrt(lr))) for
                 mh, mvh, mg, mvg in zip(self.mean_hs, self.mean_vsh,
                                         self.mean_gs, self.mean_vsg)]
        mean_eta = eta[0].sum()
        nelems = eta[0].shape.prod()
        max_eta = eta[0].max()
        min_eta = eta[0].min()
        for _eta in eta[1:]:
            mean_eta += _eta.sum()
            nelems += _eta.shape.prod()
            max_eta = TT.maximum(_eta.max(), max_eta)
            min_eta = TT.minimum(_eta.min(), min_eta)
        mean_eta = mean_eta / nelems

        new_params = [p - _eta * g
                      for p, g, _eta
                      in zip(model.params, self.gs, eta)]

        self.update_fn_new = theano.function(
            [lr], [mean_eta, max_eta, min_eta], name='update_function',
            allow_input_downcast=True,
            updates = zip(model.params, new_params),
            profile=0)



        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                 'time_step',
                 'whole_time',
                  'lr',
                  'lr_mean',
                  'lr_max',
                  'lr_min']


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
        # Run the trianing function
        g_st = time.time()

        rvals = self.train_fn()
        if self.step > self.state['switch']:
            self.second_step()
        for schedule in self.schedules:
            schedule(self.step, self.gs, rvals[-1])
        if self.step < self.state['switch']+1000:
            self.update_fn(self.lr)
            lr_rval = [self.lr, self.lr, self.lr]
        else:
            lr_rval = self.update_fn_new(self.lr)

        g_ed = time.time()
        self.state['lr'] = float(self.lr)
        cost = rvals[-1]
        # if numpy.isnan(cost) or numpy.isinf(cost):
        #    raise Exception('Got NaN in the cost!')
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' step time %s whole time %s lr %.2e'
            msg += 'lr (max,mean,min) %.2e %.2e %.2e'
            vals += [print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     float(self.lr),
                     float(lr_rval[1]),
                     float(lr_rval[0]),
                     float(lr_rval[2])
                    ]
            print msg % tuple(vals)
        self.step += 1
        ret = dict([('cost', float(cost)),
                       ('lr', float(self.lr)),
                       ('lr_mean', float(lr_rval[0])),
                        ('lr_max', float(lr_rval[1])),
                        ('lr_min', float(lr_rval[2])),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret


"""
Implementation of a language model class.

TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import itertools
import logging

import cPickle as pkl

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import id_generator
from groundhog.layers.basic import Model

logger = logging.getLogger(__name__)

class Classification_Model(Model):
    """
        Classification model for feedforward.
    """
    def  __init__(self,
                  cost_layer=None,
                  sample_fn=None,
                  valid_fn=None,
                  use_hints=False,
                  noise_fn=None,
                  clean_before_noise_fn=False,
                  clean_noise_validation=True,
                  compute_accuracy=False,
                  need_inputs_for_generating_noise=False,
                  weight_noise_amount=0,
                  exclude_params_for_norm=None,
                  rng = None):

        super(Classification_Model, self).__init__(output_layer=cost_layer,
                                                   sample_fn=sample_fn,
                                                   rng=rng)

        self.use_hints = use_hints
        self.need_inputs_for_generating_noise = need_inputs_for_generating_noise
        self.cost_layer = cost_layer
        self.validate_step = valid_fn
        self.clean_noise_validation = clean_noise_validation
        self.noise_fn = noise_fn
        self.clean_before = clean_before_noise_fn
        self.weight_noise_amount = weight_noise_amount
        self.compute_accuracy = compute_accuracy
        self.exclude_params_for_norm = exclude_params_for_norm

        self.valid_costs = ['cost',
                            'accuracy1']
        if self.use_hints:
            self.valid_costs += ['accuracy1']

        if exclude_params_for_norm is None:
            self.exclude_params_for_norm = []
        else:
            self.exclude_params_for_norm = exclude_params_for_norm

        # Assume a single cost
        # We need to merge these lists
        grad_norm = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(self.param_grads, self.params) if p not in self.exclude_params_for_norm))

        state_below = self.cost_layer.state_below

        if hasattr(self.cost_layer, 'mask') and self.cost_layer.mask:
            div = TT.sum(self.cost_layer.mask)
        else:
            div = TT.cast(state_below.shape[0], 'float32')

        if hasattr(self.cost_layer, 'cost_scale') and self.cost_layer.cost_scale:
            div = div * self.cost_layer.cost_scale

        new_properties = [
                ('grad_norm', grad_norm),
                ('traincost', self.train_cost/div)]

        self.properties += new_properties
        if len(self.noise_params) >0 and weight_noise_amount:
            if self.need_inputs_for_generating_noise:
                inps = self.inputs
            else:
                inps = []
            new_vals = []

            for p, shp_fn in zip(self.noise_params,
                                 self.noise_params_shape_fn):

                new_val = self.trng.normal(shp_fn(self.inputs),
                                        avg=0,
                                        std=weight_noise_amount,
                                        dtype=p.dtype)
                axis = numpy.where([pb!=nb for pb, nb
                                    in zip(p.broadcastable, new_val.broadcastable)])[0]
                if len(axis)>0:
                    new_val = TT.unbroadcast(new_val, *list(axis))
                new_vals.append(new_val)

            self.add_noise = theano.function(inps,[],
                                             name='add_noise',
                                             updates =zip(self.noise_params, new_vals),
                                             on_unused_input='ignore')

            for p, shp_fn in zip(self.noise_params,
                                 self.noise_params_shape_fn):

                new_val = TT.zeros(shp_fn(self.inputs), p.dtype)
                axis = numpy.where([pb!=nb for pb, nb
                                    in zip(p.broadcastable, new_val.broadcastable)])[0]
                if axis:
                    new_val = TT.unbroadcast(new_val, *list(axis))
                new_vals.append(new_val)
            self.del_noise = theano.function(inps,[],
                                             name='del_noise',
                                             updates= zip(self.noise_params,
                                                          new_vals),
                                            on_unused_input='ignore')
        else:
            self.add_noise = None
            self.del_noise = None


    def validate(self, data_iterator):
        cost = 0
        accuracy = 0
        accuracy2 = 0
        n_steps = 0

        for vals in data_iterator:
            #n_steps += vals['x'].shape[0]
            #import ipdb; ipdb.set_trace()

            if isinstance(vals, dict):
                if self.del_noise and self.clean_noise_validation:
                    if self.need_inputs_for_generating_noise:
                        self.del_noise(**vals)
                    else:
                        self.del_noise()
                if vals['x'].shape[0] > 0:
                    _rvals = self.validate_step( **vals)
                #else:
                #    import ipdb; ipdb.set_trace()
            else:
                if self.del_noise and self.clean_noise_validation:
                    if self.need_inputs_for_generating_noise:
                        self.del_noise(*vals)
                    else:
                        self.del_noise()

                inps = list(vals)
                _rvals = self.validate_step(*inps)

            if vals['x'].shape[0] > 0:
                cost += _rvals[0]

            classifications = numpy.argmax(_rvals[1], axis=1)

            if self.use_hints:
                classifications2 = numpy.argmax(_rvals[2], axis=1)

            labels = vals['y']
            if self.use_hints:
                hints = vals['hints']

            n_steps += classifications.shape[0]

            #Those controls are needed for the sequential classifications.
            if classifications.ndim != 1:
                classifications = classifications.flatten()
                classifications2 = classifications2.flatten()

            if labels.ndim != 1:
                labels = labels.flatten()
                if self.use_hints:
                    hints = hints.flatten()

            accuracy += numpy.sum(labels==classifications)
            if self.use_hints:
                accuracy2 += numpy.sum(hints==classifications2)

        if n_steps > 0:
            cost = numpy.float(cost) / numpy.float(n_steps)
            accuracy = numpy.float(accuracy) / numpy.float(n_steps)

            if self.use_hints:
                accuracy2 = numpy.float(accuracy2) / numpy.float(n_steps)

        ret_vals = [('cost', cost), ('accuracy1', accuracy)]

        if self.use_hints:
            ret_vals += [('accuracy2', accuracy2)]

        return ret_vals

    def perturb(self, *args, **kwargs):
        if args:
            inps = args
            assert not kwargs
        if kwargs:
            inps = kwargs
            assert not args

        if self.noise_fn:
            if self.clean_before and self.del_noise:
                if self.need_inputs_for_generating_noise:
                    self.del_noise(*args, **kwargs)
                else:
                    self.del_noise()
            inps = self.noise_fn(*args, **kwargs)
        if self.add_noise:
            if self.need_inputs_for_generating_noise:
                self.add_noise(*args, **kwargs)
            else:
                self.add_noise()
        return inps


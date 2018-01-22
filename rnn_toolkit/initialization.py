# initializations
import logging
import numpy
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rng = numpy.random.RandomState(1234)


def normal_weight(shape, mean=0.0, stddev=0.01, seed=None, wd=1e-6, name=None):
    var = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, seed=None, dtype=tf.float32))
    tf.add_to_collection('L1_losses', wd * tf.reduce_sum(tf.abs(var)))
    tf.add_to_collection('L2_losses', wd * tf.nn.l2_loss(var))
    return var


def constant_weight(shape, value=0., wd=1e-6, name=None):
    var = tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=value, dtype=tf.float32))
    tf.add_to_collection('L1_losses', wd * tf.reduce_sum(tf.abs(var)))
    tf.add_to_collection('L2_losses', wd * tf.nn.l2_loss(var))
    return var


def ortho_weight(shape, wd=1e-6, name=None):
    var = tf.get_variable(name=name, shape=shape, initializer=tf.orthogonal_initializer(gain=1.0, dtype=tf.float32))
    tf.add_to_collection('L1_losses', wd * tf.reduce_sum(tf.abs(var)))
    tf.add_to_collection('L2_losses', wd * tf.nn.l2_loss(var))
    return var


def uniform_weight(shape, minval=-0.1, maxval=0.1,seed=None, wd=1e-6, name=None):
    var = tf.get_variable(name=name, shape=shape, initializer=tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=None, dtype=tf.float32))
    tf.add_to_collection('L1_losses', wd * tf.reduce_sum(tf.abs(var)))
    tf.add_to_collection('L2_losses', wd * tf.nn.l2_loss(var))
    return var


def _p(pp, name):
    return '%s_%s' % (pp, name)

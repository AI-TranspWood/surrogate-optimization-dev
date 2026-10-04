import os

import h5py
import keras
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import tfkan

tfd = tfp.distributions


def createArchitecture( inputSize, outputSize, layerSize = 1):

    inputShape = ( inputSize, 1)

    elasticModuli = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:2] ) - 10,
                                                 scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )

    poissonRatios = lambda input: tfd.Normal( loc   = 6 * tf.nn.sigmoid( input[..., 0:2] ) - 3,
                                              scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )

    elasticModulus = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:1] ) - 10,
                                                  scale = 10 * tf.nn.sigmoid( input[..., 1:2] ) + 1e-6 )

    elasticModuliLayer = tfp.layers.DistributionLambda( elasticModuli )
    poissonRatioLayer = tfp.layers.DistributionLambda( poissonRatios )
    elasticModulusLayer = tfp.layers.DistributionLambda( elasticModulus )

    inputs = keras.layers.Input( shape = inputShape )

    xBayesian = tfp.layers.DenseFlipout( layerSize )( 0 * inputs )
    x = tfkan.layers.DenseKAN( layerSize )( inputs )

    x = keras.layers.concatenate([ xBayesian / np.sqrt( layerSize ), x])
    x = keras.layers.Flatten()( x )
    x = tfkan.layers.DenseKAN( 5 * layerSize )( x )
    x = keras.layers.Flatten()( x )
    x = tfkan.layers.DenseKAN( 5 * layerSize )( x )
    x = keras.layers.Flatten()( x )
    x = tfkan.layers.DenseKAN( outputSize + outputSize )( x )

    E = elasticModuliLayer( x[ ..., 0:4] )
    nu = poissonRatioLayer( x[ ..., 4:8] )
    mu = elasticModulusLayer( x[ ..., 8:10] )

    model = keras.Model( inputs = inputs, outputs = [ E, nu, mu])
    return setWeights( model )


def setWeights( model ):

    path = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "weights/weights.h5" )
    with h5py.File( path, 'r') as file:
        weights = [file['weight' + str(ii)][:] for ii in range(len(file))]

    model.set_weights( weights )
    return model

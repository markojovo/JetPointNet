
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K


#============================================================================#
##============================ FUNCTIONS ===================================##
#============================================================================#
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def repeat_for_points(tensors):
    ''' y needs to be input shape tensor '''
    x, y = tensors
    reps = y.shape[-2]
    new_tens = tf.repeat(x, reps, axis=-2)
    return new_tens

def mat_mult(tensors):
    x, y = tensors
    return tf.linalg.matmul(x, y)

def cast_to_zero(tensors):
    ''' casts all values that should be zero to zero in the modified tensor '''
    mod_input, input_tens = tensors
    full_mask = tf.logical_not(tf.math.equal(input_tens, 0.))
    reduced_mask = tf.experimental.numpy.any(full_mask, axis=-1)
    reduced_mask = tf.cast(reduced_mask, dtype=tf.float32)
    reduced_mask = tf.expand_dims(reduced_mask, axis=-1)
    return_tens = tf.math.multiply(mod_input, reduced_mask)
    return return_tens

def tdist_block(x, mask, size: int, number: str):
    dense = layers.Dense(size)
    x = layers.TimeDistributed(dense, name='t_dist_'+number)(x, mask=mask)
    x = layers.Activation('relu', name='activation_'+number)(x)
    return x

def tdist_batchNorm(x, mask, size: int, number: str):
    dense = layers.Dense(size)
    x = layers.BatchNormalization(momentum=0.0, name='batchNorm_'+number)(dense)
    x = layers.TimeDistributed(x, name='t_dist_'+number)(x, mask=mask)
    x = layers.Activation('relu', name='activation_'+number)(x)
    return x


#============================================================================#
##=============================== MODELS ===================================##
#============================================================================#
def PointNet_delta_no_tnets(shape=(None,4), name=None):
    inputs = keras.Input(shape=shape, name="input")

    mask_tens = layers.Masking(mask_value=0.0, input_shape=shape)(inputs)
    keras_mask = mask_tens._keras_mask
    
    
    #=============== UPSCALE TO NEW FEATURE SPACE ===========================#
    block_3 = tdist_block(inputs, mask=keras_mask, size=16, number='3')
    block_4 = tdist_block(block_3, mask=keras_mask, size=16, number='4')
    #========================================================================#
    
    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(block_4, mask=keras_mask, size=64, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=128, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=256, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')(
    [block_10, inputs]
    )
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')([max_pool_block, block_4])
    
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=272, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=272, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=128, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=64, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)

def PointNet_delta(shape=(None,4), name=None):
    inputs = keras.Input(shape=shape, name="input")

    mask_tens = layers.Masking(mask_value=0.0, input_shape=shape)(inputs)
    keras_mask = mask_tens._keras_mask

    #============= T-NET ====================================================#
    block_0 = tdist_block(inputs, mask=keras_mask, size=32, number='0')
    block_1 = tdist_block(block_0, mask=keras_mask, size=64, number='1')
    block_2 = tdist_block(block_1, mask=keras_mask, size=64, number='2')
    
    # mask outputs to zero
    block_2_masked = layers.Lambda(cast_to_zero, name='block_2_masked')([block_2, inputs])
    
    max_pool = layers.MaxPool1D(pool_size=shape[0], name='tnet_0_MaxPool')(block_2_masked)
    mlp_tnet_0 = layers.Dense(64, activation='relu', name='tnet_0_dense_0')(max_pool)
    mlp_tnet_1 = layers.Dense(32, activation='relu', name='tnet_0_dense_1')(mlp_tnet_0)
    
    vector_dense = layers.Dense(
        shape[1]*shape[1],
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(shape[1]).flatten()),
        name='pre_matrix_0'
    )(mlp_tnet_1)
    
    mat_layer = layers.Reshape((shape[1], shape[1]), name='matrix_0')(vector_dense)
    
    mod_inputs_0 = layers.Lambda(mat_mult, name='matrix_multiply_0')([inputs, mat_layer])
    #========================================================================#
    
    
    #=============== UPSCALE TO NEW FEATURE SPACE ===========================#
    block_3 = tdist_block(mod_inputs_0, mask=keras_mask, size=16, number='3')
    block_4 = tdist_block(block_3, mask=keras_mask, size=16, number='4')
    #========================================================================#

    
    #============= T-NET ====================================================#
    block_5 = tdist_block(block_4, mask=keras_mask, size=64, number='5')
    block_6 = tdist_block(block_5, mask=keras_mask, size=128, number='6')
    block_7 = tdist_block(block_6, mask=keras_mask, size=256, number='7')
    
    # mask outputs to zero
    block_7_masked = layers.Lambda(cast_to_zero, name='block_7_masked')([block_7, inputs])
    
    max_pool_1 = layers.MaxPool1D(pool_size=shape[0], name='tnet_1_MaxPool')(block_7_masked)
    mlp_tnet_2 = layers.Dense(256, activation='relu', name='tnet_1_dense_0')(max_pool_1)
    mlp_tnet_3 = layers.Dense(256, activation='relu', name='tnet_1_dense_1')(mlp_tnet_2)
    
    vector_dense_1 = layers.Dense(
        256,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(16).flatten()),
        name='pre_matrix_1'
    )(mlp_tnet_3)
    
    mat_layer_1 = layers.Reshape((16, 16), name='matrix_1')(vector_dense_1)
    
    mod_features_1 = layers.Lambda(mat_mult, name='matrix_multiply_1')([block_4, mat_layer_1])
    #========================================================================#
    
    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(mod_features_1, mask=keras_mask, size=64, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=128, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=256, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')(
    [block_10, inputs]
    )
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')([max_pool_block, mod_features_1])
    
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=272, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=272, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=128, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=64, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)


#=============================================================================
#=============================================================================


def PointNet_omicron(shape=(100,4), name=None):
    inputs = keras.Input(shape=shape, name="input")

    mask_tens = layers.Masking(mask_value=0.0, input_shape=shape)(inputs)
    keras_mask = mask_tens._keras_mask

    #============= T-NET ====================================================#
    block_0 = tdist_block(inputs, mask=keras_mask, size=32, number='0')
    block_1 = tdist_block(block_0, mask=keras_mask, size=64, number='1')
    block_2 = tdist_block(block_1, mask=keras_mask, size=64, number='2')
    
    # mask outputs to zero
    block_2_masked = layers.Lambda(cast_to_zero, name='block_2_masked')([block_2, inputs])
    
    max_pool = layers.MaxPool1D(pool_size=shape[-2], name='tnet_0_MaxPool')(block_2_masked)
    mlp_tnet_0 = layers.Dense(64, activation='relu', name='tnet_0_dense_0')(max_pool)
    mlp_tnet_1 = layers.Dense(32, activation='relu', name='tnet_0_dense_1')(mlp_tnet_0)
    
    vector_dense = layers.Dense(
        shape[-1]*shape[-1],
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(shape[1]).flatten()),
        name='pre_matrix_0'
    )(mlp_tnet_1)
    
    mat_layer = layers.Reshape((shape[-1], shape[-1]), name='matrix_0')(vector_dense)
    
    mod_inputs_0 = layers.Lambda(mat_mult, name='matrix_multiply_0')([inputs, mat_layer])
    #========================================================================#
    
    
    #=============== UPSCALE TO NEW FEATURE SPACE ===========================#
    block_3 = tdist_block(mod_inputs_0, mask=keras_mask, size=32, number='3')
    block_4 = tdist_block(block_3, mask=keras_mask, size=32, number='4')
    #========================================================================#

    
    #============= T-NET ====================================================#
    block_5 = tdist_block(block_4, mask=keras_mask, size=256, number='5')
    block_6 = tdist_block(block_5, mask=keras_mask, size=256, number='6')
    block_7 = tdist_block(block_6, mask=keras_mask, size=512, number='7')
    
    # mask outputs to zero
    block_7_masked = layers.Lambda(cast_to_zero, name='block_7_masked')([block_7, inputs])
    
    max_pool_1 = layers.MaxPool1D(pool_size=shape[-2], name='tnet_1_MaxPool')(block_7_masked)
    mlp_tnet_2 = layers.Dense(512, activation='relu', name='tnet_1_dense_0')(max_pool_1)
    mlp_tnet_3 = layers.Dense(512, activation='relu', name='tnet_1_dense_1')(mlp_tnet_2)
    
    vector_dense_1 = layers.Dense(
        1024,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(32).flatten()),
        name='pre_matrix_1'
    )(mlp_tnet_3)
    
    mat_layer_1 = layers.Reshape((32, 32), name='matrix_1')(vector_dense_1)
    
    mod_features_1 = layers.Lambda(mat_mult, name='matrix_multiply_1')([block_4, mat_layer_1])
    #========================================================================#
    
    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(mod_features_1, mask=keras_mask, size=128, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=256, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=512, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')(
    [block_10, inputs]
    )
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')([max_pool_block, mod_features_1])
    
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=544, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=500, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=300, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=100, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)


#==============================================================================
#==============================================================================


def PointNet_gamma(shape=(100,4), name=None):
    inputs = keras.Input(shape=shape, name="input")

    mask_tens = layers.Masking(mask_value=0.0, input_shape=shape)(inputs)
    keras_mask = mask_tens._keras_mask

    #============= T-NET ====================================================#
    block_0 = tdist_block(inputs, mask=keras_mask, size=32, number='0')
    block_1 = tdist_block(block_0, mask=keras_mask, size=32, number='1')
    block_2 = tdist_block(block_1, mask=keras_mask, size=32, number='2')
    
    # mask outputs to zero
    block_2_masked = layers.Lambda(cast_to_zero, name='block_2_masked')([block_2, inputs])
    
    max_pool = layers.MaxPool1D(pool_size=shape[-2], name='tnet_0_MaxPool')(block_2_masked)
    mlp_tnet_0 = layers.Dense(32, activation='relu', name='tnet_0_dense_0')(max_pool)
    mlp_tnet_1 = layers.Dense(16, activation='relu', name='tnet_0_dense_1')(mlp_tnet_0)
    
    vector_dense = layers.Dense(
        shape[-1]*shape[-1],
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(shape[-1]).flatten()),
        name='pre_matrix_0'
    )(mlp_tnet_1)
    
    mat_layer = layers.Reshape((shape[-1], shape[-1]), name='matrix_0')(vector_dense)
    
    mod_inputs_0 = layers.Lambda(mat_mult, name='matrix_multiply_0')([inputs, mat_layer])
    #========================================================================#
    
    
    #=============== UPSCALE TO NEW FEATURE SPACE ===========================#
    block_3 = tdist_block(mod_inputs_0, mask=keras_mask, size=32, number='3')
    block_4 = tdist_block(block_3, mask=keras_mask, size=32, number='4')
    #========================================================================#

    
    #============= T-NET ====================================================#
    block_5 = tdist_block(block_4, mask=keras_mask, size=50, number='5')
    block_6 = tdist_block(block_5, mask=keras_mask, size=100, number='6')
    block_7 = tdist_block(block_6, mask=keras_mask, size=100, number='7')
    
    # mask outputs to zero
    block_7_masked = layers.Lambda(cast_to_zero, name='block_7_masked')([block_7, inputs])
    
    max_pool_1 = layers.MaxPool1D(pool_size=shape[-2], name='tnet_1_MaxPool')(block_7_masked)
    mlp_tnet_2 = layers.Dense(200, activation='relu', name='tnet_1_dense_0')(max_pool_1)
    mlp_tnet_3 = layers.Dense(200, activation='relu', name='tnet_1_dense_1')(mlp_tnet_2)
    
    vector_dense_1 = layers.Dense(
        1024,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(32).flatten()),
        name='pre_matrix_1'
    )(mlp_tnet_3)
    
    mat_layer_1 = layers.Reshape((32, 32), name='matrix_1')(vector_dense_1)
    
    mod_features_1 = layers.Lambda(mat_mult, name='matrix_multiply_1')([block_4, mat_layer_1])
    #========================================================================#
    
    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(mod_features_1, mask=keras_mask, size=200, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=200, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=300, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')(
    [block_10, inputs]
    )
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')([max_pool_block, mod_features_1])
    
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=332, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=300, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=200, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=100, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)

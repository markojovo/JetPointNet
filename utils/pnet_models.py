
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend as K
from keras.layers import Layer



# Part segmentation models

# https://keras.io/examples/vision/pointnet_segmentation/
def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def time_dist_block(x: tf.Tensor, mask, size: int, name: str) -> tf.Tensor:
    dense = layers.Dense(size)
    x = layers.TimeDistributed(dense, name=f"{name}_conv")(x)#, mask=mask)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

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
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config
    
def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)

def transformation_net_propagate_mask(inputs: tf.Tensor, mask, num_features: int, name: str, input_points) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = time_dist_block(inputs, mask, 64, name=f"{name}_1")
    x = time_dist_block(x, mask, 128, name=f"{name}_2")
    x = time_dist_block(x, mask, 1024, name=f"{name}_3")
    # cast masked layers to 0
    x_masked = layers.Lambda(cast_to_zero, name=f"{name}_3_masked")([x, input_points])

    x = layers.GlobalMaxPooling1D()(x_masked)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block_propagate_mask(inputs: tf.Tensor, mask, num_features: int, name: str, input_points) -> tf.Tensor:
    transformed_features = transformation_net_propagate_mask(inputs, mask, num_features, name=name, input_points=input_points)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features]) # implicit masking when dot prod with inputs is taken (since 0 points stay 0) - only if masked out values are 0!!

def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])

def part_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 4))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=4, name="input_transformation_block"
    )

    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="sigmoid", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)


def part_segmentation_model_propagate_mask(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 4))

    # add mask layer
    mask_tens = layers.Masking(mask_value=0.0, input_shape=(num_points, 4))(input_points)
    mask = mask_tens._keras_mask

    # PointNet Classification Network.
    transformed_inputs = transformation_block_propagate_mask(
        input_points, mask, num_features=4, name="input_transformation_block", input_points=input_points
    )

    features_64 = time_dist_block(transformed_inputs, mask,  64, name="features_64")
    features_128_1 = time_dist_block(features_64, mask,  128, name="features_128_1")
    features_128_2 = time_dist_block(features_128_1, mask,  128, name="features_128_2")
    transformed_features = transformation_block_propagate_mask(
        features_128_2, mask, num_features=128, name="transformed_features", input_points=input_points
    )

    features_512 = time_dist_block(transformed_features, mask,  512, name="features_512")
    features_2048 = time_dist_block(features_512, mask,  2048, name="pre_maxpool_block")

    # cast masked inputs to 0
    features_2048_masked = layers.Lambda(cast_to_zero, name='pre_maxpool_block_masked')([features_2048, input_points])
    
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048_masked
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = time_dist_block(
        segmentation_input, mask, 128, name="segmentation_features"
    )
    

    last_dense = layers.Dense(num_classes)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(segmentation_features)
    outputs = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(input_points, outputs)

#============================================================================#
##======================== CLASSIFICATION MODELS ===========================##
#============================================================================#

# from ChatGPT3

class MaskedBatchNormalization(Layer):
    def __init__(self, momentum=0.0, epsilon=1e-3, mask=None, **kwargs):
        super(MaskedBatchNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        if isinstance(dim, tf.TensorShape):
            dim = dim.value

        self.gamma = self.add_weight(name='gamma', 
                                     shape=(dim,), 
                                     initializer='ones', 
                                     trainable=True)
        self.beta = self.add_weight(name='beta', 
                                    shape=(dim,), 
                                    initializer='zeros', 
                                    trainable=True)
        super(MaskedBatchNormalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = tf.not_equal(tf.reduce_sum(inputs, axis=-1, keepdims=True), 0)
        
        mask = tf.cast(mask, dtype=tf.float32)
        masked_inputs = inputs * mask

        mean, variance = tf.nn.moments(masked_inputs, axes=-1, keepdims=True)

        # Apply normalization
        normalized = (masked_inputs - mean) / (tf.sqrt(variance + self.epsilon))
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super(MaskedBatchNormalization, self).get_config()
        config.update({
            'momentum': self.momentum,
            'epsilon': self.epsilon
        })
        return config
    
########

def t_dist_block(x: tf.Tensor, size: int, name: str) -> tf.Tensor:
    dense = layers.Dense(size)
    #ln = layers.LayerNormalization(name='layerNorm_' + name)
    bn = layers.BatchNormalization(momentum=0.0, name='batchNorm_' + name)
    x = layers.TimeDistributed(dense, name=f"{name}_tdist")(x)
    x = layers.TimeDistributed(bn, name=f"{name}_bn_tdist")(x)
    #x = layers.BatchNormalization(momentum=0.0, name='batchNorm_' + name)(x) # TODO: remove just for a test
    #x = MaskedBatchNormalization(name='batchNorm_' + name)(x, mask) # TODO: remove just for a test
    #x = layers.LayerNormalization(name='layerNorm_' + name)(x) # TODO: remove just for a test
    return layers.Activation("relu", name=f"{name}_relu")(x)

def t_dist_block_masked_bn(x: tf.Tensor, size: int, name: str, mask: tf.Tensor) -> tf.Tensor:
    dense = layers.Dense(size)
    #bn = CustomBatchNormalizationMomentum(name='batchNorm_' + name, mask=mask) # for time dist NOTE: update mask to be for each N
    x = layers.TimeDistributed(dense, name=f"{name}_tdist")(x)
    #x = layers.TimeDistributed(bn, name=f"{name}_bn_tdist")(x) # for time dist
    x = MaskedBatchNormalization(name='batchNorm_' + name, mask=mask)(x)
    #x = layers.BatchNormalization(momentum=0.0, name='batchNorm_' + name)(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def t_dist_block_mask(x: tf.Tensor, size: int, name: str, mask):
    dense = layers.Dense(size)
    x = layers.TimeDistributed(dense, name=f"{name}_tdist")(x, mask)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def tnet(inputs: tf.Tensor, num_features: int, name: str, input_points, mask: tf.Tensor) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = t_dist_block_masked_bn(inputs, 64, f"{name}_1", mask)
    x = t_dist_block_masked_bn(x, 128, f"{name}_2", mask)
    x = t_dist_block_masked_bn(x, 1024, f"{name}_3", mask)
    # cast masked layers to 0
    x_masked = layers.Lambda(cast_to_zero, name=f"{name}_3_masked")([x, input_points])

    x = layers.GlobalMaxPooling1D()(x_masked)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    transformed_features = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        #activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)
    transformed_features = layers.Reshape((num_features, num_features))(transformed_features)

    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


def pnet_part_seg_no_tnets(num_points: int, num_feat: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, num_feat))
    full_mask = tf.logical_not(tf.math.equal(input_points, 0))
    mask = tf.reduce_any(full_mask, axis=-1)

    # Assuming t_dist_block_masked_bn is compatible with eager execution
    features_64 = t_dist_block_masked_bn(input_points, 64, "features_64", mask)
    features_128_1 = t_dist_block_masked_bn(features_64, 128, "features_128_1", mask)
    features_128_2 = t_dist_block_masked_bn(features_128_1, 128, "features_128_2", mask)

    features_512 = t_dist_block_masked_bn(features_128_2, 512, "features_512", mask)
    features_2048 = t_dist_block_masked_bn(features_512, 2048, "pre_maxpool_block", mask)

    # Assuming cast_to_zero is compatible with eager execution
    features_2048_masked = layers.Lambda(cast_to_zero, name='pre_maxpool_block_masked')([features_2048, input_points])
    
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(features_2048_masked)
    global_features = tf.tile(global_features, [1, num_points, 1])

    segmentation_input = layers.Concatenate(name="segmentation_input")([
        features_64,
        features_128_1,
        features_128_2,
        features_512,
        global_features,
    ])

    # Assuming t_dist_block_masked_bn is compatible with eager execution for segmentation_features
    segmentation_features = t_dist_block_masked_bn(segmentation_input, 128, "segmentation_features", mask)

    # Final layers
    last_dense = layers.Dense(num_classes)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(segmentation_features)
    outputs = layers.Activation('softmax', name="last_act")(last_time)

    return keras.Model(inputs=input_points, outputs=outputs)

def pnet_part_seg(num_points: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 6))
    full_mask = tf.logical_not(tf.math.equal(input_points, 0))
    mask = tf.experimental.numpy.any(full_mask, axis=-1)

    transformed_inputs = tnet(input_points, num_features=6, name="input_transformation_block", input_points=input_points, mask=mask)

    features_64 = t_dist_block_masked_bn(transformed_inputs,  64, "features_64", mask)
    features_128_1 = t_dist_block_masked_bn(features_64, 128, "features_128_1", mask)
    features_128_2 = t_dist_block_masked_bn(features_128_1, 128, "features_128_2", mask)
    transformed_features = tnet(features_128_2, num_features=128, name="transformed_features", input_points=input_points, mask=mask)

    features_512 = t_dist_block_masked_bn(transformed_features,  512, "features_512", mask)
    features_2048 = t_dist_block_masked_bn(features_512, 2048, "pre_maxpool_block", mask)

    # cast masked inputs to 0
    features_2048_masked = layers.Lambda(cast_to_zero, name='pre_maxpool_block_masked')([features_2048, input_points])
    
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048_masked
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            features_512,
            global_features,
        ]
    )

    segmentation_features = t_dist_block_masked_bn(segmentation_input, 128, "segmentation_features", mask)
    
    # add extra t-dist and dropout layers
    """
    segmentation_features_256_1 = t_dist_block(segmentation_input, 256, 'segmentation_features_256_1')
    dropout_0 = layers.Dropout(rate=.2)(segmentation_features_256_1)
    segmentation_features_256_2 = t_dist_block(dropout_0, 256, 'segmentation_features_256_2')
    dropout_1 = layers.Dropout(rate=.2)(segmentation_features_256_2)
    segmentation_features_128 = t_dist_block(dropout_1,  128, name="segmentation_features_128")
    dropout_2 = layers.Dropout(rate=.2)(segmentation_features_128)
    """
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(segmentation_features)
    outputs = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(input_points, outputs)


# Russell's previous models
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
    mod_input, input_tens = tensors
    full_mask = tf.logical_not(tf.math.equal(input_tens, 0))  # Find where all X values are equal to 0
    reduced_mask = tf.reduce_any(full_mask, axis=-1)  # Mask if all input features are 0
    reduced_mask = tf.cast(reduced_mask, dtype=tf.float32)
    reduced_mask = tf.expand_dims(reduced_mask, axis=-1)
    return_tens = mod_input * reduced_mask
    return return_tens

    

def tdist_block(x, mask, size: int, number: str):
    dense = layers.Dense(size)
    x = layers.TimeDistributed(dense, name='t_dist_'+number)(x)#, mask=mask)
    #x = layers.BatchNormalization(momentum=0.0, name='batchNorm_'+number)(x) # added
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

    mask_tens = layers.Masking(mask_value=-1, input_shape=shape)(inputs)
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
        #activity_regularizer=OrthogonalRegularizer(shape[-1]),
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
    mlp_tnet_3 = layers.Dense(200, activation='relu', name='tnet_1_dense_1')(mlp_tnet_2) # TODO: does this have to be 32x32??
    
    vector_dense_1 = layers.Dense(
        1024,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(32).flatten()),
        #activity_regularizer=OrthogonalRegularizer(32),
        name='pre_matrix_1'
    )(mlp_tnet_3)
    
    mat_layer_1 = layers.Reshape((32, 32), name='matrix_1')(vector_dense_1)
    
    mod_features_1 = layers.Lambda(mat_mult, name='matrix_multiply_1')([block_4, mat_layer_1])
    #========================================================================#
    
    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(mod_features_1, mask=keras_mask, size=200, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=200, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=300, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')([block_10, inputs])
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')(#[max_pool_block, mod_features_1])
        [
            mod_inputs_0,
            block_3,
            block_4,
            mod_features_1,
            block_8,
            block_9,
            max_pool_block
        ])
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=332, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=300, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=200, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=100, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3)#, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)


def PointNet_gamma_no_tnet(shape=(100,4), name=None):
    inputs = keras.Input(shape=shape, name="input")

    mask_tens = layers.Masking(mask_value=-1, input_shape=shape)(inputs)
    keras_mask = mask_tens._keras_mask
    
    
    #=============== UPSCALE TO NEW FEATURE SPACE ===========================#
    block_3 = tdist_block(inputs, mask=keras_mask, size=32, number='3')
    block_4 = tdist_block(block_3, mask=keras_mask, size=32, number='4')
    #========================================================================#

    
    #================ MLP + MAXPOOL BLOCK ===================================#
    block_8 = tdist_block(block_4, mask=keras_mask, size=200, number='8')
    block_9 = tdist_block(block_8, mask=keras_mask, size=200, number='9')
    block_10 = tdist_block(block_9, mask=keras_mask, size=300, number='10')
    
    block_10_masked = layers.Lambda(cast_to_zero, name='block_10_masked')([block_10, inputs])
    
    max_pool_2 = layers.MaxPool1D(pool_size=shape[-2], name='global_maxpool')(block_10_masked)
    #========================================================================#

    max_pool_block = layers.Lambda(repeat_for_points, name='mp_block')([max_pool_2, inputs])
    
    block_11 = layers.Concatenate(axis=-1, name='concatenation')(#[max_pool_block, mod_features_1])
        [
            block_3,
            block_4,
            block_8,
            block_9,
            max_pool_block
        ])
    
    block_12 = tdist_block(block_11, mask=keras_mask, size=332, number='12')
    dropout_0 = layers.Dropout(rate=.2)(block_12)
    block_13 = tdist_block(dropout_0, mask=keras_mask, size=300, number='13')
    dropout_1 = layers.Dropout(rate=.2)(block_13)
    block_14 = tdist_block(dropout_1, mask=keras_mask, size=200, number='14')
    dropout_2 = layers.Dropout(rate=.2)(block_14)
    block_15 = tdist_block(dropout_2, mask=keras_mask, size=100, number='15')
    dropout_3 = layers.Dropout(rate=.2)(block_15)
    
    last_dense = layers.Dense(1)
    last_time = layers.TimeDistributed(last_dense, name='last_tdist')(dropout_3)#, mask=keras_mask)
    last_act = layers.Activation('sigmoid', name="last_act")(last_time)

    return keras.Model(inputs=inputs, outputs=last_act, name=name)

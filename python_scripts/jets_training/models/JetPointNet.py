
'''
Adapted From
https://github.com/lattice-ai/pointnet/tree/master

Original Architecture From Pointnet Paper:
https://arxiv.org/pdf/1612.00593.pdf
'''


import tensorflow as tf
import numpy as np
import keras 

# =======================================================================================================================
# ============ Weird Stuff ==============================================================================================


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("JetPointNet_{epoch}.hd5".format(epoch))

class CustomMaskingLayer(tf.keras.layers.Layer):
    # For masking out the inputs properly, based on points for which the last value in the point's array (it's "type") is "-1"
    def __init__(self, **kwargs):
        super(CustomMaskingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        mask = tf.not_equal(inputs[:, :, -1], -1) # Masking
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        return inputs * mask

    def compute_output_shape(self, input_shape):
        return input_shape

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    # Used in Tnet in PointNet for transforming everything to same space
    def __init__(self, num_features=6, l2=0.001):
        self.num_features = num_features
        self.l2 = l2
        self.I = tf.eye(num_features)

    def __call__(self, inputs):
        A = tf.reshape(inputs, (-1, self.num_features, self.num_features))
        AAT = tf.tensordot(A, A, axes=(2, 2))
        AAT = tf.reshape(AAT, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2 * tf.square(AAT - self.I))

    def get_config(self):
        # Return a dictionary containing the parameters of the regularizer to allow for model serialization
        return {'num_features': self.num_features, 'l2': self.l2}
    

def TSSR_Activation(x):
    # An activation function that's linear when |x| < 1 and (an adjusted) sqrt when |x| > 1 (is kinda like a "soft tanh" function that isn't bounded in output)
    # Adapted from https://arxiv.org/pdf/2308.04832.pdf
    # Careful, can produce negative values (can mess around with this, I think it might be good to stabilize training) - ie the problem of hard constraints vs soft constraints in an optimization problem
    high_value_slope_coefficient = 1 #Value to multiply by the sqrt|x| part, lower means flatter, higher means steeper
    condition = tf.abs(x) < 1
    return tf.where(condition, x, tf.sign(x) * (high_value_slope_coefficient*tf.sqrt(tf.abs(x)) - high_value_slope_coefficient + 1))

# =======================================================================================================================
# =======================================================================================================================



# =======================================================================================================================
# ============ Main Model Blocks ========================================================================================

def conv_mlp(input_tensor, filters, dropout_rate = None):
    # Apply shared MLPs which are equivalent to 1D convolutions with kernel size 1
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def dense_block(input_tensor, units, dropout_rate=None, regularizer=None):
    x = tf.keras.layers.Dense(units, kernel_regularizer=regularizer)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def TNet(input_tensor, size, add_regularization=False):
    # size is either 6 for the first TNet or 64 for the second
    x = conv_mlp(input_tensor, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    if add_regularization:
        reg = OrthogonalRegularizer(size)
    else:
        reg = None
    x = dense_block(x, size * size, regularizer=reg)
    x = tf.reshape(x, (-1, size, size))
    return x        


def PointNetSegmentation(num_points, num_classes):
    num_features = 6  # Number of input features

    network_size_factor = 5 # Mess around with this along with the different layer sizes

    input_points = tf.keras.Input(shape=(num_points, num_features))

    # T-Net for input transformation
    input_tnet = TNet(input_points, num_features)
    x = tf.keras.layers.Dot(axes=(2, 1))([input_points, input_tnet])
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 64)
    point_features = x

    # T-Net for feature transformation
    feature_tnet = TNet(x, 64, add_regularization=True)
    x = tf.keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 64 * network_size_factor)
    x = conv_mlp(x, 128 * network_size_factor)
    x = conv_mlp(x, 1024 * network_size_factor)

    # Get global features and expand
    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_points, 1]))(global_feature_expanded)

    # Concatenate point features with global features
    c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])
    c = conv_mlp(c, 512 * network_size_factor)
    c = conv_mlp(c, 256 * network_size_factor)
    c = conv_mlp(c, 128 * network_size_factor, dropout_rate=0.3)

    # Extract energy from input and multiply by the segmentation output
    energy = tf.expand_dims(input_points[:, :, 4], -1)  # Assuming energy is at index 4
    segmentation_output_pre_sigmoid = tf.keras.layers.Conv1D(num_classes, kernel_size=1)(c)  # No activation yet ("sigmoid" here is a misnomer, we're using TSSR. Feel free to update)
    segmentation_output_pre_sigmoid = tf.keras.layers.Activation(TSSR_Activation)(segmentation_output_pre_sigmoid) # Apply activation + adjust float back to 32 bit for training (a smarter way to do this probably exists)
    segmentation_output = tf.multiply(segmentation_output_pre_sigmoid, energy)  # Multiply by energy

    model = tf.keras.Model(inputs=input_points, outputs=segmentation_output)

    return model

# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Losses ===================================================================================================

def masked_bce_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False) 
    masked_loss = base_loss * mask

    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask) / batch_size # This might be kinda dumb, might be able to not use reduce_sum and avoid having to manually get batch_size

def masked_mse_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    scale_factor = 1 # For scaling the output and label pre-squaring the difference

    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_squared_error(scale_factor*y_true, scale_factor*y_pred)
    masked_loss = base_loss * mask

    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask) / batch_size

def masked_mae_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    masked_loss = base_loss * mask

    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask) / batch_size

# =======================================================================================================================
# =======================================================================================================================

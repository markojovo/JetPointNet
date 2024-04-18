
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
    

def rectified_TSSR_Activation(x):
    a = 0.01 # leaky ReLu style slope when negative
    b = 0.1 # sqrt(x) damping coefficient when x > 1
    
    # Adapted from https://arxiv.org/pdf/2308.04832.pdf
    # An activation function that's linear when 0 < x < 1 and (an adjusted) sqrt when x > 1,
    # behaves like leaky ReLU when x < 0.

    # 'a' is the slope coefficient for x < 0.
    # 'b' is the value to multiply by the sqrt(x) part.

    negative_condition = x < 0
    small_positive_condition = tf.logical_and(tf.greater_equal(x, 0), tf.less(x, 1))
    #large_positive_condition = x >= 1
    
    negative_part = a * x
    small_positive_part = x
    large_positive_part = tf.sign(x) * (b * tf.sqrt(tf.abs(x)) - b + 1)
    
    return tf.where(negative_condition, negative_part, 
                    tf.where(small_positive_condition, small_positive_part, large_positive_part))

def custom_sigmoid(x, a = 3.0):
    return 1 / (1 + tf.exp(-a * x))

def hard_sigmoid(x):
    return tf.keras.backend.cast(x > 0, dtype=tf.float32)

# =======================================================================================================================
# =======================================================================================================================



# =======================================================================================================================
# ============ Main Model Blocks ========================================================================================

def conv_mlp(input_tensor, filters, dropout_rate=None, apply_attention=False):
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if apply_attention:
        # Self-attention
        attention_output_self = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=filters)(x, x)
        attention_output_self = tf.keras.layers.LayerNormalization()(attention_output_self + x)
        
        # Cross-attention
        attention_output_cross = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=filters)(attention_output_self, x)
        attention_output_cross = tf.keras.layers.LayerNormalization()(attention_output_cross + attention_output_self)

        x = attention_output_cross

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
    num_features = 6  # Number of input features per point
    network_size_factor = 8 # Mess around with this along with the different layer sizes 

    '''
    Input shape per point is:
       [x (mm),
        y (mm),
        z (mm),
        minimum_of_distance_to_focused_track (mm),
        energy (MeV),
        type (-1 for masked, 0 for calorimeter cell, 1 for focused track and 2 for other track)]
    
    Note that in awk_to_npz.py, if add_tracks_as_labels == False then the labels for the tracks is "-1" (to be masked of the loss and not predicted on)

    '''

    input_points = tf.keras.Input(shape=(num_points, num_features))

    # Extract energy from input and multiply by the segmentation output
    #energy = tf.expand_dims(input_points[:, :, 4], -1)  # Assuming energy is at index 4
    energy = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, 4], -1), name='energy_output')(input_points)

    # T-Net for input transformation
    input_tnet = TNet(input_points, num_features)
    x = tf.keras.layers.Dot(axes=(2, 1))([input_points, input_tnet])
    x = conv_mlp(x, 72)
    x = conv_mlp(x, 72)
    point_features = x

    # T-Net for feature transformation
    feature_tnet = TNet(x, 72, add_regularization=True)
    x = tf.keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 128 * network_size_factor)
    x = conv_mlp(x, 256 * network_size_factor)
    x = conv_mlp(x, 1024 * network_size_factor)

    # Get global features and expand
    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_points, 1]))(global_feature_expanded)

    # Concatenate point features with global features
    c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])
    c = conv_mlp(c, 512)#, apply_attention=True)
    c = conv_mlp(c, 256)#, apply_attention=True)

    c = conv_mlp(c, 128, dropout_rate=0.3)

    segmentation_output = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation='sigmoid', name='segmentation_output')(c)

    model = tf.keras.Model(inputs=input_points, outputs=[segmentation_output, energy])

    return model

# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Losses ===================================================================================================

def masked_weighted_bce_loss_wrapper(y_true, y_pred):
    return masked_weighted_bce_loss(y_true[0], y_pred[0], y_pred[1])

def masked_weighted_bce_loss(y_true, y_pred, weights):
    # Create a mask that identifies valid (non-masked) entries
    valid_mask = tf.not_equal(y_true, -1.0)
    valid_mask = tf.cast(valid_mask, tf.float32)
    valid_mask = tf.squeeze(valid_mask, axis=-1)  # Flatten the mask to match y_true's and y_pred's dimensions

    # Calculate binary cross-entropy loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    weights_squared = tf.square(weights)  # Square the weights
    bce_loss = bce_loss * valid_mask * weights_squared  # Apply both the valid mask and the squared weights

    # Calculate the sum of valid squared weights or use a fixed minimum value, whichever is larger
    normalization_factor = tf.reduce_sum(valid_mask * weights_squared, axis=1)
    normalization_factor = tf.maximum(normalization_factor, 1000)  # Ensure the normalization factor is at least 1000

    # Normalize each sample's loss by the determined factor
    sample_normalized_bce_loss = tf.reduce_sum(bce_loss, axis=1) / normalization_factor

    # Take the mean across the batch for a single loss value
    return tf.reduce_mean(sample_normalized_bce_loss)

def masked_accuracy(y_true, y_pred):
    # Create a mask to ignore positions where y_true is -1.0
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)

    # Calculate the Mean Absolute Error (MAE) only where mask is True
    absolute_errors = tf.abs(y_pred - y_true)  # Calculate absolute differences
    masked_absolute_errors = absolute_errors * mask  # Apply mask
    masked_mae = tf.reduce_sum(masked_absolute_errors) / tf.reduce_sum(mask)

    # Calculate accuracy as 1 - masked MAE
    accuracy = 1 - masked_mae
    return accuracy

# =======================================================================================================================
# =======================================================================================================================



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
        mask = tf.not_equal(inputs[:, :, -1], 1) # Masking
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        return inputs * mask

    def compute_output_shape(self, input_shape):
        return input_shape

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    # Used in Tnet in PointNet for transforming everything to same space
    def __init__(self, num_features=9, l2=0.001):
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
    num_features = 9  # Number of input features per point

    '''
    Input shape per point is:
       [x (mm),
        y (mm),
        z (mm),
        minimum_of_distance_to_focused_track (mm),
        energy (GeV),
        type (focused track, cells, associated track, masked out)
    
    Note that in awk_to_npz.py, if add_tracks_as_labels == False then the labels for the tracks is "-1" (to be masked of the loss and not predicted on)

    '''

    input_points = tf.keras.Input(shape=(num_points, num_features))

    # Masking layer to ignore points with the last feature index as -1
    masks = tf.keras.layers.Lambda(lambda x: tf.not_equal(x[:, :, -1], 1), output_shape=(num_points,))(input_points)
    masks = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(masks)  # Cast boolean to float for multiplication
    masks = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(masks)  # Expand dimensions to apply mask


    # Apply mask
    input_points_masked = tf.keras.layers.Multiply()([input_points, masks])

    #energy = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, 4], -1), name='e')(input_points_masked)

    # T-Net for input transformation
    input_tnet = TNet(input_points_masked, num_features)  # Assuming TNet is properly defined elsewhere
    x = tf.keras.layers.Dot(axes=(2, 1))([input_points_masked, input_tnet])
    x = conv_mlp(x, 96)  # Assuming conv_mlp is properly defined elsewhere
    x = conv_mlp(x, 96 )
    point_features = x

    # T-Net for feature transformation
    feature_tnet = TNet(x, 96, add_regularization=True)
    x = tf.keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 256)
    x = conv_mlp(x, 1024)

    # Get global features and expand
    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_points, 1]))(global_feature_expanded)

    # Concatenate point features with global features
    c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])
    c = conv_mlp(c, 512, apply_attention=False)
    c = conv_mlp(c, 256, apply_attention=False)

    c = conv_mlp(c, 128, dropout_rate=0.3)

    segmentation_output = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation='linear', name='SEG')(c)

    model = tf.keras.Model(inputs=input_points, outputs=segmentation_output)

    return model
# =======================================================================================================================
# =======================================================================================================================


# =======================================================================================================================
# ============ Losses ===================================================================================================

def masked_weighted_bce_loss(y_true, y_pred, energies):

    #energies = tf.square(energies)

    # Ensure valid_mask and y_true are compatible for operations
    valid_mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)  # This should be [batch, points, 1]

    # Adjust y_true based on the threshold, maintain dimensions as [batch, points, 1]
    y_true_adjusted = tf.cast(tf.greater_equal(y_true, 0.5), tf.float32) * valid_mask

    # Calculate binary cross-entropy loss, ensuring to keep the dimensions consistent

    y_pred_masked = y_pred * valid_mask
    bce_loss = tf.keras.losses.binary_crossentropy(y_true_adjusted, y_pred_masked, from_logits=True)
    bce_loss = tf.expand_dims(bce_loss, axis=-1)  # Ensure BCE loss has the same [batch, points, 1] shape as others

    # Weighted binary cross-entropy loss, ensuring all dimensions match
    energies_times_mask = energies * valid_mask
    weighted_bce_loss = bce_loss * energies_times_mask

    # Normalize the weighted BCE loss
    total_energy_weight = tf.reduce_sum(energies * valid_mask, axis=1, keepdims=True)  # Keep dimensions with 'keepdims'
    total_num_points = tf.reduce_sum(valid_mask, axis=1, keepdims=True)  # Keep dimensions with 'keepdims'
    normalized_bce_loss = weighted_bce_loss / (total_num_points + 1) / (total_energy_weight + 1) 

    # Combine the mean losses from both labels
    return normalized_bce_loss 


'''
def masked_weighted_bce_loss(y_true, y_pred, energies):
    energies = tf.square(energies)

    # Ensure valid_mask and y_true are compatible for operations
    valid_mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)  # This should be [batch, points, 1]

    # Adjust y_true based on the threshold, maintain dimensions as [batch, points, 1]
    y_true_adjusted = tf.cast(tf.greater_equal(y_true, 0.5), tf.float32) * valid_mask

    # Calculate binary cross-entropy loss, ensuring to keep the dimensions consistent
    bce_loss = tf.keras.losses.binary_crossentropy(y_true_adjusted, y_pred, from_logits=True)
    bce_loss = tf.expand_dims(bce_loss, axis=-1)  # Ensure BCE loss has the same [batch, points, 1] shape as others

    # Weighted binary cross-entropy loss, ensuring all dimensions match
    weighted_bce_loss = bce_loss * energies * valid_mask

    # Normalize the weighted BCE loss
    total_energy_weight = tf.reduce_sum(energies * valid_mask, axis=1, keepdims=True)  # Keep dimensions with 'keepdims'
    normalized_bce_loss = weighted_bce_loss / (total_energy_weight + 1e-5)

    # Create masks for 'zero' and 'one' labels, apply to normalized loss, and compute means
    mask_zeros = tf.cast(tf.equal(y_true_adjusted, 0.0), tf.float32)
    mask_ones = tf.cast(tf.equal(y_true_adjusted, 1.0), tf.float32)
    mean_normalized_bce_loss_zeros = tf.reduce_sum(normalized_bce_loss * mask_zeros) / (tf.reduce_sum(mask_zeros) + 1 )
    mean_normalized_bce_loss_ones = tf.reduce_sum(normalized_bce_loss * mask_ones) / (tf.reduce_sum(mask_ones) + 1 )

    # Combine the mean losses from both labels
    return (mean_normalized_bce_loss_zeros + mean_normalized_bce_loss_ones) / 2

'''



def masked_regular_accuracy(y_true, y_pred, energies):
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)

    adjusted_y_true = tf.cast(tf.greater(y_true, 0.5), tf.float32)
    adjusted_y_pred = tf.cast(tf.greater(y_pred, 0.0), tf.float32)
    
    correct_predictions = tf.equal(adjusted_y_pred, adjusted_y_true)

    masked_correct_predictions = tf.cast(correct_predictions, tf.float32) * mask

    accuracy = tf.reduce_sum(masked_correct_predictions) / (tf.reduce_sum(mask) + 1e-5)
    return accuracy


def masked_weighted_accuracy(y_true, y_pred, energies):
    #energies = tf.square(energies)


    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)

    adjusted_y_true = tf.cast(tf.greater(y_true, 0.5), tf.float32)
    adjusted_y_pred = tf.cast(tf.greater(y_pred, 0.0), tf.float32)
    
    correct_predictions = tf.equal(adjusted_y_pred, adjusted_y_true)

    masked_correct_predictions = tf.cast(correct_predictions, tf.float32) * mask

    weighted_correct_predictions = masked_correct_predictions * energies
    sum_weights = tf.reduce_sum(energies * mask, axis=1)
    normalized_accuracy = tf.reduce_sum(weighted_correct_predictions, axis=1) / (sum_weights + 1e-5)  


    return normalized_accuracy





# =======================================================================================================================
# =======================================================================================================================


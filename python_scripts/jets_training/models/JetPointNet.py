
'''
Adapted From
https://github.com/lattice-ai/pointnet/tree/master

Original Architecture From
https://arxiv.org/pdf/1612.00593.pdf
'''


import tensorflow as tf
import numpy as np
import keras 
from tensorflow.keras.layers import Input, Dot, Conv1D, BatchNormalization, Activation, Dense, GlobalMaxPooling1D, Lambda, Concatenate
from tensorflow.keras.models import Model

class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("JetPointNet_{epoch}.hd5".format(epoch))

class CustomMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomMaskingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        mask = tf.not_equal(inputs[:, :, -1], -1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, -1)
        return inputs * mask

    def compute_output_shape(self, input_shape):
        return input_shape

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
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
    

def conv_mlp(input_tensor, filters, dropout_rate = None):
    # Apply shared MLPs which are equivalent to 1D convolutions with kernel size 1
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
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



def PointNetSegmentation(num_points, num_classes, training=False):
    # Adjust num_features to accommodate the concatenated targets
    num_features = 6
    combined_input = Input(shape=(num_points, num_features + 1), name='combined_input')  # +1 for the target

    # Separate features and targets from the combined input tensor
    input_points = Lambda(lambda x: x[:, :, :num_features])(combined_input)  # Features
    targets = Lambda(lambda x: x[:, :, -1])(combined_input)  # This assumes the targets have shape (batch_size, num_points)

    energy_weights = Lambda(lambda x: x[:, :, 4])(input_points)  # since it goes x, y, z, dist_to_track, E, type
    targets = Lambda(lambda x: tf.expand_dims(x, -1))(targets)

    # Proceed with the original architecture using separated features
    input_tnet = TNet(input_points, num_features)  # Adjust num_features
    x = Dot(axes=(2, 1))([input_points, input_tnet])
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 64)
    point_features = x
    feature_tnet = TNet(x, 64, add_regularization=True)
    x = Dot(axes=(2, 1))([x, feature_tnet])
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)
    global_feature = GlobalMaxPooling1D()(x)
    global_feature_expanded = Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = Lambda(lambda x: tf.tile(x, [1, num_points, 1]))(global_feature_expanded)
    c = Concatenate()([point_features, global_feature_expanded])
    c = conv_mlp(c, 512)
    c = conv_mlp(c, 256)
    c = conv_mlp(c, 128, dropout_rate=0.3)
    segmentation_output = Conv1D(num_classes, kernel_size=1, activation="sigmoid")(c)


    if training:
        def input_energy_weighted_loss(energy_weights, y_true, y_pred):
            # y_true should have the shape (batch_size, num_points, 1), but it has an extra dimension for some reason.
            # We need to remove any singleton dimensions to ensure it matches y_pred's shape.
            y_true = tf.squeeze(y_true, axis=-1)  # This removes the last singleton dimension if it's size 1.

            # Next, ensure the mask is applied correctly
            mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)  # define as zero for where targets are -1, and 1 otherwise
            
            # Calculate binary cross-entropy loss
            bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
            
            # Apply energy weights and mask to the loss
            weighted_loss = bce_loss * energy_weights * mask
            
            # Return the mean of the weighted loss
            return tf.reduce_mean(weighted_loss)


        model = Model(inputs=combined_input, outputs=segmentation_output)
        
        # It's crucial that 'targets' passed to input_energy_weighted_loss matches the shape of 'segmentation_output'
        model.add_loss(input_energy_weighted_loss(energy_weights, targets, segmentation_output))
    else:
        model = Model(inputs=combined_input, outputs=segmentation_output)

    return model

def masked_overall_accuracy(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    masked_loss = (1 - base_loss) * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

'''
Adapted From
https://github.com/lattice-ai/pointnet/tree/master

Original Architecture From
https://arxiv.org/pdf/1612.00593.pdf
'''


import tensorflow as tf
import numpy as np
import keras 



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


def PointNetSegmentation(num_points, num_classes, outputFcn='relu'):
    num_features = 6  # Number of input features

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
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)

    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, num_points, 1]))(global_feature_expanded)

    # Concatenate point features with global features
    c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])
    c = conv_mlp(c, 512)
    c = conv_mlp(c, 256)
    c = conv_mlp(c, 128, dropout_rate=0.3)

    # Extract energy from input and multiply by the segmentation output
    #energy = tf.expand_dims(input_points[:, :, 4], -1)  # Assuming energy is at index 4
    #segmentation_output_pre_sigmoid = tf.keras.layers.Conv1D(num_classes, kernel_size=1)(c)  # No activation yet
    #segmentation_output_pre_sigmoid = tf.keras.layers.Activation('sigmoid')(segmentation_output_pre_sigmoid)  # Apply sigmoid
    #segmentation_output = tf.multiply(segmentation_output_pre_sigmoid, energy)  # Multiply by energy

    segmentation_output = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation="relu")(c)
    #segmentation_output = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation="sigmoid")(c)  # No activation yet
    model = tf.keras.Model(inputs=input_points, outputs=segmentation_output)

    return model


def masked_bce_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False) 
    masked_loss = base_loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_mse_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    scale_factor = 100

    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_squared_error(scale_factor*y_true, scale_factor*y_pred)
    masked_loss = base_loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_mae_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    masked_loss = (1 - base_loss) * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
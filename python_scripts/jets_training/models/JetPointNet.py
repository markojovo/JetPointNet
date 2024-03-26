
'''
From
https://github.com/lattice-ai/pointnet/tree/master
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
        mask = tf.not_equal(inputs[:, :, 5], -1)
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
    

def regression_net(input_tensor, n_classes):
    x = dense_block(input_tensor, 512)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = dense_block(x, 256)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.layers.Dense(
        n_classes, activation="sigmoid"
    )(x)

def conv_block(input_tensor, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def dense_block(input_tensor, units):
    x = tf.keras.layers.Dense(units)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


class CustomBiasInitializer(tf.keras.initializers.Initializer):
    def __init__(self, features):
        self.features = features

    def __call__(self, shape, dtype=None):
        assert shape[0] == self.features * self.features
        return tf.reshape(tf.cast(tf.eye(self.features), dtype=tf.float32), [-1])

    def get_config(self):  # For saving and loading the model
        return {'features': self.features}
        
def TNet(input_tensor, num_points, features):
    x = conv_block(input_tensor, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 1024)
    x = tf.keras.layers.MaxPooling1D(pool_size=num_points)(x)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    x = tf.keras.layers.Dense(
        features * features,
        kernel_initializer="zeros",
        bias_initializer=CustomBiasInitializer(features),
        activity_regularizer=OrthogonalRegularizer(features)
    )(x)
    x = tf.reshape(x, (-1, features, features))
    return x

def point_segmentation_net(input_tensor, n_classes):
    # This block is used for per-point predictions.
    # Use a few more convolutional layers to refine the features per point.
    x = conv_block(input_tensor, 256)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = conv_block(x, 128)
    x = tf.keras.layers.Dropout(0.3)(x)
    # The final layer should predict n_classes for each point.
    return tf.keras.layers.Conv1D(
        n_classes, kernel_size=1, activation="sigmoid"  # Use softmax for classification; adjust if doing regression.
    )(x)

def PointNetRegression(num_points, n_classes):
    input_tensor = tf.keras.Input(shape=(num_points, 6))  # Assuming 6 features, including the one to check for types (-1, 0, 1, 2)
    
    x = CustomMaskingLayer()(input_tensor)
    x_t = TNet(x, num_points, 6)
    x = tf.matmul(x, x_t)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x_t = TNet(x, num_points, 64)
    x = tf.matmul(x, x_t)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 1024)

    # For segmentation, we don't use MaxPooling across all points. Instead, we proceed to per-point prediction.
    output_tensor = point_segmentation_net(x, n_classes)
    
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)



'''
#TODO: Get rid of this (bce loss does the same)

def masked_kl_divergence_loss(y_true, y_pred):
    epsilon = tf.constant(1e-7, dtype=tf.float32)  
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    # Note: This step needs careful consideration to avoid altering true labels outside the mask
    y_true_clipped = tf.clip_by_value(y_true, epsilon, 1 - epsilon)
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)

    kl_divergences = y_true_clipped * tf.math.log(y_true_clipped / y_pred) + \
                     (1 - y_true_clipped) * tf.math.log((1 - y_true_clipped) / (1 - y_pred)) 
      
    masked_loss = kl_divergences * mask  
    safe_loss = tf.where(mask > 0, masked_loss, tf.zeros_like(masked_loss))
    return tf.reduce_sum(safe_loss) / tf.reduce_sum(mask)
'''

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
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    masked_loss = base_loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_mae_loss(y_true, y_pred_outputs):
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    masked_loss = base_loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
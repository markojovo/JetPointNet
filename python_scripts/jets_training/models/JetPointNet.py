
'''
From
https://github.com/lattice-ai/pointnet/tree/master
'''


import tensorflow as tf
import numpy as np


class StreamlineLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        training_loss = logs.get('loss')
        evaluation_loss = logs.get('masked_evaluation_loss')  # Adjust the key if needed
        print(f"Epoch {epoch + 1}: Training Loss: {training_loss}, Evaluation Loss: {evaluation_loss}")

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
    
    def __init__(self, num_features, l2=0.001):
        self.num_features = num_features
        self.l2 = l2
        self.I = tf.eye(num_features)

    def __call__(self, inputs):
        A = tf.reshape(inputs, (-1, self.num_features, self.num_features))
        AAT = tf.tensordot(A, A, axes=(2, 2))
        AAT = tf.reshape(AAT, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2 * tf.square(AAT - self.I))
    

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

def PointNetRegression(num_points, n_classes):
    input_tensor = tf.keras.Input(shape=(num_points, 6))  # Assuming 6 features, including the one to check for types (-1, 0, 1, 2)
    
    # This extracts the type information (last feature) for each point
    type_info = tf.keras.layers.Lambda(lambda x: x[:, :, 5:])(input_tensor)
    
    x = CustomMaskingLayer()(input_tensor)
    x_t = TNet(x, num_points, 6)  # Adjust accordingly if your feature count changes
    x = tf.matmul(x, x_t)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x_t = TNet(x, num_points, 64)
    x = tf.matmul(x, x_t)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 1024)
    x = tf.keras.layers.MaxPooling1D(pool_size=num_points)(x)
    
    # Classification or regression output
    output_tensor = regression_net(x, n_classes)
    
    # Construct the model with two outputs: the main output and the pass-through type information
    #return tf.keras.Model(inputs=input_tensor, outputs=[output_tensor, type_info])
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)




def masked_training_loss(y_true, y_pred_outputs):
    # Access elements using TensorFlow operations
    #y_pred = y_pred_outputs[0]  # The first element: output_tensor
    #type_info = y_pred_outputs[1]  # The second element: type_info
    y_pred = y_pred_outputs
    
    mask = tf.not_equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Removes the last dimension if it's 1
    base_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    masked_loss = base_loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_evaluation_loss(y_true, y_pred_outputs):
    # Access elements using TensorFlow operations
    y_pred = y_pred_outputs[0]  # The first element: output_tensor
    type_info = y_pred_outputs[1]  # The second element: type_info
    
    mask = tf.equal(type_info, 0.0)
    mask = tf.cast(mask, tf.float32)
    
    y_true_masked = tf.multiply(y_true, mask)
    y_pred_masked = tf.multiply(y_pred, mask)
    mae_loss = tf.abs(y_true_masked - y_pred_masked)
    mae_loss = tf.reduce_sum(mae_loss) / tf.reduce_sum(mask)
    return mae_loss

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
    

def conv_mlp(input_tensor, filters):
    # Apply shared MLPs which are equivalent to 1D convolutions with kernel size 1
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return x

def dense_block(input_tensor, units, dropout_rate=None, regularizer=None):
    x = tf.keras.layers.Dense(units, kernel_regularizer=regularizer)(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def classification_head(global_feature, n_classes):
    x = dense_block(global_feature, 512, dropout_rate=0.3)
    x = dense_block(x, 256, dropout_rate=0.3)
    return tf.keras.layers.Dense(n_classes, activation="softmax")(x)


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
    num_features = 6  # Adjust based on your input features
    input_points = tf.keras.Input(shape=(num_points, num_features))

    # Input Transformation Net
    input_tnet = TNet(input_points, num_features)
    x = tf.keras.layers.Dot(axes=(2, 1))([input_points, input_tnet])
    
    # First few shared MLPs / Conv layers
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 64)
    
    # Save local features from the intermediate layer for segmentation
    point_features = x
    
    # Feature Transformation Net
    feature_tnet = TNet(x, 64, add_regularization=True)
    x = tf.keras.layers.Dot(axes=(2, 1))([x, feature_tnet])
    
    # Additional shared MLPs / Conv layers
    x = conv_mlp(x, 64)
    x = conv_mlp(x, 128)
    x = conv_mlp(x, 1024)
    
    # Global Feature Vector
    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    # Expand the global feature to be concatenated with local point features
    global_feature_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(global_feature)
    global_feature_expanded = tf.keras.layers.Tile((1, num_points))(global_feature_expanded)
    
    # Concatenate global and local features for each point
    c = tf.keras.layers.Concatenate()([point_features, global_feature_expanded])
    
    # Shared MLPs for segmentation (Can add more layers or change the sizes as needed)
    c = conv_mlp(c, 512)
    c = conv_mlp(c, 256)
    c = conv_mlp(c, 128)
    
    # Final layer to output per-point scores for each class
    segmentation_output = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation="sigmoid")(c)
    
    # Create the model
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
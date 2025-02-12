"""
A big thanks to DeepLearning.ai! Their course has helped me learn this convolutional model.
You might see some code that is very similar (or the same), but I promise you this is simply
a learning tool to help me grow as a programmer! Check out their course on Machine Learning, Deep Learning,
and Artificial Intelligence at https://www.coursera.org/learn/introduction-tensorflow
"""

import tensorflow as tf

(training_data, training_label), _ = tf.keras.datasets.mnist.load_data()

def reshape_and_normalize(images):
    """Reshapes the array of images and normalizes pixel values.

    Args:
        images (numpy.ndarray): The images encoded as numpy arrays

    Returns:
        numpy.ndarray: The reshaped and normalized images.
    """
    
    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = tf.cast(images, tf.float32)
    images = tf.expand_dims(images, axis=-1)
    
    # Normalize pixel values
    images = images / 255.0
    images = images.numpy()

    return images

(training_images, _), _ = tf.keras.datasets.mnist.load_data()
training_images = reshape_and_normalize(training_images)

class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    '''
    Will Stop our model once we reach 99.5% accuracy.
    '''
    def on_epoch_end(self, epoch, logs = None):
        if logs['accuracy'] >= 0.995:
            print("Accuracy hit 99.5%, stopping epochs.")
            self.model.stop_training = True

def ConvolutionalModel():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """
    
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape = (28,28,1)),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    
    #Compile the model
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model

#create our official model
model = ConvolutionalModel()

# run our model, set our epochs, and instill the callback
model.fit(
    training_images,
    training_label,
    epochs = 15,
    callbacks = [EarlyStoppingCallback()]
)

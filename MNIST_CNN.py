"""
A big thanks to DeepLearning.ai! Their course has helped me learn this convolutional model.
You might see some code that is very similar (or the same), but I promise you this is simply
a learning tool to help me grow as a programmer! Check out their course on Machine Learning, Deep Learning,
and Artificial Intelligence at https://www.coursera.org/learn/introduction-tensorflow
"""

#Removing rounding errors
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf


(training_images, training_labels), (validation_images, validation_labels) = tf.keras.datasets.mnist.load_data()

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


training_images = reshape_and_normalize(training_images)
validation_images = reshape_and_normalize(validation_images)


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    '''
    - Stops training when accuracy reaches 99.5% or validation accuracy reaches 98%.
    '''

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Check if we should stop training
        if logs.get("accuracy") >= 0.995 or logs.get("val_accuracy") >= 0.98:
            print("\n✅ Training accuracy hit 99.5% / Validation accuracy hit 98%. Stopping training.")
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
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    return model

# Augment Data for better training
data_augmentation = tf.keras.Sequential([
    tf.keras.Input(shape = (28,28,1)),
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2, fill_mode = 'nearest'),
    tf.keras.layers.RandomTranslation(0.2,0.2, fill_mode = 'nearest'),
    tf.keras.layers.RandomZoom(0.2, fill_mode = 'nearest')
])

#create our official model
def create_final_model():
    '''
    Creates our augmented model, then appends that to our convolutional model
    Then compiles the two for training.
    '''
    model_without_aug = ConvolutionalModel()
    model_with_aug = tf.keras.models.Sequential([
        data_augmentation,
        model_without_aug
    ])
    #Compile Model
    model_with_aug.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )

    return model_with_aug


final_model = create_final_model()
# run our model, set our epochs, and instill the callback. We set epochs and batch size numbers arbitrarily.
final_model.fit(
    training_images,
    training_labels,
    validation_data = (validation_images,validation_labels),
    epochs = 30,
    batch_size = 16,
    callbacks = [EarlyStoppingCallback()]
)
#TODO: Create a Model that fits the callback sequence for accracy AND validation accuracy
# Because I had added augmentation and dropout, the model will not get as good as it once was
# However, I believe this is still good practice for larger datasets
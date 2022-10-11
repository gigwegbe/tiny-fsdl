import glob
import logging
import os
import warnings

# warnings.filterwarnings('ignore')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import splitfolders
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Flatten, Dense, Input, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def normalize_image(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# To quantize the input and output tensors, and make the converter throw an error if it encounters an
# operation it cannot quantize, convert the model again with some additional parameters
def representative_dataset():
    """
    This function is used to generate a representative dataset for the quantization process.
    """
    for _ in range(100):
        data = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3)
        yield [data.astype(np.float32)]


LOG = logging.getLogger(__name__)


# General Params
SEED = 1337

# Data Params
LABEL_DIR = "../ml/data/raw_images/labeled_bmp_data/"
SPLIT_DIR = "../ml/data/raw_images/split_data/"
SPLIT_RATIO = (0.7, 0.2, 0.1)

# Training Params
MODEL_OUTPUT_DIR = "../ml/models/"
NUM_CLASSES = 10
BATCH_SIZE = 32
IMG_SIZE = (15, 25)
BASE_LEARNING_RATE = 0.001
EPOCHS = 100


def prep_data(label_dir, split_dir, split_ratio, batch_size, img_size, seed):
    """
    Split the data into train, validation, and test sets.
    """
    if not os.path.exists:
        splitfolders.ratio(
            label_dir,
            output=split_dir,
            seed=seed,
            ratio=split_ratio,
        )

    train_dir = os.path.join(split_dir, "train")
    validation_dir = os.path.join(split_dir, "val")

    # Train dataset
    train_dataset = image_dataset_from_directory(
        train_dir,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        image_size=img_size,
    )

    # Validation dataset
    validation_dataset = image_dataset_from_directory(
        validation_dir,
        shuffle=True,
        seed=seed,
        batch_size=batch_size,
        image_size=img_size,
    )

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    train_dataset = train_dataset.map(normalize_image)
    validation_dataset = validation_dataset.map(normalize_image)

    return train_dataset, validation_dataset, test_dataset


class Trainer:
    """
    Build and compile the model.
    """

    def __init__(self, train_dataset, validation_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.scheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.custom_early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=8, min_delta=0.001, mode="max"
        )
        self.model = None
        self.history = None

    def build_model(self, img_size, num_classes):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(img_size[0], img_size[1], 3)),
                tf.keras.layers.Conv2D(
                    32,
                    kernel_size=3,
                    kernel_constraint=tf.keras.constraints.MaxNorm(1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"),
                tf.keras.layers.Conv2D(
                    16,
                    kernel_size=3,
                    kernel_constraint=tf.keras.constraints.MaxNorm(1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="same"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                lr=BASE_LEARNING_RATE,
                beta_1=0.9,
                beta_2=0.999,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        LOG.info("Model compiled")
        LOG.info(f"Model summary: \n{model.summary()}")
        self.model = model

    def train(self, epochs):
        """
        Train the model.
        """
        # Train model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=[self.custom_early_stopping, self.scheduler_cb],
        )
        self.history = history

    def evaluate(self, validation=True, test=True):
        """
        Evaluate the model.
        """
        if not validation and not test:
            raise ValueError("Must evaluate on validation or test set.")
        if validation:
            LOG.info("Evaluating on validation set.")
            val_loss, val_acc = self.model.evaluate(self.validation_dataset)
            LOG.info(f"Validation loss: {val_loss}")
            LOG.info(f"Validation accuracy: {val_acc}")
        if test:
            LOG.info("Evaluating on test set.")
            test_loss, test_acc = self.model.evaluate(self.test_dataset)
            LOG.info(f"Test loss: {test_loss}")
            LOG.info(f"Test accuracy: {test_acc}")

    def save(self, output_dir, filename="digit_model.h5"):
        """
        Save the model.
        """
        output_file = os.path.join(output_dir, filename)
        self.model.save(output_file)
        LOG.info(f"Model saved to {output_file}")


def main():
    train_dataset, validation_dataset, test_dataset = prep_data()
    trainer = Trainer(train_dataset, validation_dataset, test_dataset)
    trainer.build_model()
    trainer.evaluate()
    trainer.train()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main()

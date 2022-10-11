import logging
import os

import numpy as np
import tensorflow as tf


def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3)
        yield [data.astype(np.float32)]


LOG = logging.getLogger(__name__)

IMG_SIZE = (15, 25)


def quantize_model(tf_model_path, output_dir, file_name, save=True):
    # Load saved model
    model = tf.keras.models.load_model(tf_model_path)

    # Apply model compression -  integer-only quantization¶
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()
    LOG.info("Converted to TFLite model")
    LOG.info(f"Quantized model size: {len(tflite_quant_model)} bytes")

    if save:
        quantized_path = os.path.join(output_dir, file_name)
        with open(quantized_path, "wb") as f:
            f.write(tflite_quant_model)
        LOG.info(f"Quantized model saved to {quantized_path}")

    return tflite_quant_model


def test_quantized_model(quantized_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=quantized_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # # Test the model on random input data.
    input_shape = input_details[0]["shape"]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)
    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]["index"])
    LOG.info(output_data)
    LOG.info(f"Input shape: {input_shape}")
    LOG.info(f"Input details: {input_details}")
    LOG.info(f"Output details: {output_details}")

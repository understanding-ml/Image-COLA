from tensorflow.keras.models import load_model

def load_mnist_cnn_model(model_path="mnist_cnn.h5"):
    """
    Loads the pre-trained MNIST CNN model from the specified file path.

    Args:
        model_path (str): Path to the model file. Default is 'mnist_cnn.h5'.

    Returns:
        model: Loaded Keras model.
    """
    model = load_model(model_path)
    return model
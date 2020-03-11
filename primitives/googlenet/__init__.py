from .cnn import ConvolutionalNeuralNetwork


__all__ = ['GoogleNetCNN', 'Dataset']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

from .gnetcnn import GoogleNetCNN


__all__ = ['MobileNetCNN', 'Dataset']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

from .vggnetcnn import VGG16CNN


__all__ = ['VGG16CNN', 'Dataset']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

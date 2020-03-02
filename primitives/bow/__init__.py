from .bag_of_words import BagOfWords


__all__ = ['BagOfWords']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

from .bag_of_characters import BagOfCharacters


__all__ = ['BagOfCharacters']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

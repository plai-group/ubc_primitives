from .dmm import DMM


__all__ = ['DMM', 'DeepMarkovModelPrimitive']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

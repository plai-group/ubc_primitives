from .logistic_regression import LogisticRegressionPrimitive


__all__ = ['LogisticRegressionPrimitive']

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

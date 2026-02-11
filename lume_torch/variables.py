"""
This module contains definitions of LUME-model variables for use with lume tools.
Variables are designed as pure descriptors and thus aren't intended to hold actual values,
but they can be used to validate encountered values.

For now, only scalar floating-point variables are supported.
"""

import logging
from typing import Optional, Type

from torch.distributions import Distribution as TDistribution
from lume.variables import Variable, ScalarVariable, ConfigEnum

logger = logging.getLogger(__name__)

# Re-export base classes for backward compatibility and clean API
__all__ = [
    "Variable",
    "ScalarVariable",
    "ConfigEnum",
    "DistributionVariable",
    "get_variable",
]


class DistributionVariable(Variable):
    """Variable for distributions. Must be a subclass of torch.distributions.Distribution.

    Attributes
    ----------
    unit : str, optional
        Unit associated with the variable.

    """

    unit: Optional[str] = None

    def validate_value(self, value: TDistribution, config: ConfigEnum = None):
        """Validates the given value.

        Parameters
        ----------
        value : Distribution
            The value to be validated.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not an instance of Distribution.

        """
        _config = self.default_validation_config if config is None else config
        # mandatory validation
        self._validate_value_type(value)
        # optional validation
        if config != "none":
            pass  # not implemented

    @staticmethod
    def _validate_value_type(value: TDistribution):
        if not isinstance(value, TDistribution):
            raise TypeError(
                f"Expected value to be of type {TDistribution}, "
                f"but received {type(value)}."
            )


def get_variable(name: str) -> Type[Variable]:
    """Returns the Variable subclass with the given name.

    Parameters
    ----------
    name : str
        Name of the Variable subclass.

    Returns
    -------
    Type[Variable]
        Variable subclass with the given name.

    """
    classes = [ScalarVariable, DistributionVariable]
    class_lookup = {c.__name__: c for c in classes}
    if name not in class_lookup.keys():
        logger.error(
            f"Unknown variable type '{name}', valid names are {list(class_lookup.keys())}"
        )
        raise KeyError(
            f"No variable named {name}, valid names are {list(class_lookup.keys())}"
        )
    return class_lookup[name]

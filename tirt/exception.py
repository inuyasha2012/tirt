from __future__ import unicode_literals, print_function, absolute_import


class ConvergenceError(Exception):
    """
    no convergence
    """


class UnknownModelError(Exception):
    """
    unknown model
    """


class ItemParamError(Exception):
    """
    Item Param Type Error
    """


class ScoreError(Exception):
    """
    score error
    """


class ThetaError(Exception):
    """
    theta error
    """


class IterMethodError(Exception):
    """
    iter method error
    """
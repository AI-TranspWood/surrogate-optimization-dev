"""AI-TranspWood surrogate-based optimization of biocomposite materials."""

__version__ = '0.1.0'


def optimize(inputSpecification, outputSpecification, mode, settings=None, model=None):
    """Optimize material parameters; import the numerical libraries on demand."""
    from .optimization import optimize as runOptimization

    return runOptimization(inputSpecification, outputSpecification, mode, settings, model)

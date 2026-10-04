"""Differential evolution with batched surrogate evaluation."""

import math

import numpy as np

from . import dataLoaders as dl
from .common import evaluate


def optimize(inputSpecification, outputSpecification, mode, settings=None, model=None):
    outputs, inputWeights = dl.validateSpecifications(inputSpecification, outputSpecification, mode)
    options = dict(maxiter=20 if mode == 'matching' else 10,
                   popsize=200 if mode == 'matching' else 1000, tol=1e-4, seed=None)
    if settings is not None:
        if not isinstance(settings, dict) or settings.keys() - options.keys():
            raise ValueError('settings supports only maxiter, popsize, tol, and seed')
        options.update(settings)
    for key, minimum in (('maxiter', 0), ('popsize', 1)):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    options['tol'] = dl.number(options['tol'], 'tol')
    if options['tol'] < 0:
        raise ValueError('tol cannot be negative')
    seed = options['seed']
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
        raise ValueError('seed must be an integer between 0 and 2**32 - 1')

    if model is None:
        from . import surrogate
        if seed is not None:
            surrogate.tf.random.set_seed(seed)
        model = surrogate.createArchitecture(inputSize=18, outputSize=5, layerSize=24)

    nFree = sum(isinstance(value, str) and value in dl.freeValues for value in inputSpecification.values())

    def objective(x):
        values = evaluate(x, inputSpecification, outputs, inputWeights, mode, model)[0]
        return float(values[0]) if np.ndim(x) == 1 else values

    if nFree:
        from scipy.optimize import differential_evolution

        result = differential_evolution(objective, [(-0.5, 0.5)] * nFree,
                                        vectorized=True, updating='deferred', polish=False, **options)
        candidate = result.x
        converged = bool(result.success)
        message = str(result.message)
        iterations, evaluations = int(result.nit), int(result.nfev)
        searchLoss = float(result.fun)
    else:
        candidate = np.empty((0, 1))
        converged, message = True, 'No free parameters; evaluated the fixed material once.'
        iterations, evaluations, searchLoss = 0, 0, None

    # Use this same final evaluation for the reported predictions, objective, and
    # feasibility: the Bayesian layer can differ from its last search evaluation.
    loss, valid, physical, inputs, predictions, constraints = evaluate(
        candidate, inputSpecification, outputs, inputWeights, mode, model)
    feasible = bool(valid[0])

    def finite(value):
        value = float(value)
        return value if math.isfinite(value) else None

    estimates = {}
    for key, prediction in predictions.items():
        estimates[key] = {field: finite(prediction[field][0])
                          for field in ('median', 'lowerBound', 'upperBound')}
        estimates[key].update(confidence=prediction['confidence'], quantiles=prediction['quantiles'])
    checks = {}
    for key, conditions in constraints.items():
        checks[key] = {field: {'limit': outputs[key][field], 'passed': bool(passed[0])}
                       for field, passed in conditions.items()}
    parameters = {key: float(physical[0, ii]) for ii, key in enumerate(dl.keyOrder)}
    parameters[dl.keyOrder[0]] = ('vonMises', 'vonMisesFisher')[int(physical[0, 0])]
    status = 'success' if converged else 'notConverged'
    if not feasible:
        status = 'noFeasibleSolution'
    return {
        'parameters': parameters,
        'effectiveModelInput': [[finite(value) for value in row] for row in inputs[0]],
        'predictions': estimates,
        'constraints': checks,
        'objectiveValue': finite(loss[0]),
        'feasible': feasible,
        'optimizer': {
            'status': status, 'converged': converged, 'message': message,
            'iterations': iterations, 'evaluations': evaluations + 1,
            'searchObjectiveValue': finite(searchLoss) if searchLoss is not None else None,
        },
    }

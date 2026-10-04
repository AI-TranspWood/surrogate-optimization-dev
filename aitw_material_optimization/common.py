"""Batched candidate construction, predictions, objectives, and constraints."""

import numpy as np

from . import dataLoaders as dl


def createInputs(x, inputSpecification):
    """Return physical candidates and tensors using the existing normalization."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    freeIndices = [ii for ii, key in enumerate(dl.keyOrder)
                   if isinstance(inputSpecification[key], str) and inputSpecification[key] in dl.freeValues]
    if x.ndim != 2 or x.shape[0] != len(freeIndices):
        raise ValueError('Candidate shape must be (number of free parameters, number of candidates)')
    physical = np.empty((x.shape[1], 18))
    scaled = np.empty_like(physical)
    counter = 0
    for ii, key in enumerate(dl.keyOrder):
        if ii in freeIndices:
            scaled[:, ii] = x[counter] + 0.5
            physical[:, ii] = dl.unscale(scaled[:, ii], ii)
            counter += 1
        else:
            value = inputSpecification[key]
            physical[:, ii] = dl.orientation(value) if ii == 0 else value
            scaled[:, ii] = dl.scale(value, ii)

    # Preserve the input loader's five-content normalization (ash excluded),
    # followed by the optimizer's six-content normalization (ash included).
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled[:, 6:11] /= np.sum(scaled[:, 6:11], axis=1, keepdims=True)
        scaled[:, 6:12] /= np.sum(scaled[:, 6:12], axis=1, keepdims=True)
    return physical, (scaled - 0.5)[:, :, np.newaxis]


def predict(model, inputs, outputs):
    distributions = model(inputs)
    locations = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    predictions = {}
    for key, (group, column) in zip(dl.outputOrder, locations):
        confidence = outputs.get(key, {}).get('confidence', 0.5)
        quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]
        values = []
        for q in (0.5, *quantiles):
            value = distributions[group].quantile(q)
            if hasattr(value, 'numpy'):
                value = value.numpy()
            values.append(np.asarray(value)[:, column])
        if key in dl.modulusKeys:
            with np.errstate(over='ignore', invalid='ignore'):
                values = [np.exp(value) for value in values]
        predictions[key] = dict(zip(('median', 'lowerBound', 'upperBound'), values))
        predictions[key]['confidence'] = confidence
        predictions[key]['quantiles'] = quantiles
    return predictions


def evaluate(x, inputSpecification, outputs, inputWeights, mode, model):
    physical, inputs = createInputs(x, inputSpecification)
    valid = np.all(np.isfinite(inputs), axis=(1, 2))
    # Invalid normalization must not feed NaNs into a batched model evaluation.
    predictions = predict(model, np.where(valid[:, None, None], inputs, 0.0), outputs)
    constraints = {}
    for key, prediction in predictions.items():
        for field in ('median', 'lowerBound', 'upperBound'):
            valid &= np.isfinite(prediction[field])
        spec = outputs.get(key, {})
        checks = {}
        if 'min' in spec:
            checks['min'] = prediction['lowerBound'] >= spec['min']
        if 'max' in spec:
            checks['max'] = prediction['upperBound'] <= spec['max']
        if 'maxUncertainty' in spec:
            width = prediction['upperBound'] - prediction['lowerBound']
            checks['maxUncertainty'] = width <= spec['maxUncertainty']
        if checks:
            constraints[key] = checks
            for passed in checks.values():
                valid &= passed

    loss = np.zeros(physical.shape[0])
    if mode == 'matching':
        nTargets = 0
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for key, spec in outputs.items():
                if 'target' not in spec:
                    continue
                target, median = spec['target'], predictions[key]['median']
                if key in dl.modulusKeys:
                    target, median = np.log(target), np.log(median)
                loss += (target - median) ** 2
                nTargets += 1
            loss /= nTargets
    else:
        for ii, key in enumerate(dl.keyOrder):
            direction = inputSpecification[key]
            if isinstance(direction, str) and direction in ('minimize', 'maximize'):
                sign = 1 if direction == 'minimize' else -1
                loss += sign * inputWeights[key] * physical[:, ii]
        for key, spec in outputs.items():
            if 'objective' in spec:
                sign = 1 if spec['objective'] == 'minimize' else -1
                loss += sign * spec['weight'] * predictions[key]['median']
    valid &= np.isfinite(loss)
    loss = np.where(valid, loss, np.inf)
    return loss, valid, physical, inputs, predictions, constraints

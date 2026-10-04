"""JSON specifications and the surrogate's physical parameter scaling."""

import json
import math
from numbers import Real


keyOrder = [
    'fiberOrientationDistribution', 'fiberAspectRatio', 'orientationConcentration',
    'microfibrilAngle', 'lumenPorosity', 'fiberVolumeFraction', 'celluloseContent',
    'hemicelluloseContent', 'ligninContent', 'pectinContent', 'extractivesContent',
    'ashContent', 'celluloseCrystallinity', 'matrixYoungsModulus', 'matrixPoissonRatio',
    'airPorosity', 'tangentialInterfaceCompliance', 'longitudinalInterfaceCompliance',
]

# Physical bounds. Zero is mapped to 1e-10 only for the logarithmic transform.
limits = [
    (0, 1), (1.2, 20000), (0, 100), (0, 45), (0, 0.99), (0, 1),
    (0.05, 0.9), (0.01, 0.5), (0.01, 0.6), (0, 0.15), (0, 0.15), (0, 0.20),
    (0.20, 1), (0, 100), (0, 0.5), (0, 0.5), (0, 1e5), (0, 1e5),
]
logIndices = (1, 2, 16, 17)
outputOrder = ['E_T', 'E_L', 'nu_T', 'nu_LT', 'mu_LT']
modulusKeys = ('E_T', 'E_L', 'mu_LT')
freeValues = ('optimize', 'minimize', 'maximize')


def loadInput(filePath):
    with open(filePath, encoding='utf-8') as file:
        return json.load(file)


def saveResult(result, filePath):
    with open(filePath, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def orientation(value):
    if isinstance(value, str) and value in ('vonMises', 'vonMisesFisher'):
        return ('vonMises', 'vonMisesFisher').index(value)
    if not isinstance(value, bool) and isinstance(value, Real) and value in (0, 1):
        return int(value)
    raise ValueError('fiberOrientationDistribution must be 0, 1, vonMises, or vonMisesFisher')


def scale(value, ii):
    if ii == 0:
        return float(orientation(value))
    value = number(value, keyOrder[ii])
    lower, upper = limits[ii]
    if not lower <= value <= upper:
        raise ValueError(f'{keyOrder[ii]} must be within [{lower}, {upper}]')
    if ii in logIndices:
        lower = math.log10(max(lower, 1e-10))
        upper = math.log10(upper)
        value = math.log10(max(value, 1e-10))
    return (value - lower) / (upper - lower)


def unscale(value, ii):
    lower, upper = limits[ii]
    if ii in logIndices:
        lower = math.log10(max(lower, 1e-10))
        upper = math.log10(upper)
        return 10 ** (lower + value * (upper - lower))
    return lower + value * (upper - lower)


def validateSpecifications(inputSpecification, outputSpecification, mode):
    if mode not in ('matching', 'constrained'):
        raise ValueError('mode must be matching or constrained')
    if not isinstance(inputSpecification, dict):
        raise ValueError('Input specification must be a JSON object')
    missing = set(keyOrder) - inputSpecification.keys()
    unknown = inputSpecification.keys() - set(keyOrder)
    if missing or unknown:
        raise ValueError(f'Input parameters: missing {sorted(missing)}, unknown {sorted(unknown)}')
    orientation(inputSpecification[keyOrder[0]])
    inputObjectives = {}
    for ii, key in enumerate(keyOrder[1:], 1):
        value = inputSpecification[key]
        if isinstance(value, str) and value in freeValues:
            if value != 'optimize':
                inputObjectives[key] = value
        else:
            scale(value, ii)

    if not isinstance(outputSpecification, dict):
        raise ValueError('Output specification must be a JSON object')
    if 'outputs' in outputSpecification:
        if outputSpecification.keys() - {'outputs', 'inputWeights'}:
            raise ValueError('Output specification supports only outputs and inputWeights')
        outputs = outputSpecification['outputs']
    else:
        outputs = {key: value for key, value in outputSpecification.items() if key != 'inputWeights'}
    if not isinstance(outputs, dict):
        raise ValueError('outputs must be an object')
    if outputs.keys() - set(outputOrder):
        raise ValueError(f'Unknown output names. Valid outputs: {", ".join(outputOrder)}')
    inputWeights = outputSpecification.get('inputWeights', {})
    if not isinstance(inputWeights, dict) or inputWeights.keys() - inputObjectives.keys():
        raise ValueError('inputWeights must name only minimize/maximize input objectives')

    cleaned = {}
    fields = {'target', 'objective', 'weight', 'min', 'max', 'confidence', 'maxUncertainty'}
    for key, spec in outputs.items():
        if not isinstance(spec, dict) or spec.keys() - fields:
            raise ValueError(f'{key} must contain only {", ".join(sorted(fields))}')
        item = {}
        for field in ('target', 'min', 'max', 'maxUncertainty', 'weight', 'confidence'):
            if field in spec:
                if field in ('min', 'max') and spec[field] == '':
                    continue
                item[field] = number(spec[field], f'{key}.{field}')
        if 'objective' in spec:
            if spec['objective'] not in ('minimize', 'maximize'):
                raise ValueError(f'{key}.objective must be minimize or maximize')
            item['objective'] = spec['objective']
        if 'target' in item and key in modulusKeys and item['target'] <= 0:
            raise ValueError(f'{key}.target must be positive for logarithmic residuals')
        if 'min' in item and 'max' in item and item['min'] > item['max']:
            raise ValueError(f'{key}.min cannot exceed max')
        if 'maxUncertainty' in item and item['maxUncertainty'] < 0:
            raise ValueError(f'{key}.maxUncertainty cannot be negative')
        item.setdefault('confidence', 0.5)
        if not 0 < item['confidence'] < 1:
            raise ValueError(f'{key}.confidence must be strictly between 0 and 1')
        if 'weight' in item and ('objective' not in item or item['weight'] <= 0):
            raise ValueError(f'{key}.weight requires an objective and must be positive')
        cleaned[key] = item

    outputObjectives = {key: spec for key, spec in cleaned.items() if 'objective' in spec}
    targets = {key: spec for key, spec in cleaned.items() if 'target' in spec}
    if mode == 'matching':
        if not targets or inputObjectives or outputObjectives or inputWeights:
            raise ValueError('matching requires targets and does not allow directed objectives or weights')
    else:
        if targets or not (inputObjectives or outputObjectives):
            raise ValueError('constrained requires directed objectives and does not allow targets')
    multiple = len(inputObjectives) + len(outputObjectives) > 1
    weights = {}
    for key in inputObjectives:
        if multiple and key not in inputWeights:
            raise ValueError(f'An explicit weight is required for {key} with multiple objectives')
        weights[key] = number(inputWeights.get(key, 1), f'inputWeights.{key}')
        if weights[key] <= 0:
            raise ValueError(f'inputWeights.{key} must be positive')
    for key, spec in outputObjectives.items():
        if multiple and 'weight' not in spec:
            raise ValueError(f'An explicit weight is required for {key} with multiple objectives')
        spec.setdefault('weight', 1.0)
    return cleaned, weights

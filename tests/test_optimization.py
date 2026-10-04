import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
from click.testing import CliRunner

from aitw_material_optimization import optimize
from aitw_material_optimization import dataLoaders as dl
from aitw_material_optimization.cli import aitw_material_optimization
from aitw_material_optimization.common import createInputs, evaluate


root = Path(__file__).resolve().parents[1]


@pytest.fixture
def material():
    return json.loads((root / 'input.json').read_text())


class Distribution:
    def __init__(self, loc, scale):
        self.loc = np.asarray(loc)
        self.scale = np.broadcast_to(scale, self.loc.shape)

    def quantile(self, q):
        return self.loc + NormalDist().inv_cdf(q) * self.scale


class Model:
    """Deterministic model double with five independent output distributions."""
    def __init__(self):
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        fraction = inputs[:, 5, 0] + 0.5
        moduli = np.column_stack((np.log(5 + 20 * fraction), np.log(3 + 10 * fraction)))
        ratios = np.tile([0.2, 0.3], (len(inputs), 1))
        shear = np.log(2 + 5 * fraction)[:, None]
        return [Distribution(moduli, [0.02, 0.04]), Distribution(ratios, [0.01, 0.03]),
                Distribution(shear, 0.08)]


def evaluateSpecification(material, output, mode='matching', x=None, model=None):
    outputs, weights = dl.validateSpecifications(material, output, mode)
    if x is None:
        x = np.empty((0, 1))
    return evaluate(x, material, outputs, weights, mode, model or Model())


def test_scaling_and_orientation():
    for ii, (lower, upper) in enumerate(dl.limits[1:], 1):
        assert dl.scale(lower, ii) == pytest.approx(0)
        assert dl.scale(upper, ii) == pytest.approx(1)
        value = dl.unscale(0.3, ii)
        assert dl.scale(value, ii) == pytest.approx(0.3)
    for ii in (2, 16, 17):
        assert dl.scale(0, ii) == dl.scale(1e-10, ii)
    assert dl.orientation('vonMises') == dl.orientation(0) == 0
    assert dl.orientation('vonMisesFisher') == dl.orientation(1) == 1


def test_legacy_preprocessing_and_named_candidates(material):
    scaled = np.array([dl.scale(material[key], ii) for ii, key in enumerate(dl.keyOrder)])
    scaled[6:11] /= scaled[6:11].sum()
    scaled[6:12] /= scaled[6:12].sum()
    physical, inputs = createInputs(np.empty((0, 1)), material)
    np.testing.assert_allclose(inputs[0, :, 0], scaled - 0.5)
    assert physical[0, 6] == material['celluloseContent']
    material['fiberVolumeFraction'] = 'optimize'
    material['longitudinalInterfaceCompliance'] = 'optimize'
    physical, inputs = createInputs(np.array([[-0.3, 0.2], [-0.5, 0.5]]), material)
    np.testing.assert_allclose(physical[:, 5], [0.2, 0.7])
    np.testing.assert_allclose(physical[:, 17], [1e-10, 1e5])
    assert inputs.shape == (2, 18, 1)


@pytest.mark.parametrize('key, expected', [('E_T', 7), ('E_L', 4), ('nu_T', 0.2),
                                         ('nu_LT', 0.3), ('mu_LT', 2.5)])
def test_partial_matching_and_all_outputs(material, key, expected):
    loss, valid, _, _, predictions, _ = evaluateSpecification(material, {key: {'target': expected}})
    assert loss[0] == pytest.approx(0)
    assert valid[0]
    assert predictions[key]['median'][0] == pytest.approx(expected)


def test_mean_loss_preserves_log_residuals(material):
    loss = evaluateSpecification(material, {'E_T': {'target': 7 * math.e},
                                            'nu_T': {'target': 0.4}})[0]
    assert loss[0] == pytest.approx((1 + 0.2**2) / 2)


@pytest.mark.parametrize('key, scale, logged', [('E_T', 0.02, True), ('E_L', 0.04, True),
                                               ('nu_T', 0.01, False), ('nu_LT', 0.03, False),
                                               ('mu_LT', 0.08, True)])
def test_each_interval_uses_its_own_distribution(material, key, scale, logged):
    result = optimize(material, {key: {'target': 1, 'confidence': 0.8}}, 'matching', model=Model())
    prediction = result['predictions'][key]
    z = NormalDist().inv_cdf(0.9)
    if logged:
        assert math.log(prediction['upperBound'] / prediction['median']) == pytest.approx(z * scale)
    else:
        assert prediction['upperBound'] - prediction['median'] == pytest.approx(z * scale)


@pytest.mark.parametrize('direction, sign', [('minimize', 1), ('maximize', -1)])
@pytest.mark.parametrize('key, expected', [('E_T', 7), ('E_L', 4), ('nu_T', 0.2),
                                         ('nu_LT', 0.3), ('mu_LT', 2.5)])
def test_output_objectives(material, key, expected, direction, sign):
    loss = evaluateSpecification(material, {key: {'objective': direction}}, 'constrained')[0]
    assert loss[0] == pytest.approx(sign * expected)


@pytest.mark.parametrize('direction, sign', [('minimize', 1), ('maximize', -1)])
def test_input_objective_and_vectorized_shape(material, direction, sign):
    material['fiberVolumeFraction'] = direction
    loss = evaluateSpecification(material, {}, 'constrained', np.array([[-0.3, 0.2]]))[0]
    np.testing.assert_allclose(loss, sign * np.array([0.2, 0.7]))
    scalarLoss = evaluateSpecification(material, {}, 'constrained', np.array([-0.3]))[0]
    assert scalarLoss.shape == (1,)


def test_weighted_tradeoff(material):
    material['fiberVolumeFraction'] = 'minimize'
    spec = {'outputs': {'E_T': {'objective': 'maximize', 'weight': 2}},
            'inputWeights': {'fiberVolumeFraction': 10}}
    loss = evaluateSpecification(material, spec, 'constrained', np.array([[-0.3, 0.2]]))[0]
    np.testing.assert_allclose(loss, 10 * np.array([0.2, 0.7]) - 2 * np.array([9, 19]))
    with pytest.raises(ValueError, match='explicit weight'):
        dl.validateSpecifications(material, {'E_T': {'objective': 'maximize'}}, 'constrained')


def test_central_interval_constraints_and_uncertainty(material):
    base = {'E_T': {'target': 7, 'min': 6.8, 'max': 7.2, 'confidence': 0.5,
                   'maxUncertainty': 0.3}}
    loss, valid, _, _, _, checks = evaluateSpecification(material, base)
    assert valid[0] and np.isfinite(loss[0])
    assert all(passed[0] for passed in checks['E_T'].values())
    for change in ({'min': 7}, {'max': 7}, {'maxUncertainty': 0.01}, {'confidence': 0.99}):
        output = copy.deepcopy(base)
        output['E_T'].update(change)
        loss, valid, *_ = evaluateSpecification(material, output)
        assert not valid[0] and np.isinf(loss[0])


def test_fixed_material_once_and_json(material, tmp_path):
    model = Model()
    material['orientationConcentration'] = 0
    result = optimize(material, {'E_T': {'target': 7}}, 'matching', model=model)
    assert model.calls == 1
    assert result['parameters']['orientationConcentration'] == 0
    assert result['optimizer']['evaluations'] == 1
    assert result['feasible']
    path = tmp_path / 'result.json'
    dl.saveResult(result, path)
    assert json.loads(path.read_text()) == result


def test_infeasible_result_and_invalid_normalization(material):
    result = optimize(material, {'E_T': {'target': 7, 'min': 100}}, 'matching', model=Model())
    assert not result['feasible']
    assert result['objectiveValue'] is None
    assert result['optimizer']['status'] == 'noFeasibleSolution'
    json.dumps(result, allow_nan=False)
    for ii in range(6, 12):
        material[dl.keyOrder[ii]] = dl.limits[ii][0]
    result = optimize(material, {'E_T': {'target': 7}}, 'matching', model=Model())
    assert not result['feasible']
    json.dumps(result, allow_nan=False)


def test_final_evaluation_drives_feasibility(material):
    class VaryingModel(Model):
        def __call__(self, inputs):
            distributions = super().__call__(inputs)
            if self.calls > 1:
                distributions[0].loc[:, 0] += 1
            return distributions

    model = VaryingModel()
    model(np.zeros((1, 18, 1)))
    result = optimize(material, {'E_T': {'target': 7, 'max': 8}}, 'matching', model=model)
    assert not result['feasible']
    assert result['predictions']['E_T']['median'] > 8


@pytest.mark.parametrize('output', [
    {'nu_L': {}}, {'E_T': {'target': 0}}, {'E_T': {'target': 1, 'confidence': 1}},
    {'E_T': {'target': 1, 'min': 2, 'max': 1}},
    {'E_T': {'target': 1, 'maxUncertainty': -1}}, {'E_T': {'target': float('nan')}},
    {'E_T': {'target': 1, 'weight': 2}}, {'E_T': {'target': 1, 'unknown': 1}},
    {'E_T': {'objective': 'maximize'}}, {'outputs': []},
])
def test_invalid_output_specs(material, output):
    with pytest.raises(ValueError):
        dl.validateSpecifications(material, output, 'matching')


@pytest.mark.parametrize('key, value', [('fiberOrientationDistribution', True),
                                      ('fiberVolumeFraction', 1.1), ('microfibrilAngle', -1),
                                      ('ashContent', None), ('fiberAspectRatio', float('inf'))])
def test_invalid_inputs(material, key, value):
    material[key] = value
    with pytest.raises(ValueError):
        dl.validateSpecifications(material, {'E_T': {'target': 10}}, 'matching')


@pytest.mark.parametrize('settings', [{'maxiter': -1}, {'popsize': 0}, {'tol': float('nan')},
                                      {'tol': -1}, {'seed': -1}, {'seed': True}, {'unknown': 1}])
def test_invalid_settings(material, settings):
    with pytest.raises(ValueError):
        optimize(material, {'E_T': {'target': 7}}, 'matching', settings, model=Model())


def test_report_examples_and_flat_format():
    for name in ('report-material-1.json', 'report-material-2.json'):
        material = dl.loadInput(root / 'example-input' / name)
        spec = dl.loadInput(root / 'example-input/minimum-stiffness.json')
        outputs, weights = dl.validateSpecifications(material, spec, 'constrained')
        assert outputs['E_T']['min'] == 10
        assert 'max' not in outputs['E_T']
        assert weights['fiberVolumeFraction'] == 1


def test_cli_help_is_lazy():
    code = ('import sys; from click.testing import CliRunner; '
            'from aitw_material_optimization.cli import aitw_material_optimization; '
            'r = CliRunner().invoke(aitw_material_optimization, ["--help"]); '
            'assert r.exit_code == 0; '
            'assert not any(m in sys.modules for m in ["tensorflow", "numpy", "scipy"])')
    subprocess.run([sys.executable, '-c', code], cwd=root, check=True)


def test_cli_and_api_agree(material, tmp_path, monkeypatch):
    import aitw_material_optimization as package

    original = package.optimize
    monkeypatch.setattr(package, 'optimize', lambda *args, **kwargs: original(*args, **kwargs, model=Model()))
    inputPath, outputPath, resultPath = [tmp_path / name for name in ('input.json', 'outputs.json', 'result.json')]
    dl.saveResult(material, inputPath)
    dl.saveResult({'mu_LT': {'target': 2.5}}, outputPath)
    cli = CliRunner().invoke(aitw_material_optimization,
                            [str(inputPath), str(outputPath), str(resultPath), '--mode', 'matching'])
    assert cli.exit_code == 0, cli.output
    assert json.loads(resultPath.read_text()) == original(material, {'mu_LT': {'target': 2.5}},
                                                        'matching', model=Model())
    dl.saveResult({'nu_L': {'target': 1}}, outputPath)
    cli = CliRunner().invoke(aitw_material_optimization,
                            [str(inputPath), str(outputPath), str(resultPath), '--mode', 'matching'])
    assert cli.exit_code != 0
    assert 'Valid outputs' in cli.output


def test_real_differential_evolution_with_model_double(material):
    pytest.importorskip('scipy')
    material['fiberVolumeFraction'] = 'optimize'
    result = optimize(material, {'E_T': {'target': 13}}, 'matching',
                      {'maxiter': 50, 'popsize': 10, 'seed': 1}, Model())
    assert result['parameters']['fiberVolumeFraction'] == pytest.approx(0.4, abs=0.002)
    assert result['objectiveValue'] < 1e-6
    material['fiberVolumeFraction'] = 'minimize'
    result = optimize(material, {'E_T': {'min': 10, 'confidence': 0.5}}, 'constrained',
                      {'maxiter': 50, 'popsize': 10, 'seed': 1}, Model())
    expected = (10 * math.exp(NormalDist().inv_cdf(0.75) * 0.02) - 5) / 20
    assert result['feasible']
    assert result['parameters']['fiberVolumeFraction'] == pytest.approx(expected, abs=0.003)


def test_bundled_model_smoke(tmp_path, monkeypatch):
    pytest.importorskip('tensorflow')
    pytest.importorskip('tensorflow_probability')
    pytest.importorskip('scipy')
    from aitw_material_optimization import surrogate

    monkeypatch.chdir(tmp_path)
    model = surrogate.createArchitecture(18, 5, 24)
    for mode, inputName, outputName in (
            ('matching', 'matching-material.json', 'matching-outputs.json'),
            ('constrained', 'report-material-1.json', 'minimum-stiffness.json')):
        result = optimize(dl.loadInput(root / 'example-input' / inputName),
                          dl.loadInput(root / 'example-input' / outputName), mode,
                          {'maxiter': 1, 'popsize': 5, 'seed': 1}, model)
        assert set(result['predictions']) == set(dl.outputOrder)
        assert result['optimizer']['iterations'] <= 1
        json.dumps(result, allow_nan=False)

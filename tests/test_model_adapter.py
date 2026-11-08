"""Checkpoint compatibility and probability averaging without trained model files."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.service import InputError, ModelService


class Network:
    def __init__(self, *sizes, pretrained, arch):
        assert sizes == (2, 2, 2)
        assert not pretrained
        self.arch = arch
        self.evaluating = False

    def load_state_dict(self, weights):
        self.logits = torch.tensor([weights], dtype=torch.float32)

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, x):
        assert self.evaluating
        return [self.logits] * 3


def service():
    instance = ModelService(Path('.'))
    instance.original = SimpleNamespace(CarNet=Network)
    instance.torch = torch
    return instance


def checkpoint(**changes):
    return {'classes': {k: ['first', 'second'] for k in ['model', 'color', 'body']}, **changes}


def test_legacy_checkpoint_remains_supported():
    instance = service()
    instance.load_recognizer(checkpoint(weights=[2., 0.]))
    assert [n.arch for n in instance.nets] == ['convnext_tiny']
    np.testing.assert_allclose(instance.predict_probabilities(None)[0], torch.softmax(torch.tensor([2., 0.]), 0))


def test_ensemble_averages_probabilities_instead_of_logits():
    instance = service()
    instance.load_recognizer(checkpoint(members=[
        {'arch': 'convnext_tiny', 'weights': [5., 0.]},
        {'arch': 'efficientnet_v2_s', 'weights': [0., 1.]},
        {'arch': 'resnet50', 'weights': [0., 2.]},
    ]))
    assert [n.arch for n in instance.nets] == ['convnext_tiny', 'efficientnet_v2_s', 'resnet50']
    expected = torch.softmax(torch.tensor([[5., 0.], [0., 1.], [0., 2.]]), 1).mean(0).numpy()
    for result in instance.predict_probabilities(None):
        np.testing.assert_allclose(result, expected)
        assert np.isclose(result.sum(), 1)


def test_invalid_checkpoint_has_a_clear_error():
    with pytest.raises(InputError, match='нет весов'):
        service().load_recognizer(checkpoint(members=[]))


def test_incompatible_price_file_reports_error_instead_of_exiting(monkeypatch, tmp_path):
    import app.service as module

    def incompatible_price():
        raise SystemExit('Переобучите модель цены: python train.py --price-only')

    fake_train = SimpleNamespace(CarNet=Network, PriceModel=object, Specs=object)
    fake_predict = SimpleNamespace(Pricer=incompatible_price)
    original_import = module.importlib.import_module
    monkeypatch.setattr(module.importlib, 'import_module',
                        lambda name, *args, **kwargs: {'train': fake_train, 'predict': fake_predict}[name]
                        if name in ('train', 'predict') else original_import(name, *args, **kwargs))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint(weights=[2., 0.], img_size=224))
    instance = ModelService(tmp_path)
    instance.load()
    assert instance.status == 'error'
    assert '--price-only' in instance.load_error

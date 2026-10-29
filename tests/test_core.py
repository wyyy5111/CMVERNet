import numpy as np
import torch
from cmvernet.features import stable_features, compact_features
from cmvernet.model_stable import V45Config, V45SalienceStratifiedPerceiver
from cmvernet.compact import predict_outer_fold
from cmvernet.aurora_router import AuroraConfig, fit_context_gate, confidence_risk_transport
from cmvernet.historical_routing import route_oof


def test_feature_assembly_and_model():
    raw = np.zeros((2, 13, 1536), dtype=np.float32)
    conditioned = np.ones_like(raw)
    side = np.full((2, 88), 2, dtype=np.float32)
    static = np.full((2, 13632), 3, dtype=np.float32)
    stable = stable_features(raw, conditioned, side)
    compact = compact_features(static, raw, conditioned)
    assert stable.shape == (2, 40024) and compact.shape == (2, 53568)
    assert np.all(stable[:, -88:] == 2) and np.all(compact[:, :13632] == 3)
    torch.set_num_threads(1)
    model = V45SalienceStratifiedPerceiver(variant="latent-cross-attention", config=V45Config(classes=3))
    out = model(torch.zeros(2, 3072), torch.zeros(2, 1024))["ensemble_probability"]
    assert out.shape == (2, 3)
    assert torch.allclose(out.sum(dim=1), torch.ones(2))


def test_compact_and_training_only_centering():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 48)).astype(np.float32)
    y = np.arange(30) % 3
    pred = predict_outer_fold(x, y, np.arange(30) % 5, np.arange(24), np.arange(24, 30), k=32)
    assert pred.shape == (6,) and set(pred) <= {0, 1, 2}


def test_routing():
    cfg = AuroraConfig(confidence_margin_boundary=0.45)
    stable = np.array([[0.55, 0.45], [0.95, 0.05]])
    context = np.array([[0.2, 0.8], [0.3, 0.7]])
    transported, route = confidence_risk_transport(stable, np.array([1, 1]), cfg)
    assert route.tolist() == [True, False]
    assert np.allclose(transported, [[0.45, 0.55], [0.95, 0.05]])
    assert fit_context_gate(np.array([0.1, np.nan]), cfg)["context_active"]
    result = route_oof(stable, np.array([1, 1]), context, np.array([1, 2]), np.array([0.2, np.nan]), 0.45)
    assert np.allclose(result[0], context[0]) and np.allclose(result[1], stable[1])

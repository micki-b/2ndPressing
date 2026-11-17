import pytest

torch = pytest.importorskip("torch")

from pressing_tools.nn.model import (ClickNet, NetConfig, macs_per_sample,  # noqa: E402
                                     param_count_formula, receptive_field)


def _randomised(cfg):
    torch.manual_seed(0)
    net = ClickNet(cfg).eval()
    with torch.no_grad():               # zero-initialised branches would hide dependencies
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net


@pytest.mark.parametrize("C", [8, 16, 24])
def test_params_and_macs(C):
    cfg = NetConfig(C=C)
    net = ClickNet(cfg)
    n = sum(p.numel() for p in net.parameters())
    assert n == param_count_formula(cfg) == 33 * C * C + 46 * C + 4
    assert macs_per_sample(cfg) == pytest.approx(8.25 * C * C + 7 * C)
    assert receptive_field(cfg) == 2048


def test_causality_and_receptive_field():
    cfg = NetConfig(C=8)
    net = _randomised(cfg)
    T = 8192
    x = torch.randn(1, 3, T)
    base = net(x)
    assert base.shape == (1, T)
    for k in (1000, 3001, 4094):
        y = x.clone()
        y[0, :, k] += 5.0
        d = (net(y) - base).abs()[0] > 1e-6
        changed = torch.nonzero(d).flatten()
        assert changed.min() >= 4 * (k // 4)              # nothing before k's frame
        assert changed.max() < k + receptive_field(cfg)   # nothing beyond the receptive field

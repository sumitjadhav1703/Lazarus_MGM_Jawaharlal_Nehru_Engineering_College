import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("batch", [1, 2])
def test_residual_chunk_encoder_shape_finite_gradients_and_parameter_budget(width, batch):
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    model = ResidualChunkEncoder(width)
    values = torch.randn(batch, 3, 500, requires_grad=True)
    encoded = model(values)
    assert encoded.shape == (batch, width, 1)
    assert torch.isfinite(encoded).all()
    encoded.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_residual_chunk_encoder_does_not_mix_statistics_across_chunks_in_train_mode():
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    model = ResidualChunkEncoder().train()
    assert not any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) for layer in model.modules())
    first = torch.randn(1, 3, 500)
    second = torch.randn(1, 3, 500) * 100 + 500
    with torch.no_grad():
        alone = model(first)
        together = model(torch.cat([first, second]))[:1]
    torch.testing.assert_close(alone, together, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("width", [0, 8, 17, 64])
def test_residual_chunk_encoder_rejects_undeclared_widths(width):
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    with pytest.raises(ValueError, match="16 and 32"):
        ResidualChunkEncoder(width)


@pytest.mark.parametrize("shape", [(3, 500), (1, 2, 500), (1, 3, 499), (1, 3, 1000)])
def test_residual_chunk_encoder_rejects_incomplete_or_nonchunk_inputs(shape):
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    with pytest.raises(ValueError, match="completed.*500"):
        ResidualChunkEncoder()(torch.randn(shape))


def test_residual_chunk_encoder_preserves_existing_waveform_temporal_interface():
    from cardiotwin.neural import WaveformCNN
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    model = WaveformCNN(width=16)
    model.encoder = ResidualChunkEncoder(width=16)
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 1000))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_residual_waveform_candidate_uses_shared_chunk_encoder_and_causal_head():
    from cardiotwin.neural import WaveformResNet
    from cardiotwin.neural_architectures import ResidualChunkEncoder

    model = WaveformResNet(16).eval()
    assert isinstance(model.encoder, ResidualChunkEncoder)
    values = torch.randn(2, 3, 1000)
    logits = model(values)
    assert logits.shape == (2,)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    tokens = torch.randn(1, 16, 20)
    changed = tokens.clone()
    changed[:, :, 10:] += 100
    torch.testing.assert_close(model.temporal(tokens)[:, :, :10], model.temporal(changed)[:, :, :10])


def test_residual_candidate_can_fit_and_restore_state(tmp_path):
    import numpy as np

    from cardiotwin.neural import WaveformResNet, fit_network, network_scores

    fit = [dict(patient_id="fit", accepted=True, label=label) for label in (0, 1)]
    stop = [dict(patient_id="stop", accepted=True, label=label) for label in (0, 1)]

    def provider(row):
        values = np.ones((3, 500), dtype=np.float32)
        values[0] *= 0.5 + row["label"]
        return values

    model, normalizer, report = fit_network(
        fit, stop, provider, representation="waveform_resnet", max_epochs=1, batch_size=2, device="cpu"
    )
    assert isinstance(model, WaveformResNet) and report["representation"] == "waveform_resnet"
    original = network_scores(model, normalizer, stop, provider)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = WaveformResNet(16)
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    np.testing.assert_allclose(original, network_scores(restored, normalizer, stop, provider))

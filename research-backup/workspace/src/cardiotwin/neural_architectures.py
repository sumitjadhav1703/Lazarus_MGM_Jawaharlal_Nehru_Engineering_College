"""Optional residual encoding of completed five-second ECG chunks."""

try:
    from torch import nn
except ImportError as error:
    raise ImportError("Neural experiments require the optional cardiotwin[neural] dependencies") from error


class _ResidualChunkBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 7, stride=stride, padding=3, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, 7, padding=3, bias=False),
            nn.GroupNorm(4, output_channels),
        )
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(4, output_channels),
            )
        )

    def forward(self, values):
        return nn.functional.relu(self.layers(values) + self.shortcut(values))


class ResidualChunkEncoder(nn.Module):
    """Two ECG leads plus validity mask: (B,3,500) -> (B,width,1).

    Symmetric convolutions and GroupNorm operate only inside a completed chunk.
    The existing causal temporal head remains responsible for token chronology.
    """

    def __init__(self, width=16):
        super().__init__()
        if width not in (16, 32):
            raise ValueError("Prespecified residual widths are 16 and 32")
        self.layers = nn.Sequential(
            nn.Conv1d(3, 16, 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            _ResidualChunkBlock(16, 16),
            _ResidualChunkBlock(16, 32, stride=2),
            _ResidualChunkBlock(32, 32, stride=2),
            nn.Conv1d(32, width, 1),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (3, 500):
            raise ValueError("Residual encoder requires completed two-lead/mask chunks of 500 samples")
        return self.layers(values)

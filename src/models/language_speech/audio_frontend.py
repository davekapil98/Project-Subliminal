"""Stage 0 frontend for precomputed log-Mel audio features."""

from torch import Tensor, nn


class AudioFrontend(nn.Module):
    def __init__(self, mel_bins: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(mel_bins, d_model), nn.LayerNorm(d_model), nn.SiLU()
        )

    def forward(self, log_mel: Tensor) -> Tensor:
        if log_mel.ndim != 3:
            raise ValueError("log_mel must have shape [B, T, mel_bins]")
        return self.projection(log_mel)

import torch


def pytest_sessionstart() -> None:
    # Small tensors are much faster without a large CPU thread-pool startup cost.
    torch.set_num_threads(1)

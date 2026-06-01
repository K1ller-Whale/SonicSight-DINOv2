"""pytest configuration and shared fixtures."""
import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)

import numpy as np
import pytest

from thesis.data.datasets import load_opportunity, load_pamap2

@pytest.mark.parametrize("loader", [load_opportunity, load_pamap2])
def test_dataset_smoke(loader):
    X, y = loader(n_samples=1_000)
    assert X.shape[0] == y.shape[0] == 1_000
    assert X.ndim == 2
    assert y.ndim == 1
    # basic sanity – finite numbers
    assert np.isfinite(X).all()
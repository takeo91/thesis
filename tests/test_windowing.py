import numpy as np
import pytest

from thesis.data import WindowedData, train_test_split_windows


@pytest.fixture()

def simple_windowed_data() -> WindowedData:
    """Create a small synthetic WindowedData fixture with 3 classes."""
    n_per_class = 20
    n_classes = 3
    n_windows = n_per_class * n_classes
    window_size = 16
    n_features = 4

    # Synthetic data: shape (n_windows, window_size, n_features)
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(n_windows, window_size, n_features))

    # Labels: 0,1,2 repeated
    labels = np.repeat(np.arange(n_classes), n_per_class)

    # Dummy indices
    starts = np.arange(n_windows)[:, None]
    window_indices = np.hstack([starts, starts + window_size])

    metadata = {"description": "synthetic fixture"}
    return WindowedData(windows=windows, labels=labels, window_indices=window_indices, metadata=metadata)


def _check_integrity(full: WindowedData, lib: WindowedData, qry: WindowedData):
    """Helper to assert integrity properties of the split results."""
    # Sizes add up
    assert lib.n_windows + qry.n_windows == full.n_windows

    # No overlapping indices
    lib_idx = set(map(tuple, lib.window_indices))
    qry_idx = set(map(tuple, qry.window_indices))
    assert lib_idx.isdisjoint(qry_idx)

    # All indices covered
    assert lib_idx.union(qry_idx) == set(map(tuple, full.window_indices))


@pytest.mark.parametrize("test_fraction", [0.2, 0.3, 0.5])

def test_split_fraction_stratified(simple_windowed_data: WindowedData, test_fraction):
    lib, qry = train_test_split_windows(simple_windowed_data, test_fraction=test_fraction, stratified=True, random_state=0)

    _check_integrity(simple_windowed_data, lib, qry)

    # Check approximate per-class proportions (allowing ±1 due to rounding)
    for cls in np.unique(simple_windowed_data.labels):
        total_cls = np.sum(simple_windowed_data.labels == cls)
        lib_cls = np.sum(lib.labels == cls)
        expected_lib = int(round(total_cls * (1 - test_fraction)))
        assert abs(lib_cls - expected_lib) <= 1


@pytest.mark.parametrize("library_per_class", [1, 5, 10])

def test_split_exact_library_per_class(simple_windowed_data: WindowedData, library_per_class):
    lib, qry = train_test_split_windows(simple_windowed_data, library_per_class=library_per_class, stratified=True, random_state=42)

    _check_integrity(simple_windowed_data, lib, qry)

    # Each class should have <= library_per_class windows in library
    for cls in np.unique(simple_windowed_data.labels):
        lib_cls = np.sum(lib.labels == cls)
        assert lib_cls == library_per_class


def test_split_non_stratified(simple_windowed_data):
    lib, qry = train_test_split_windows(simple_windowed_data, test_fraction=0.25, stratified=False, random_state=1)

    _check_integrity(simple_windowed_data, lib, qry)

    # When not stratified the class distribution can differ, but total size should match expected fraction
    expected_qry = int(round(simple_windowed_data.n_windows * 0.25))
    assert abs(qry.n_windows - expected_qry) <= 1 
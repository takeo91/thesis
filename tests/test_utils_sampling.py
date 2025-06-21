import pytest
from thesis.core.utils import get_dataset_specific_window_sizes


def test_default_rates():
    ws_opp = get_dataset_specific_window_sizes("opportunity", (4, 6))
    ws_pamap = get_dataset_specific_window_sizes("PAMAP2", (4, 6))
    assert ws_opp == [120, 180]
    assert ws_pamap == [400, 600]


def test_custom_rate():
    ws_custom = get_dataset_specific_window_sizes("custom", (2,), custom_rates={"custom": 50})
    assert ws_custom == [100]


def test_unknown_dataset():
    with pytest.raises(ValueError):
        get_dataset_specific_window_sizes("unknown", (4,)) 
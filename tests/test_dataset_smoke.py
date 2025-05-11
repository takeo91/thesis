import numpy as np
import pytest

from thesis.data.datasets import create_opportunity_dataset, create_pamap2_dataset

@pytest.mark.parametrize("loader,body_part", [
    (create_opportunity_dataset, "RKN^"),  # RKN^ is a verified body part with Accelerometer data
    (create_pamap2_dataset, "Hand"),       # Hand is a verified body part with Accelerometer data
])
def test_dataset_smoke(loader, body_part):
    """Test that datasets can be loaded and basic data can be retrieved."""
    dataset = loader()
    dataset.load_data()

    # Skip if dataset is empty
    if dataset.df.empty:
        pytest.skip(f"Dataset is empty - skipping smoke test")
    
    try:
        # Get accelerometer data for known working body part
        sensor_data = dataset.get_sensor_data(
            sensor_type="Accelerometer",
            body_part=body_part,
            axis="X"
        )
        
        # Verify we got some data
        assert len(sensor_data) > 0
        assert np.isfinite(sensor_data.values).all()
    except Exception as e:
        pytest.fail(f"Failed to retrieve Accelerometer data for {body_part}: {e}")
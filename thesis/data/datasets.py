"""
thesis.data.datasets
====================

Dataset processors for health and activity monitoring datasets.

This module provides classes and functions for loading, processing, and analyzing
sensor data from health and activity monitoring datasets. It supports both the
Opportunity and PAMAP2 datasets with a consistent interface.

Datasets:
---------

OPPORTUNITY Activity Recognition dataset:
    • Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00226/
    • Features: Body-worn sensors (IMUs, accelerometers), object sensors, ambient sensors
    • Activities: Daily activities with multi-level annotations (locomotion, gestures, etc.)

PAMAP2 Physical Activity Monitoring dataset:
    • Download: https://archive.ics.uci.edu/ml/machine-learning-databases/00231/
    • Features: IMUs and heart rate monitor
    • Activities: 18 different physical activities (walking, cycling, etc.)

Usage:
------
    # Create an Opportunity dataset processor
    opportunity_dataset = OpportunityDataset("path/to/S1-ADL1.dat")
    opportunity_dataset.load_data()
    
    # Create a PAMAP2 dataset processor
    pamap2_dataset = PAMAP2Dataset("path/to/subject101.dat")
    pamap2_dataset.load_data()
    
    # Use the consistent interface to retrieve sensor data
    acc_data = opportunity_dataset.get_sensor_data(
        sensor_type="Accelerometer", 
        body_part="LowerArm", 
        axis="X"
    )

Directory Structure:
-------------------
Un-zip the dataset files like so:

    $THESIS_DATA/
    ├─ OpportunityUCIDataset/…
    └─ PAMAP2_Dataset/…

`THESIS_DATA` defaults to ~/Data. Override with an env-var.

Simple API:
----------
    load_opportunity(split="train", n_samples=None) → (X, y)
    load_pamap2(split="train", n_samples=None)     → (X, y)

If the raw data is missing *and* you pass `n_samples`, the loaders return a
repeatable synthetic slice so CI tests stay fast and offline-safe.
"""
from __future__ import annotations
import os
import pathlib
import abc
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis.core.utils import extract_column_names, extract_metadata, extract_labels

# --------------------------------------------------------------------
DATA_ROOT = pathlib.Path(os.environ.get("THESIS_DATA", pathlib.Path.home() / "personal/thesis/Data"))
OPP_ROOT = DATA_ROOT / "OpportunityUCIDataset"
PAMAP_ROOT = DATA_ROOT / "PAMAP2_Dataset"

__all__ = [
    "SensorDataset", 
    "OpportunityDataset", 
    "PAMAP2Dataset",
    "create_opportunity_dataset",
    "create_pamap2_dataset"
]

_rng = np.random.default_rng(0)

def _synthetic(rows: int, feats: int, labels: int):
    X = _rng.standard_normal((rows, feats)).astype("float32")
    y = _rng.integers(0, labels, size=(rows,), dtype="int16")
    return X, y

def _missing(name: str, where: pathlib.Path):
    raise FileNotFoundError(
        f"{name} dataset not found at {where}\n"
        "Download & unzip manually — see datasets.py doc-string."
    )

# -------------------- Abstract Base Class --------------------------
class SensorDataset(abc.ABC):
    """
    Abstract base class defining the interface for sensor datasets.
    
    This class provides a common interface for working with different sensor
    datasets used for activity recognition and health monitoring.
    """
    
    def __init__(self, data_file: str):
        """
        Initialize a sensor dataset.
        
        Args:
            data_file: Path to the data file.
        """
        self.data_file = data_file
        self.df = None
        self.metadata_df = None
        self.time_column = None
    
    @abc.abstractmethod
    def load_data(self) -> None:
        """
        Load data from the specified file and structure appropriately.
        
        This method should:
        1. Read the raw data file
        2. Apply proper column names/structure
        3. Handle missing values
        4. Convert data types as needed
        5. Create any metadata needed for data access
        """
        pass
    
    @abc.abstractmethod
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded dataset.
        
        Returns:
            Dictionary with dataset summary information.
        """
        pass
    
    @abc.abstractmethod
    def get_sensor_data(
        self,
        sensor_type: str,
        body_part: str,
        measurement_type: Optional[str] = None,
        axis: Optional[str] = None,
        activity_filter: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Get specific sensor data with optional activity filtering.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "Arm").
            measurement_type: Optional measurement type (e.g., "acceleration").
            axis: Optional axis specifier (e.g., "X").
            activity_filter: Optional dictionary to filter by activity.
            
        Returns:
            Series containing the selected sensor data.
        """
        pass
    
    @abc.abstractmethod
    def get_time_series(self, activity_filter: Optional[Dict[str, str]] = None) -> pd.Series:
        """
        Get the time series data with optional activity filtering.
        
        Args:
            activity_filter: Optional dictionary to filter by activity.
            
        Returns:
            Series containing the time values.
        """
        pass
    
    def plot_sensor_data(
        self,
        sensor_type: str,
        body_part: str,
        measurement_type: Optional[str] = None,
        axis: Optional[str] = None,
        activity_filter: Optional[Dict[str, str]] = None,
        show_plot: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Plot sensor data with optional activity filtering.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "Arm").
            measurement_type: Optional measurement type (e.g., "acceleration").
            axis: Optional axis specifier (e.g., "X").
            activity_filter: Optional dictionary to filter by activity.
            show_plot: Whether to display the plot.
            save_path: Optional path to save the plot.
            figsize: Figure size as (width, height) in inches.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get the data
        sensor_data = self.get_sensor_data(
            sensor_type, body_part, measurement_type, axis, activity_filter
        )
        time_series = self.get_time_series(activity_filter)
        
        # Create title and y-axis label
        activity_str = ""
        if activity_filter:
            activity_str = ", ".join([f"{k}: {v}" for k, v in activity_filter.items()])
            activity_str = f" ({activity_str})"
        
        parts = [sensor_type, body_part]
        if measurement_type:
            parts.append(measurement_type)
        if axis:
            parts.append(axis)
        
        title = " - ".join(parts) + activity_str
        
        # Create the plot
        plt.figure(figsize=figsize)
        plt.plot(time_series, sensor_data)
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel(self._get_unit_label(sensor_type, measurement_type))
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _get_unit_label(self, sensor_type: str, measurement_type: Optional[str] = None) -> str:
        """
        Get a formatted label with units for the y-axis.
        
        Args:
            sensor_type: Type of sensor.
            measurement_type: Optional measurement type.
            
        Returns:
            Formatted label string.
        """
        unit = self._get_unit(sensor_type, measurement_type)
        label = sensor_type
        if measurement_type:
            label = f"{measurement_type} ({unit})" if unit else measurement_type
        return label
    
    @abc.abstractmethod
    def _get_unit(self, sensor_type: str, measurement_type: Optional[str] = None) -> str:
        """
        Get the appropriate unit based on sensor type and measurement.
        
        Args:
            sensor_type: Type of sensor.
            measurement_type: Optional measurement type.
            
        Returns:
            String representing the unit.
        """
        pass
    
    @abc.abstractmethod
    def get_available_activities(self) -> Dict[str, List[str]]:
        """
        Get available activity types and their values.
        
        Returns:
            Dictionary with activity categories as keys and lists of possible 
            values as values.
        """
        pass
    
    @abc.abstractmethod
    def get_available_sensors(self) -> Dict[str, List[str]]:
        """
        Get available sensor types, body parts, and other categorization.
        
        Returns:
            Dictionary with sensor information.
        """
        pass

# -------------------- Opportunity -----------------------------------
def create_opportunity_dataset(
    data_file: Optional[str] = None,
    column_names_file: Optional[str] = None,
    label_legend_file: Optional[str] = None,
) -> OpportunityDataset:
    """
    Create and initialize an OpportunityDataset instance.
    
    Args:
        data_file: Path to the data file. If None, uses a default path.
        column_names_file: Path to the column names file. If None, determined from data_file.
        label_legend_file: Path to the label legend file. If None, determined from data_file.
    
    Returns:
        An initialized OpportunityDataset instance.
    """
    if not OPP_ROOT.exists():
        _missing("Opportunity", OPP_ROOT)
    
    if data_file is None:
        data_file = str(next(OPP_ROOT.glob("**/dataset/S1-ADL1.dat"), None))
        if data_file is None:
            raise RuntimeError(f"Default data file S1-ADL1.dat not found in {OPP_ROOT}")
    
    dataset = OpportunityDataset(data_file, column_names_file, label_legend_file)
    dataset.load_data()
    return dataset

# -------------------- PAMAP2 ----------------------------------------
def create_pamap2_dataset(data_file: Optional[str] = None) -> PAMAP2Dataset:
    """
    Create and initialize a PAMAP2Dataset instance.
    
    Args:
        data_file: Path to the data file. If None, uses a default path.
    
    Returns:
        An initialized PAMAP2Dataset instance.
    """
    if not PAMAP_ROOT.exists():
        _missing("PAMAP2", PAMAP_ROOT)
    
    if data_file is None:
        data_file = str(next(PAMAP_ROOT.glob("**/Protocol/subject101.dat"), None))
        if data_file is None:
            raise RuntimeError(f"Default data file subject101.dat not found in {PAMAP_ROOT}")
    
    dataset = PAMAP2Dataset(data_file)
    dataset.load_data()
    return dataset

# -------------------- Opportunity Dataset Implementation ------------------
class OpportunityDataset(SensorDataset):
    """
    Class for processing and analyzing sensor data from the Opportunity dataset.
    
    This class implements the SensorDataset interface specifically for the 
    Opportunity activity recognition dataset.
    """
    
    def __init__(
        self,
        data_file: str,
        column_names_file: Optional[str] = None,
        label_legend_file: Optional[str] = None,
    ):
        """
        Initialize the OpportunityDataset processor.
        
        Args:
            data_file: Path to the sensor data file.
            column_names_file: Path to the column definitions file.
                If None, uses default path relative to data_file.
            label_legend_file: Path to the label legend file.
                If None, uses default path relative to data_file.
        """
        super().__init__(data_file)
        
        # Determine default paths if not provided
        if column_names_file is None:
            data_dir = os.path.dirname(data_file)
            column_names_file = os.path.join(data_dir, "column_names.txt")
        
        if label_legend_file is None:
            data_dir = os.path.dirname(data_file)
            label_legend_file = os.path.join(data_dir, "label_legend.txt")
        
        self.column_names_file = column_names_file
        self.label_legend_file = label_legend_file
        self.time_column = "MILLISEC"
    
    def load_data(self) -> None:
        """
        Load data from the specified files and structure with proper column names.
        """
        # Load the data
        self.df = pd.read_csv(self.data_file, sep=r"\s+", header=None, na_values="NaN")
        
        # Read the column names
        column_names = extract_column_names(self.column_names_file)
        
        # Check if the number of column names matches the DataFrame columns
        if len(column_names) == self.df.shape[1]:
            self.df.columns = column_names
        else:
            print(
                f"Warning: Column names count ({len(column_names)}) does not match "
                f"data columns ({self.df.shape[1]})."
            )
        
        # Handle missing values with forward fill
        self.df.fillna(method="ffill", inplace=True)
        
        # Convert time column to integer
        self.df[self.time_column] = self.df[self.time_column].astype(int)
        
        # Convert sensor data to float
        sensor_columns = self.df.columns[1:243]  # Adjust range as needed
        self.df[sensor_columns] = self.df[sensor_columns].astype(float)
        
        # Process label columns
        self._process_labels()
        
        # Create metadata and multi-index
        self._create_metadata()
    
    def _process_labels(self) -> None:
        """
        Process label columns by converting them to proper types and mapping labels.
        """
        # Define label columns
        label_columns = [
            "Locomotion",
            "HL_Activity",
            "LL_Left_Arm",
            "LL_Left_Arm_Object",
            "LL_Right_Arm",
            "LL_Right_Arm_Object",
            "ML_Both_Arms",
        ]
        
        # Convert labels to integers
        self.df[label_columns] = self.df[label_columns].astype(int)
        
        # Map numerical labels to descriptions
        label_mappings = extract_labels(self.label_legend_file)
        
        for label_col in label_columns:
            if label_col in label_mappings:
                mapping = label_mappings[label_col]
                self.df[label_col] = self.df[label_col].map(mapping)
            else:
                if label_col == "ML_Both_Arms":
                    self.df["Combined_Index"] = (
                        self.df["LL_Right_Arm"].astype(str)
                        + "_"
                        + self.df["LL_Right_Arm_Object"].astype(str)
                    )
                    self.df[label_col] = self.df["Combined_Index"].map(label_mappings[label_col])
                    self.df.drop("Combined_Index", axis=1, inplace=True)
        
        # Fill any remaining NaN labels with "Unknown"
        self.df[label_columns] = self.df[label_columns].fillna("Unknown")
    
    def _create_metadata(self) -> None:
        """
        Create metadata for each column and structure the DataFrame with multi-index.
        """
        # Initialize a list to hold metadata dictionaries
        metadata_list = []
        label_columns = [
            "Locomotion",
            "HL_Activity",
            "LL_Left_Arm",
            "LL_Left_Arm_Object",
            "LL_Right_Arm",
            "LL_Right_Arm_Object",
            "ML_Both_Arms",
        ]
        
        for col_name in self.df.columns:
            meta = extract_metadata(col_name, label_columns)
            meta["original_name"] = col_name
            metadata_list.append(meta)
        
        # Create a DataFrame from the metadata
        self.metadata_df = pd.DataFrame(metadata_list)
        self.metadata_df.set_index("original_name", inplace=True)
        
        # Reindex metadata_df to match df.columns
        self.metadata_df = self.metadata_df.reindex(self.df.columns)
        
        # Create arrays for MultiIndex levels
        arrays = [
            self.metadata_df["sensor_type"].values,
            self.metadata_df["body_part"].values,
            self.metadata_df["measurement_type"].values,
            self.metadata_df["axis"].values,
        ]
        
        # Replace None or NaN with 'Unknown'
        arrays = [[a if pd.notnull(a) else "Unknown" for a in array] for array in arrays]
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_arrays(
            arrays, names=["SensorType", "BodyPart", "MeasurementType", "Axis"]
        )
        
        # Assign MultiIndex to df columns
        self.df.columns = multi_index
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded dataset.
        
        Returns:
            Dictionary with dataset summary information.
        """
        if self.df is None:
            return {"error": "Data not loaded. Call load_data() first."}
        
        idx = pd.IndexSlice
        time_series = self.df.loc[:, idx["Time", "N/A", "Time", "N/A"]]
        
        return {
            "dataset": "Opportunity",
            "file": self.data_file,
            "rows": len(self.df),
            "time_range_seconds": (time_series.max() - time_series.min()) / 1000,
            "sensor_types": list(self.df.columns.get_level_values("SensorType").unique()),
            "body_parts": list(self.df.columns.get_level_values("BodyPart").unique()),
            "measurement_types": list(self.df.columns.get_level_values("MeasurementType").unique()),
            "label_activities": {
                "locomotion": list(self.df.loc[:, idx["Label", "Locomotion", "Label", "N/A"]].unique()),
                "high_level": list(self.df.loc[:, idx["Label", "HL_Activity", "Label", "N/A"]].unique()),
            }
        }
    
    def get_sensor_data(
        self,
        sensor_type: str,
        body_part: str,
        measurement_type: Optional[str] = None,
        axis: Optional[str] = None,
        activity_filter: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Get specific sensor data with optional activity filtering.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "LowerArm").
            measurement_type: Optional measurement type (e.g., "acc").
            axis: Optional axis specifier (e.g., "X").
            activity_filter: Optional dictionary to filter by activity,
                            e.g., {"Locomotion": "Walk"}.
        
        Returns:
            Series containing the selected sensor data.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        idx = pd.IndexSlice
        
        # Create the mask for filtered activities if needed
        mask = pd.Series(True, index=self.df.index)
        if activity_filter:
            for activity_col, activity_value in activity_filter.items():
                activity_mask = self.df.loc[:, idx["Label", activity_col, "Label", "N/A"]] == activity_value
                mask = mask & activity_mask.squeeze()
        
        # Create the column selection based on available parameters
        if measurement_type is None:
            if axis is None:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, :, :]]
            else:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, :, axis]]
        else:
            if axis is None:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, :]]
            else:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, axis]]
        
        return selected_data.squeeze()
    
    def get_time_series(self, activity_filter: Optional[Dict[str, str]] = None) -> pd.Series:
        """
        Get the time series data with optional activity filtering.
        
        Args:
            activity_filter: Optional dictionary to filter by activity,
                            e.g., {"Locomotion": "Walk"}.
        
        Returns:
            Series containing the time values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        idx = pd.IndexSlice
        
        # Create the mask for filtered activities if needed
        mask = pd.Series(True, index=self.df.index)
        if activity_filter:
            for activity_col, activity_value in activity_filter.items():
                activity_mask = self.df.loc[:, idx["Label", activity_col, "Label", "N/A"]] == activity_value
                mask = mask & activity_mask.squeeze()
        
        time_series = self.df.loc[mask, idx["Time", "N/A", "Time", "N/A"]]
        return time_series.squeeze()
    
    def _get_unit(self, sensor_type: str, measurement_type: Optional[str] = None) -> str:
        """
        Get the appropriate unit based on sensor type and measurement.
        
        Args:
            sensor_type: Type of sensor.
            measurement_type: Optional measurement type.
            
        Returns:
            String representing the unit.
        """
        if not measurement_type:
            return ""
            
        measurement_type_lower = measurement_type.lower()
        if measurement_type_lower in ["acc", "acceleration"]:
            return "m/s²"
        elif measurement_type_lower in ["gyroscope", "gyro"]:
            return "rad/s"
        elif measurement_type_lower in ["magnetometer", "magnet"]:
            return "μT"
        elif measurement_type_lower == "time":
            return "ms"
        elif measurement_type_lower == "quaternion":
            return "quat"
        else:
            return ""
    
    def get_available_activities(self) -> Dict[str, List[str]]:
        """
        Get available activity types and their values.
        
        Returns:
            Dictionary with activity categories as keys and lists of possible 
            values as values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        idx = pd.IndexSlice
        
        return {
            "Locomotion": list(self.df.loc[:, idx["Label", "Locomotion", "Label", "N/A"]].unique()),
            "HL_Activity": list(self.df.loc[:, idx["Label", "HL_Activity", "Label", "N/A"]].unique()),
            "LL_Left_Arm": list(self.df.loc[:, idx["Label", "LL_Left_Arm", "Label", "N/A"]].unique()),
            "LL_Right_Arm": list(self.df.loc[:, idx["Label", "LL_Right_Arm", "Label", "N/A"]].unique()),
            "ML_Both_Arms": list(self.df.loc[:, idx["Label", "ML_Both_Arms", "Label", "N/A"]].unique()),
        }
    
    def get_available_sensors(self) -> Dict[str, List[str]]:
        """
        Get available sensor types, body parts, and other categorization.
        
        Returns:
            Dictionary with sensor information.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            "sensor_types": list(self.df.columns.get_level_values("SensorType").unique()),
            "body_parts": list(self.df.columns.get_level_values("BodyPart").unique()),
            "measurement_types": list(self.df.columns.get_level_values("MeasurementType").unique()),
            "axes": list(self.df.columns.get_level_values("Axis").unique()),
        }

# -------------------- PAMAP2 Dataset Implementation ------------------
class PAMAP2Dataset(SensorDataset):
    """
    Class for processing and analyzing sensor data from the PAMAP2 dataset.
    
    This class implements the SensorDataset interface specifically for the 
    PAMAP2 physical activity monitoring dataset.
    """
    
    def _setup_column_definitions(self) -> Dict[str, Union[int, slice]]:
        """Setup column definitions for PAMAP2 dataset."""
        return {
            "timestamp": 0,
            "activity_id": 1,
            "heart_rate": 2,
            "imu_hand": slice(3, 20),
            "imu_chest": slice(20, 37),
            "imu_ankle": slice(37, 54),
        }
    
    def _setup_activity_mappings(self) -> Dict[int, str]:
        """Setup activity ID to name mappings for PAMAP2."""
        return {
            0: "other",
            1: "lying",
            2: "sitting",
            3: "standing",
            4: "walking",
            5: "running",
            6: "cycling",
            7: "Nordic_walking",
            9: "watching_TV",
            10: "computer_work",
            11: "car_driving",
            12: "ascending_stairs",
            13: "descending_stairs",
            16: "vacuum_cleaning",
            17: "ironing",
            18: "folding_laundry",
            19: "house_cleaning",
            20: "playing_soccer",
            24: "rope_jumping",
        }
    
    def _setup_imu_columns(self) -> List[str]:
        """Setup IMU column names for PAMAP2."""
        return [
            "temperature",
            "acc_16g_x", "acc_16g_y", "acc_16g_z",
            "acc_6g_x", "acc_6g_y", "acc_6g_z",
            "gyro_x", "gyro_y", "gyro_z",
            "mag_x", "mag_y", "mag_z",
            "orientation_1", "orientation_2", "orientation_3", "orientation_4"
        ]
    
    def _create_sensor_mappings_for_location(self, location: str) -> Dict[str, Dict[str, str]]:
        """Create sensor mappings for a specific body location."""
        mappings = {}
        
        # Accelerometer mappings
        for axis in ["x", "y", "z"]:
            mappings[f"acc_16g_{axis}_{location}"] = {
                "SensorType": "Accelerometer", 
                "BodyPart": location.capitalize(), 
                "MeasurementType": "acc", 
                "Axis": axis.upper()
            }
            mappings[f"acc_6g_{axis}_{location}"] = {
                "SensorType": "Accelerometer", 
                "BodyPart": location.capitalize(), 
                "MeasurementType": "acc_low", 
                "Axis": axis.upper()
            }
        
        # Gyroscope mappings
        for axis in ["x", "y", "z"]:
            mappings[f"gyro_{axis}_{location}"] = {
                "SensorType": "Gyroscope", 
                "BodyPart": location.capitalize(), 
                "MeasurementType": "gyro", 
                "Axis": axis.upper()
            }
        
        # Magnetometer mappings
        for axis in ["x", "y", "z"]:
            mappings[f"mag_{axis}_{location}"] = {
                "SensorType": "Magnetometer", 
                "BodyPart": location.capitalize(), 
                "MeasurementType": "mag", 
                "Axis": axis.upper()
            }
        
        # Temperature mapping
        mappings[f"temperature_{location}"] = {
            "SensorType": "Temperature", 
            "BodyPart": location.capitalize(), 
            "MeasurementType": "temp", 
            "Axis": "N/A"
        }
        
        return mappings
    
    def _setup_sensor_mappings(self) -> Dict[str, Dict[str, str]]:
        """Setup complete sensor mappings for PAMAP2 dataset."""
        mappings = {}
        
        # Add mappings for each body location
        for location in ["hand", "chest", "ankle"]:
            mappings.update(self._create_sensor_mappings_for_location(location))
        
        # Add non-IMU sensors
        mappings.update({
            "heart_rate": {"SensorType": "HeartRate", "BodyPart": "Chest", "MeasurementType": "bpm", "Axis": "N/A"},
            "timestamp": {"SensorType": "Time", "BodyPart": "N/A", "MeasurementType": "Time", "Axis": "N/A"},
            "activity_id": {"SensorType": "Label", "BodyPart": "Activity", "MeasurementType": "Label", "Axis": "N/A"},
        })
        
        return mappings

    def __init__(self, data_file: str):
        """
        Initialize the PAMAP2Dataset processor.
        
        Args:
            data_file: Path to the sensor data file (e.g., "subject101.dat").
        """
        super().__init__(data_file)
        self.time_column = "timestamp"
        
        # Setup configuration using helper methods
        self.column_definitions = self._setup_column_definitions()
        self.activity_map = self._setup_activity_mappings()
        self.imu_columns = self._setup_imu_columns()
        self.sensor_mapping = self._setup_sensor_mappings()
        
    def load_data(self) -> None:
        """
        Load data from the PAMAP2 data file and structure it.
        """
        # Load the raw data
        self.df = pd.read_csv(self.data_file, sep=" ", header=None, na_values="nan")
        
        # Create more descriptive temporary column names
        temp_columns = ["timestamp", "activity_id", "heart_rate"]
        
        # Add IMU sensor columns
        for location in ["hand", "chest", "ankle"]:
            for col in self.imu_columns:
                temp_columns.append(f"{col}_{location}")
        
        # Set temporary column names
        self.df.columns = temp_columns
        
        # Fill NaN values (missing values)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0
        
        # Map activity IDs to names
        self.df["activity_name"] = self.df["activity_id"].map(self.activity_map)
        
        # Create a more interpretable structure with MultiIndex
        self._create_multi_index()
        
    def _create_multi_index(self) -> None:
        """
        Create a MultiIndex for the DataFrame columns for easier access.
        """
        # Create lists for MultiIndex levels
        sensor_types = []
        body_parts = []
        measurement_types = []
        axes = []
        
        # Assign MultiIndex based on the sensor_mapping dictionary
        for col in self.df.columns:
            if col in self.sensor_mapping:
                mapping = self.sensor_mapping[col]
                sensor_types.append(mapping["SensorType"])
                body_parts.append(mapping["BodyPart"])
                measurement_types.append(mapping["MeasurementType"])
                axes.append(mapping["Axis"])
            elif col == "activity_name":
                sensor_types.append("Label")
                body_parts.append("Activity")
                measurement_types.append("Name")
                axes.append("N/A")
            else:
                sensor_types.append("Unknown")
                body_parts.append("Unknown")
                measurement_types.append("Unknown")
                axes.append("Unknown")
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_arrays(
            [sensor_types, body_parts, measurement_types, axes],
            names=["SensorType", "BodyPart", "MeasurementType", "Axis"]
        )
        
        # Assign MultiIndex to df columns
        self.df.columns = multi_index
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded dataset.
        
        Returns:
            Dictionary with dataset summary information.
        """
        if self.df is None:
            return {"error": "Data not loaded. Call load_data() first."}
        
        idx = pd.IndexSlice
        time_series = self.df.loc[:, idx["Time", "N/A", "Time", "N/A"]]
        activities = self.df.loc[:, idx["Label", "Activity", "Name", "N/A"]]
        
        return {
            "dataset": "PAMAP2",
            "file": self.data_file,
            "rows": len(self.df),
            "duration_seconds": (time_series.max() - time_series.min()),
            "sensor_types": list(self.df.columns.get_level_values("SensorType").unique()),
            "body_parts": list(self.df.columns.get_level_values("BodyPart").unique()),
            "measurement_types": list(self.df.columns.get_level_values("MeasurementType").unique()),
            "activities": list(activities.unique()),
        }
    
    def get_sensor_data(
        self,
        sensor_type: str,
        body_part: str,
        measurement_type: Optional[str] = None,
        axis: Optional[str] = None,
        activity_filter: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Get specific sensor data with optional activity filtering.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "Hand").
            measurement_type: Optional measurement type (e.g., "acc").
            axis: Optional axis specifier (e.g., "X").
            activity_filter: Optional dictionary to filter by activity,
                            e.g., {"activity": "walking"}.
        
        Returns:
            Series containing the selected sensor data.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        idx = pd.IndexSlice
        
        # Create the mask for filtered activities if needed
        mask = pd.Series(True, index=self.df.index)
        if activity_filter:
            for activity_key, activity_value in activity_filter.items():
                if activity_key == "activity":
                    activity_mask = self.df.loc[:, idx["Label", "Activity", "Name", "N/A"]] == activity_value
                    mask = mask & activity_mask.squeeze()
        
        # Create the column selection based on available parameters
        if measurement_type is None:
            if axis is None:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, :, :]]
            else:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, :, axis]]
        else:
            if axis is None:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, :]]
            else:
                selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, axis]]
        
        return selected_data.squeeze()
    
    def get_time_series(self, activity_filter: Optional[Dict[str, str]] = None) -> pd.Series:
        """
        Get the time series data with optional activity filtering.
        
        Args:
            activity_filter: Optional dictionary to filter by activity,
                            e.g., {"activity": "walking"}.
        
        Returns:
            Series containing the time values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        idx = pd.IndexSlice
        
        # Create the mask for filtered activities if needed
        mask = pd.Series(True, index=self.df.index)
        if activity_filter:
            for activity_key, activity_value in activity_filter.items():
                if activity_key == "activity":
                    activity_mask = self.df.loc[:, idx["Label", "Activity", "Name", "N/A"]] == activity_value
                    mask = mask & activity_mask.squeeze()
        
        time_series = self.df.loc[mask, idx["Time", "N/A", "Time", "N/A"]]
        return time_series.squeeze()
    
    def _get_unit(self, sensor_type: str, measurement_type: Optional[str] = None) -> str:
        """
        Get the appropriate unit based on sensor type and measurement.
        
        Args:
            sensor_type: Type of sensor.
            measurement_type: Optional measurement type.
            
        Returns:
            String representing the unit.
        """
        if not measurement_type:
            return ""
            
        sensor_type_lower = sensor_type.lower()
        measurement_type_lower = measurement_type.lower()
        
        if sensor_type_lower == "accelerometer":
            return "m/s²"
        elif sensor_type_lower == "gyroscope":
            return "rad/s"
        elif sensor_type_lower == "magnetometer":
            return "μT"
        elif sensor_type_lower == "heartrate":
            return "bpm"
        elif sensor_type_lower == "temperature":
            return "°C"
        elif measurement_type_lower == "time":
            return "s"
        else:
            return ""
    
    def get_available_activities(self) -> Dict[str, List[str]]:
        """
        Get available activity types and their values.
        
        Returns:
            Dictionary with activity categories as keys and lists of possible 
            values as values.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        idx = pd.IndexSlice
        activities = self.df.loc[:, idx["Label", "Activity", "Name", "N/A"]].unique()
        
        return {
            "activity": list(activities),
        }
    
    def get_available_sensors(self) -> Dict[str, List[str]]:
        """
        Get available sensor types, body parts, and other categorization.
        
        Returns:
            Dictionary with sensor information.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            "sensor_types": list(self.df.columns.get_level_values("SensorType").unique()),
            "body_parts": list(self.df.columns.get_level_values("BodyPart").unique()),
            "measurement_types": list(self.df.columns.get_level_values("MeasurementType").unique()),
            "axes": list(self.df.columns.get_level_values("Axis").unique()),
        }
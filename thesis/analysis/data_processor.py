"""
Data processing module for sensor data.

This module provides functions for loading, processing, and structuring
sensor data from the Opportunity dataset for analysis.
"""

from __future__ import annotations
from typing import Dict, List, Union, Optional

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from thesis.core.utils import extract_column_names, extract_metadata, extract_labels


class SensorDataProcessor:
    """
    A class for processing and analyzing sensor data from the Opportunity dataset.
    """

    def __init__(
        self,
        data_file: str,
        column_names_file: str,
        label_legend_file: str,
    ):
        """
        Initialize the SensorDataProcessor with dataset paths.
        
        Args:
            data_file: Path to the sensor data file.
            column_names_file: Path to the column definitions file.
            label_legend_file: Path to the label legend file.
        """
        self.data_file = data_file
        self.column_names_file = column_names_file
        self.label_legend_file = label_legend_file
        self.df = None
        self.metadata_df = None
    
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
        self.df["MILLISEC"] = self.df["MILLISEC"].astype(int)
        
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
    
    def get_data_summary(self) -> Dict[str, Union[int, float, List[str]]]:
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
        measurement_type: str,
        axis: Optional[str] = None,
        activity_filter: Optional[Dict[str, str]] = None
    ) -> pd.Series:
        """
        Get specific sensor data with optional activity filtering.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "BACK").
            measurement_type: Measurement type (e.g., "acc").
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
        
        # Create the column selection
        if axis:
            selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, axis]]
        else:
            selected_data = self.df.loc[mask, idx[sensor_type, body_part, measurement_type, :]]
        
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
        
    def plot_sensor_data(
        self,
        sensor_type: str,
        body_part: str,
        measurement_type: str,
        axis: str,
        activity_filter: Optional[Dict[str, str]] = None,
        show_plot: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot sensor data for visualization.
        
        Args:
            sensor_type: Type of sensor (e.g., "Accelerometer").
            body_part: Body part (e.g., "BACK").
            measurement_type: Measurement type (e.g., "acc").
            axis: Axis specifier (e.g., "X").
            activity_filter: Optional dictionary to filter by activity,
                            e.g., {"Locomotion": "Walk"}.
            show_plot: Whether to display the plot. Defaults to True.
            save_path: Optional path to save the plot.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get sensor data and time series
        sensor_data = self.get_sensor_data(
            sensor_type, body_part, measurement_type, axis, activity_filter
        )
        time_series = self.get_time_series(activity_filter)
        
        # Create a descriptive title
        activity_str = ""
        if activity_filter:
            activity_str = " during " + ", ".join(
                [f"{k}: {v}" for k, v in activity_filter.items()]
            )
        
        title = f"{body_part} {sensor_type} {measurement_type}{axis}{activity_str}"
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(time_series, sensor_data)
        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel(f"{measurement_type} ({self._get_unit(sensor_type, measurement_type)})")
        plt.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _get_unit(self, sensor_type: str, measurement_type: str) -> str:
        """
        Get the appropriate unit for a sensor/measurement type.
        
        Args:
            sensor_type: Type of sensor.
            measurement_type: Type of measurement.
            
        Returns:
            String representing the unit.
        """
        # Simple mapping of common units
        if sensor_type == "Accelerometer":
            return "milli-g"
        elif sensor_type == "Gyroscope":
            return "deg/s"
        elif measurement_type == "magnetic":
            return "uT"
        else:
            return "N/A"


def process_opportunity_dataset(
    data_file: str = "Data/OpportunityUCIDataset/dataset/S1-ADL1.dat",
    column_names_file: str = "Data/OpportunityUCIDataset/dataset/column_names.txt",
    label_legend_file: str = "Data/OpportunityUCIDataset/dataset/label_legend.txt",
) -> SensorDataProcessor:
    """
    Process the Opportunity dataset and return a data processor instance.
    
    Args:
        data_file: Path to the sensor data file.
        column_names_file: Path to the column definitions file.
        label_legend_file: Path to the label legend file.
        
    Returns:
        Initialized and loaded SensorDataProcessor instance.
    """
    processor = SensorDataProcessor(data_file, column_names_file, label_legend_file)
    processor.load_data()
    return processor 
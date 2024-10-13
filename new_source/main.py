import pandas as pd
import numpy as np
from utils import extract_column_names, extract_metadata, extract_labels
import matplotlib.pyplot as plt

# Path to your data file
data_file = 'Data/OpportunityUCIDataset/dataset/S1-ADL1.dat'
column_names_file = 'Data/OpportunityUCIDataset/dataset/column_names.txt'
label_legend_file = "Data/OpportunityUCIDataset/dataset/label_legend.txt"

# Load the data
df = pd.read_csv(data_file, sep='\s+', header=None, na_values='NaN')

# Read the column names
column_names = extract_column_names(column_names_file)

# Check if the number of column names matches the DataFrame columns
if len(column_names) == df.shape[1]:
    df.columns = column_names
else:
    print("The number of column names does not match the data columns.")
    # Optionally, handle the mismatch here

# Option to fill missing values (e.g., with forward fill)
df.fillna(method='ffill', inplace=True)

# # Or drop rows with missing values
# df.dropna(inplace=True)

# Convert time column to integer (if necessary)
df['MILLISEC'] = df['MILLISEC'].astype(int)

# Convert sensor data to float
sensor_columns = df.columns[1:243]
df[sensor_columns] = df[sensor_columns].astype(float)

# Convert labels to integers
label_columns = ['Locomotion', 'HL_Activity', 'LL_Left_Arm', 'LL_Left_Arm_Object',
                 'LL_Right_Arm', 'LL_Right_Arm_Object', 'ML_Both_Arms']

df[label_columns] = df[label_columns].astype(int)

# Handle label columns

# Map numerical labels to descriptions
label_mappings = extract_labels(label_legend_file)

for label_col in label_columns:
    if label_col in label_mappings:
        mapping = label_mappings[label_col]
        df[label_col] = df[label_col].map(mapping)
    else:
        if label_col == 'ML_Both_Arms':
            df['Combined_Index'] = df['LL_Right_Arm'].astype(str) + '_' + df['LL_Right_Arm_Object'].astype(str)
            df[label_col] = df['Combined_Index'].map(label_mappings[label_col])
            df.drop('Combined_Index', axis=1, inplace=True)

df[label_columns] = df[label_columns].fillna('Unknown')

# Initialize a list to hold metadata dictionaries
metadata_list = []

for col_name in column_names:
    meta = extract_metadata(col_name, label_columns)
    meta['original_name'] = col_name
    metadata_list.append(meta)


# Create a DataFrame from the metadata
metadata_df = pd.DataFrame(metadata_list)
metadata_df.set_index('original_name', inplace=True)

# Reindex metadata_df to match df.columns
metadata_df = metadata_df.reindex(df.columns)

# Create arrays for MultiIndex levels
arrays = [
    metadata_df['sensor_type'].values,
    metadata_df['body_part'].values,
    metadata_df['measurement_type'].values,
    metadata_df['axis'].values
]

# Replace None or NaN with 'Unknown'
arrays = [
    [a if pd.notnull(a) else 'Unknown' for a in array]
    for array in arrays
]

# Create MultiIndex
multi_index = pd.MultiIndex.from_arrays(arrays, names=['SensorType', 'BodyPart', 'MeasurementType', 'Axis'])

# Assign MultiIndex to df columns
df.columns = multi_index

# Now you can proceed with your analysis
print(df.head())
print(df.columns)
print(df.columns.levels)

# Access labels
labels_df = df['Label']
print(labels_df.head())

# Access time column
time_series = df['Time', 'N/A', 'Time', 'N/A']
print(time_series.head())

# Access sensor data
accelerometer_df = df['Accelerometer']
print(accelerometer_df.head())

idx = pd.IndexSlice

# Select the 'Locomotion' label column, accounting for all values in the 'OriginalName' level
locomotion_column = df.loc[:, idx['Label', 'Locomotion', 'Label', 'N/A']]

# Since there's only one column, convert it to a Series
locomotion_series = locomotion_column.squeeze()

# Create the mask for rows where Locomotion is 'Walk'
walk_mask = locomotion_series == 'Walk'

# Now select the 'Back accX' data
back_accX_column = df.loc[:, idx['Accelerometer', 'BACK', 'acc', 'X']]
back_accX = back_accX_column.squeeze()

# Similarly, select the time column
time_column = df.loc[:, idx['Time', 'N/A', 'Time', 'N/A']]
time_series = time_column.squeeze()

# Apply the mask to get data during 'Walk'
back_accX_walk = back_accX[walk_mask]
time_series_walk = time_series[walk_mask]

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(time_series_walk, back_accX_walk)
# plt.title('Back Accelerometer accX During Walking')
# plt.xlabel('Time (ms)')
# plt.ylabel('Acceleration (milli g)')
# plt.show()

print(df['Time'].head())

print(df['Time'].squeeze().head())

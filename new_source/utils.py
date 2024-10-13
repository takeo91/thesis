import re

def extract_column_names(file_path):
    column_names = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Skip empty lines or lines not starting with 'Column:'
            if not line or (not line.startswith('Column:') and not line.startswith('Label columns:')):
                continue
            # Handle 'Label columns:' section
            if line.startswith('Label columns:'):
                continue  # Skip this line
            # Extract column number and name
            else:
                # Split only on the first two spaces to handle descriptions with colons
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    col_num = parts[1]
                    col_name = parts[2]
                    # Use the part before ';' as the column name, if present
                    col_name = col_name.split(';')[0].strip()
                    column_names.append(col_name)
    return column_names

def extract_labels(file_path):
    # Initialize a dictionary to hold mappings for each category
    label_mappings = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            # Extract the components using regex
            match = re.match(r'(\d+)\s*-\s*([\w_]+)\s*-\s*(.+)', line)
            if match:
                index = int(match.group(1))
                track_name = match.group(2)
                label_name = match.group(3).strip()
                # Initialize the mapping for the track if not already
                if track_name not in label_mappings:
                    label_mappings[track_name] = {}
                # Add the mapping
                label_mappings[track_name][index] = label_name
            else:
                # Handle entries with combined indices (e.g., ML_Both_Arms)
                match_ml = re.match(r'(\d+)\s+(\d+)\s*-\s*([\w_]+)\s*-\s*(.+)', line)
                if match_ml:
                    index1 = int(match_ml.group(1))
                    index2 = int(match_ml.group(2))
                    combined_index = f"{index1}_{index2}"
                    track_name = match_ml.group(3)
                    label_name = match_ml.group(4).strip()
                    if track_name not in label_mappings:
                        label_mappings[track_name] = {}
                    label_mappings[track_name][combined_index] = label_name
    return label_mappings


def extract_metadata(column_name, label_columns):

    metadata = {
        'sensor_type': None,
        'body_part': None,
        'measurement_type': None,
        'axis': None,
        'unit': None
    }

    # Check if column_name is a label column
    if column_name in label_columns:
        metadata['sensor_type'] = 'Label'
        metadata['body_part'] = column_name  # Use label name as body_part or 'Label'
        metadata['measurement_type'] = 'Label'
        metadata['axis'] = 'N/A'
        metadata['unit'] = 'N/A'
        return metadata

    # Handle 'MILLISEC' column (time column)
    if column_name == 'MILLISEC':
        metadata['sensor_type'] = 'Time'
        metadata['body_part'] = 'N/A'
        metadata['measurement_type'] = 'Time'
        metadata['axis'] = 'N/A'
        metadata['unit'] = 'ms'
        return metadata

    # Regular expressions for different metadata attributes
    sensor_type_pattern = r'^(Accelerometer|Gyroscope|InertialMeasurementUnit|REED SWITCH|LOCATION)'
    body_part_pattern = r'(BACK|RUA|LUA|RLA|LLA|RKN\^|RKN_|HIP|LH|RH|RWR|LWR|L-SHOE|R-SHOE|CUP|SALAMI|WATER|CHEESE|BREAD|KNIFE1|KNIFE2|MILK|SPOON|SUGAR|PLATE|GLASS|DOOR1|DOOR2|LAZYCHAIR|DISHWASHER|UPPERDRAWER|LOWERDRAWER|MIDDLEDRAWER|FRIDGE|TAG1|TAG2|TAG3|TAG4)'
    measurement_type_pattern = r'(acc|gyro|magnetic|Quaternion|Eu|Nav_A|Body_A|AngVelBodyFrame|AngVelNavFrame|Compass|S[0-9])'
    axis_pattern = r'(X|Y|Z|1|2|3|4)'
    unit_pattern = r'unit = ([^,]+)'

    # Extract sensor type
    sensor_type_match = re.search(sensor_type_pattern, column_name)
    if sensor_type_match:
        metadata['sensor_type'] = sensor_type_match.group(0)

    # Extract body part
    body_part_match = re.search(body_part_pattern, column_name)
    if body_part_match:
        metadata['body_part'] = body_part_match.group(0)

    # Extract measurement type
    measurement_type_match = re.search(measurement_type_pattern, column_name)
    if measurement_type_match:
        metadata['measurement_type'] = measurement_type_match.group(0)

    # Extract axis
    axis_match = re.search(axis_pattern, column_name)
    if axis_match:
        metadata['axis'] = axis_match.group(0)

    # Extract unit (if available in the column name)
    unit_match = re.search(unit_pattern, column_name)
    if unit_match:
        metadata['unit'] = unit_match.group(1)

    return metadata
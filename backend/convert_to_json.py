import pandas as pd
import json
import os

# --- Configuration ---
CSV_FILE_PATH = 'data/household_power_consumption.txt'
JSON_OUTPUT_PATH = 'data/api_input.json'
ROWS_TO_PROCESS = 24

def create_dummy_csv():
    """Creates a dummy CSV file if the original is not found."""
    print(f"File '{CSV_FILE_PATH}' not found. Creating a dummy file for demonstration.")
    dummy_data = (
        "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
        "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
        "16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000\n"
    )
    # Add 22 more dummy rows
    for i in range(22):
        dummy_data += f"16/12/2006;17:{26+i}:00;3.50{i};0.30{i};235.0{i};15.0{i};0.000;2.000;17.000\n"

    with open(CSV_FILE_PATH, 'w') as f:
        f.write(dummy_data)
    print(f"Dummy file '{CSV_FILE_PATH}' created.")

def convert_csv_to_json():
    """
    Reads the first N rows of the power consumption CSV and converts them
    to the JSON format required by the Foresight API.
    """
    if not os.path.exists(CSV_FILE_PATH):
        create_dummy_csv()

    print(f"Reading the first {ROWS_TO_PROCESS} rows from '{CSV_FILE_PATH}'...")

    # Read the CSV, specifying the semicolon separator
    try:
        df = pd.read_csv(CSV_FILE_PATH, sep=';', nrows=ROWS_TO_PROCESS)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # A list to hold each formatted row dictionary
    data_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Combine Date and Time into a single string
        datetime_str = f"{row['Date']} {row['Time']}"

        # Convert to a datetime object and then format it to 'YYYY-MM-DD HH:MM:SS'
        formatted_dt = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

        # Create the dictionary for this row, converting values to float
        row_dict = {
            'dt': formatted_dt
        }
        # Loop through the rest of the columns to add them to the dict
        for col_name in df.columns:
            if col_name not in ['Date', 'Time']:
                try:
                    # Convert the value to a float, handling potential missing values
                    row_dict[col_name] = float(row[col_name])
                except (ValueError, TypeError):
                    row_dict[col_name] = None # Use null for invalid numbers

        data_list.append(row_dict)

    # Final JSON structure required by the API
    final_json_object = {"data": data_list}

    # Write the formatted JSON to the output file
    with open(JSON_OUTPUT_PATH, 'w') as json_file:
        json.dump(final_json_object, json_file, indent=4)

    print(f"\nSuccess! Conversion complete.")
    print(f"Check the output file: '{JSON_OUTPUT_PATH}'")

if __name__ == "__main__":
    convert_csv_to_json()

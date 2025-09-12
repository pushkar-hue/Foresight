import streamlit as st
import pandas as pd
import requests
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Foresight: Energy Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# --- App Title and Description ---
st.title("⚡ Foresight: Intelligent Energy Load Forecasting")
st.markdown("""
Welcome to Foresight, a high-performance tool for short-term energy load forecasting.
This dashboard demonstrates how our Bidirectional LSTM model can predict future energy needs,
enabling smarter energy management and helping to decarbonize buildings.
""")

# --- Constants ---
API_URL = "http://127.0.0.1:8000/predict"
N_STEPS = 24  # The lookback window your model expects

# --- Helper Function ---
def api_request_to_df(data_slice: pd.DataFrame) -> dict:
    """Converts a DataFrame slice to the list of dicts format required by the API."""
    data_list = []
    for index, row in data_slice.iterrows():
        datetime_str = f"{row['Date']} {row['Time']}"
        formatted_dt = pd.to_datetime(datetime_str, format='%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

        row_dict = {'dt': formatted_dt}
        for col_name in data_slice.columns:
            if col_name not in ['Date', 'Time']:
                try:
                    row_dict[col_name] = float(row[col_name])
                except (ValueError, TypeError):
                    row_dict[col_name] = None
        data_list.append(row_dict)
    return {"data": data_list}

# --- Main Application ---

# 1. File Uploader
st.header("1. Upload Your Energy Consumption Data")
st.write("Please upload a CSV file in the same format as the UCI dataset.")

uploaded_file = st.file_uploader("Choose a file (household_power_consumption.txt)", type="txt")

if uploaded_file is not None:
    try:
        # Load and cache the data
        @st.cache_data
        def load_data(file):
            df = pd.read_csv(file, sep=';', low_memory=False, na_values=['nan', '?'])
            df.dropna(subset=['Global_active_power'], inplace=True)
            return df

        df = load_data(uploaded_file)
        
        st.subheader("Data Preview")
        st.write(f"Successfully loaded and cleaned {len(df)} rows.")
        st.dataframe(df.tail())

        # 2. Prediction Section
        st.header("2. Generate and Evaluate Forecast")
        st.write("To demonstrate accuracy, we'll select a point in your data, use the 24 hours before it as input, and compare the model's forecast to the actual data that followed.")

        # Let user select a starting point for the demo prediction
        max_start_point = len(df) - N_STEPS - 72  # 72 is max forecast horizon
        if max_start_point > 0:
            test_start_point = st.slider(
                "Select a starting row for the test prediction",
                min_value=0,
                max_value=max_start_point,
                value=max_start_point - 168,  # Default to a week before the end
                step=24
            )
            forecast_horizon = st.slider("Forecast Horizon (hours)", min_value=1, max_value=72, value=24, step=1)
        else:
            st.warning("Not enough data in the uploaded file to perform a forecast comparison.")
            st.stop()

        if st.button("🔮 Generate Forecast & Compare"):
            with st.spinner('Asking the model for a prediction and preparing comparison...'):
                # Prepare data for the API from the selected slice
                input_data = df.iloc[test_start_point : test_start_point + N_STEPS * 60] # Use 24 hours of minute data
                payload = api_request_to_df(input_data)
                payload['forecast_horizon'] = forecast_horizon

                try:
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:
                        st.success("✅ Forecast generated successfully!")
                        prediction_data = response.json()['forecast']
                        
                        # Convert forecast to DataFrame
                        df_forecast = pd.DataFrame(prediction_data)
                        df_forecast.rename(columns={'timestamp': 'datetime', 'predicted_kw': 'Global_active_power'}, inplace=True)
                        df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime'])
                        
                        # Get the actual data for the same period for comparison
                        actual_data_slice = df.iloc[test_start_point + N_STEPS * 60 : test_start_point + N_STEPS * 60 + forecast_horizon * 60]
                        df_actual_raw = pd.DataFrame({
                            'datetime': pd.to_datetime(actual_data_slice['Date'] + ' ' + actual_data_slice['Time'], format='%d/%m/%Y %H:%M:%S'),
                            'Global_active_power': actual_data_slice['Global_active_power']
                        })
                        # Resample actual data to hourly to match forecast
                        df_actual = df_actual_raw.set_index('datetime').resample('h').mean().reset_index()


                        # --- Combined Visualization ---
                        st.subheader("Input, Actuals vs. Forecast")
                        
                        chart_data_hist_raw = pd.DataFrame({
                            'datetime': pd.to_datetime(input_data['Date'] + ' ' + input_data['Time'], format='%d/%m/%Y %H:%M:%S'),
                            'Global_active_power': input_data['Global_active_power']
                        })
                        # Resample historical data to hourly for clean plotting
                        chart_data_hist = chart_data_hist_raw.set_index('datetime').resample('h').mean().reset_index()

                        chart_data_hist['type'] = 'Input (Historical)'
                        df_forecast['type'] = 'Forecast (Predicted)'
                        df_actual['type'] = 'Actual'
                        
                        combined_chart_data = pd.concat([chart_data_hist, df_forecast, df_actual])

                        chart = alt.Chart(combined_chart_data).mark_line(
                            point=alt.OverlayMarkDef(size=15)
                        ).encode(
                            x=alt.X('datetime:T', title='Timestamp'),
                            y=alt.Y('Global_active_power:Q', title='Global Active Power (kW)', scale=alt.Scale(zero=False)),
                            color=alt.Color('type:N', scale=alt.Scale(domain=['Input (Historical)', 'Forecast (Predicted)', 'Actual'], range=['#0068c9', '#ff8700', '#00c968']), legend=alt.Legend(title="Data Type")),
                            tooltip=['datetime:T', 'Global_active_power:Q', 'type:N']
                        ).properties(
                            title='Energy Consumption: Past vs. Future'
                        ).interactive()
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # --- Display Forecast Data Table with Comparison ---
                        st.subheader("Forecasted vs. Actual Values")
                        df_forecast_table = df_forecast.set_index('datetime').rename(columns={'Global_active_power': 'Predicted_kW'})
                        df_actual_table = df_actual.set_index('datetime').rename(columns={'Global_active_power': 'Actual_kW'})
                        
                        comparison_df = df_forecast_table.join(df_actual_table)
                        comparison_df['Difference (kW)'] = comparison_df['Predicted_kW'] - comparison_df['Actual_kW']
                        
                        st.dataframe(comparison_df.style.format("{:.3f}"))

                    else:
                        st.error(f"❌ API Error: Status code {response.status_code}")
                        st.json(response.json())

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Connection Error: Could not connect to the API at {API_URL}. Is the backend server running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting file upload...")


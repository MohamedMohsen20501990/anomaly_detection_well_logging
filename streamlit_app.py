import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path

from src.inference import InferencePipeline


# --------------------------------------------------
# Application Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Well Log Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Model Configuration
# --------------------------------------------------

MODEL_PATH = Path("artifacts/model.pkl")

FEATURE_COLUMNS = (
    "NPHI",
    "RHOB",
    "GR",
    "CALI",
    "PEF",
    "DTC",
    "RDEP",
    "DEPTH_MD"
)


@st.cache_resource
def load_model():
    """Load and cache the trained inference pipeline."""
    return InferencePipeline(model_path=MODEL_PATH)


pipeline = load_model()


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("🛢️ Well Log Anomaly Detection")

st.markdown(
    """
    This application uses an **Isolation Forest model**
    to detect anomalies in well-log measurements.

    Choose your preferred input method below.
    """
)

st.divider()


# --------------------------------------------------
# Prediction Helper
# --------------------------------------------------

def make_predictions(df: pd.DataFrame):
    """
    Generate anomaly predictions using the inference pipeline.

    Args:
        df: DataFrame containing the required features.

    Returns:
        DataFrame containing the original features and predictions.
    """

    predictions = pipeline.predict(df)

    result_df = df.copy()

    result_df["Prediction"] = predictions["Predictions"]

    return result_df


def display_results(result_df: pd.DataFrame):
    """Display prediction results and download functionality."""

    total_records = len(result_df)

    anomalies = (result_df["Prediction"] == "anomaly").sum()

    normal = (result_df["Prediction"] == "normal").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", total_records)

    col2.metric("Normal Records", normal)

    col3.metric("Anomalies Detected", anomalies)

    st.subheader("Prediction Results")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv,
        file_name="anomaly_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# Input Tabs
# --------------------------------------------------

single_tab, multiple_tab, csv_tab = st.tabs(
    [
        "Single Prediction",
        "Multiple Predictions",
        "Upload CSV"
    ]
)


# ==================================================
# TAB 1: SINGLE PREDICTION
# ==================================================

with single_tab:

    st.subheader("Single Well Log Prediction")

    st.write("Enter the measurements for one record.")

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            nphi = st.number_input(
                "NPHI - Neutron Porosity",
                value=0.45,
                format="%.4f"
            )

            rhob = st.number_input(
                "RHOB - Bulk Density",
                value=2.35,
                format="%.4f"
            )

            gr = st.number_input(
                "GR - Gamma Ray",
                value=75.2,
                format="%.4f"
            )

            cali = st.number_input(
                "CALI - Borehole Diameter",
                value=8.5,
                format="%.4f"
            )

        with col2:

            pef = st.number_input(
                "PEF - Photoelectric Factor",
                value=3.1,
                format="%.4f"
            )

            dtc = st.number_input(
                "DTC - Sonic Travel Time",
                value=95.0,
                format="%.4f"
            )

            rdep = st.number_input(
                "RDEP - Deep Resistivity",
                value=12.5,
                format="%.4f"
            )

            depth_md = st.number_input(
                "DEPTH_MD - Measured Depth",
                value=1500.0,
                format="%.4f"
            )

        submitted = st.form_submit_button(
            "Predict Anomaly",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [{
                "NPHI": nphi,
                "RHOB": rhob,
                "GR": gr,
                "CALI": cali,
                "PEF": pef,
                "DTC": dtc,
                "RDEP": rdep,
                "DEPTH_MD": depth_md
            }],
            columns=FEATURE_COLUMNS
        )

        try:

            result_df = make_predictions(input_df)

            display_results(result_df)

            prediction = result_df["Prediction"].iloc[0]

            if prediction == "normal":
                st.success("The record is classified as NORMAL.")

            else:
                st.error("The record is classified as ANOMALY.")

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ==================================================
# TAB 2: MULTIPLE PREDICTIONS
# ==================================================

with multiple_tab:

    st.subheader("Multiple Well Log Predictions")

    st.write(
        "Enter multiple measurements for each feature. "
        "Each position represents one record."
    )

    with st.form("multiple_prediction_form"):

        number_of_records = st.number_input(
            "Number of records",
            min_value=1,
            max_value=100,
            value=3,
            step=1
        )

        st.info(
            "Enter comma-separated values for each feature. "
            "Every feature must contain the same number of values."
        )

        col1, col2 = st.columns(2)

        with col1:

            nphi_values = st.text_input(
                "NPHI",
                "0.45, 0.20, 0.40"
            )

            rhob_values = st.text_input(
                "RHOB",
                "2.35, 2.40, 2.30"
            )

            gr_values = st.text_input(
                "GR",
                "75.2, 82.5, 68.4"
            )

            cali_values = st.text_input(
                "CALI",
                "8.5, 8.7, 8.3"
            )

        with col2:

            pef_values = st.text_input(
                "PEF",
                "3.1, 2.9, 3.2"
            )

            dtc_values = st.text_input(
                "DTC",
                "95.0, 102.5, 91.0"
            )

            rdep_values = st.text_input(
                "RDEP",
                "12.5, 15.2, 10.8"
            )

            depth_values = st.text_input(
                "DEPTH_MD",
                "1500.0, 1501.0, 1502.0"
            )

        submitted_multiple = st.form_submit_button(
            "Predict Multiple Records",
            use_container_width=True
        )

    if submitted_multiple:

        try:

            input_data = {
                "NPHI": nphi_values,
                "RHOB": rhob_values,
                "GR": gr_values,
                "CALI": cali_values,
                "PEF": pef_values,
                "DTC": dtc_values,
                "RDEP": rdep_values,
                "DEPTH_MD": depth_values
            }

            parsed_data = {}

            for feature, values in input_data.items():

                parsed_data[feature] = [
                    float(value.strip())
                    for value in values.split(",")
                    if value.strip()
                ]

                if len(parsed_data[feature]) != number_of_records:
                    raise ValueError(
                        f"{feature} must contain exactly "
                        f"{number_of_records} values."
                    )

            input_df = pd.DataFrame(
                parsed_data,
                columns=FEATURE_COLUMNS
            )

            result_df = make_predictions(input_df)

            display_results(result_df)

        except ValueError as e:

            st.error(f"Invalid input: {e}")

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ==================================================
# TAB 3: CSV UPLOAD
# ==================================================

with csv_tab:

    st.subheader("Upload Well Log CSV")

    st.write(
        "Upload a CSV file containing the eight required "
        "well-log features."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            input_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")

            st.dataframe(
                input_df.head(),
                use_container_width=True
            )

            missing_columns = [
                col for col in FEATURE_COLUMNS
                if col not in input_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            elif input_df.empty:

                st.warning("The uploaded CSV file is empty.")

            else:

                if st.button(
                    "Run Anomaly Detection",
                    use_container_width=True
                ):

                    input_df = input_df.loc[
                        :, FEATURE_COLUMNS
                    ].copy()

                    result_df = make_predictions(input_df)

                    display_results(result_df)

        except Exception as e:

            st.error(f"Error processing CSV file: {e}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Well Log Anomaly Detection | "
    "Powered by Python, Streamlit and Isolation Forest"
)
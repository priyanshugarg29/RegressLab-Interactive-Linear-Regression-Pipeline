import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, probplot
import matplotlib.pyplot as plt
import io

st.title("RegressLab: Interactive Linear Regression Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.header("Initial Data Preview and Quality Checks")
    st.write(data.head())

    dup_count = data.duplicated().sum()
    if dup_count > 0:
        with st.expander("Duplicate Records Detected"):
            st.write(f"Duplicate records found: {dup_count}")
            if st.radio("Drop duplicate rows?", ("Yes", "No")) == "Yes":
                data = data.drop_duplicates()
                st.write(f"Duplicates dropped. New data shape: {data.shape}")
    else:
        st.info("No duplicate records found.")

    missing_info = data.isnull().mean() * 100
    missing_info = missing_info[missing_info > 0]
    if not missing_info.empty:
        with st.expander("Missing Values Info"):
            st.write("Columns with missing values (percentage):")
            st.write(missing_info)
            action = st.radio("Handle missing values:", ("Drop rows", "Impute", "Do nothing"))
            if action == "Drop rows":
                data = data.dropna()
                st.write(f"Rows with missing values dropped. New shape: {data.shape}")
            elif action == "Impute":
                for col in missing_info.index:
                    impute_method = st.selectbox(f"Imputation method for {col}", ("Mean", "Median", "Mode"), key=f"impute_{col}")
                    if impute_method == "Mean":
                        val = data[col].mean()
                    elif impute_method == "Median":
                        val = data[col].median()
                    else:
                        val = data[col].mode().iloc[0]
                    data[col] = data[col].fillna(val)
                    st.write(f"Imputed {col} with {impute_method} ({val})")
    else:
        st.info("No missing values found.")

    st.header("Target and Feature Selection")
    columns = data.columns.to_list()
    target = st.selectbox('Select target variable', columns)
    features = st.multiselect('Select feature columns', [col for col in columns if col != target], default=[col for col in columns if col != target])

    selected_columns = [target] + features
    non_numeric = [col for col in selected_columns if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric:
        st.warning(f"Non-numeric columns selected: {non_numeric}. Please select numeric columns only.")
    else:
        st.success("Numeric columns selected.")

        final_stats = {}
        final_plots = {}

        def plot_distributions(col):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].hist(data[col].dropna(), bins=30, color='c', edgecolor='k', alpha=0.7)
            axs[0].set_title(f"Histogram of {col}")
            axs[0].set_xlabel(col)
            axs[0].set_ylabel("Frequency")

            probplot(data[col].dropna(), dist="norm", plot=axs[1])
            axs[1].set_title(f"Q-Q Plot of {col}")

            plt.tight_layout()
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            return buf

        def analyze_column(col):
            st.subheader(f"Analysis for {col}")

            orig_skew = skew(data[col].dropna())
            orig_kurt = kurtosis(data[col].dropna())
            st.write(f"Original skewness: {orig_skew:.3f}")
            st.write(f"Original kurtosis: {orig_kurt:.3f}")

            if abs(orig_skew) < 0.5:
                st.write("- Skewness close to zero: roughly symmetric.")
                skew_reco = "No transform needed."
            elif 0.5 <= abs(orig_skew) < 1:
                st.write("- Moderate skewness: consider sqrt or log transform.")
                skew_reco = "Sqrt or log recommended."
            else:
                st.write("- High skewness: sqrt or log transform recommended.")
                skew_reco = "Sqrt or log strongly recommended."
            st.write(f"Skewness recommendation: {skew_reco}")

            treatment = st.radio(f"Outlier treatment for {col}", ("None", "Remove outliers", "Clip outliers"), key=f"outlier_{col}")
            if treatment == "Remove outliers":
                Q1 = np.percentile(data[col], 25)
                Q3 = np.percentile(data[col], 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before_rows = data.shape[0]
                data[col] = data[col].where((data[col] >= lower) & (data[col] <= upper), np.nan)
                data.dropna(inplace=True)
                after_rows = data.shape[0]
                removed = before_rows - after_rows
                st.write(f"Removed {removed} rows due to outliers.")
            elif treatment == "Clip outliers":
                low_pct = st.slider(f"Lower percentile clip for {col}", 0, 20, 5, key=f"lowclip_{col}")
                high_pct = st.slider(f"Upper percentile clip for {col}", 80, 100, 95, key=f"highclip_{col}")
                low_val = np.percentile(data[col], low_pct)
                high_val = np.percentile(data[col], high_pct)
                data[col] = np.clip(data[col], low_val, high_val)
                st.write(f"Clipped values outside [{low_val:.3f}, {high_val:.3f}].")

            transformation = st.selectbox(f"Transformation for {col}", ("None", "Square root", "Log (log1p)"), key=f"trans_{col}")

            if transformation == "Square root":
                if (data[col] < 0).any():
                    st.warning("Negative values present, sqrt transform skipped.")
                else:
                    data[col] = np.sqrt(data[col])
                    st.write("Applied square root transformation.")
            elif transformation == "Log (log1p)":
                if (data[col] < 0).any():
                    st.warning("Negative values present, log1p transform skipped.")
                else:
                    data[col] = np.log1p(data[col])
                    st.write("Applied log1p transformation.")
            else:
                st.write("No transformation applied.")

            final_skew = skew(data[col].dropna())
            final_kurt = kurtosis(data[col].dropna())
            st.write(f"Final skewness: {final_skew:.3f}")
            st.write(f"Final kurtosis: {final_kurt:.3f}")

            buf = plot_distributions(col)

            final_stats[col] = {
                "original_skew": orig_skew,
                "original_kurtosis": orig_kurt,
                "final_skew": final_skew,
                "final_kurtosis": final_kurt,
                "transformation": transformation,
                "outlier_treatment": treatment
            }
            final_plots[col] = buf

        for feature in features:
            analyze_column(feature)

        st.header("Target Variable Analysis")
        analyze_column(target)

        st.header("Summary")
        st.write("Final feature and target statistics and plots are ready to be passed for linear regression suitability analysis.")

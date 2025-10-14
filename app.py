import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

st.title("RegressLab: Interactive Linear Regression Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("First five rows of your data:")
    st.write(data.head())

    st.write("DATASET SUMMARY")
    columns = data.columns.to_list()
    st.write(f"THE DATASET HAS {len(columns)} COLUMNS: {columns}")

    st.write("[target and feature selection]")
    target = st.selectbox('Select target variable', columns)
    features = st.multiselect('Select feature columns', columns, default=[col for col in columns if col != target])

    selected_columns = [target] + features
    non_numeric = [col for col in selected_columns if not pd.api.types.is_numeric_dtype(data[col])]

    if non_numeric:
        st.warning(f"The following selected columns are not numeric: {non_numeric}. Please select only numeric columns for regression.")
    else:
        st.write(f"Selected target variable: {target}, Data_type: {type(target)} (should be <str>)")
        st.write(f"Selected {len(features)} feature columns: {features}, Data_type: {type(features)} (should be <list>)")
        st.write("[Checking skewness and kurtosis of features]")

        def recommendations_skew_kurt(col, skew_val, kurt_val):
            st.write(f"Feature: {col}")
            st.write(f"Skewness: {skew_val:.3f}")
            st.write(f"Kurtosis: {kurt_val:.3f}")

            # Skewness recommendations
            if abs(skew_val) < 0.5:
                st.write("  - Skewness near 0: data is approximately symmetric.")
                skew_recommendation = "No transformation needed."
            elif 0.5 <= abs(skew_val) < 1:
                st.write("  - Moderate skewness detected.")
                skew_recommendation = "Consider mild transformations such as square root or cube root."
            else:
                st.write("  - High skewness detected.")
                skew_recommendation = "Consider log or Box-Cox/Yeo-Johnson transformation."

            # Kurtosis recommendations
            if 2 < kurt_val < 4:  # around normal kurtosis ~3
                st.write("  - Kurtosis near normal (3), no action needed.")
                kurt_recommendation = "No transformation needed."
            elif kurt_val >= 4:
                st.write("  - High kurtosis: heavy tails/outliers likely.")
                kurt_recommendation = "Consider outlier treatment like winsorizing/trimming."
            else:
                st.write("  - Low kurtosis: light tails, fewer outliers.")
                kurt_recommendation = "Usually no transformation needed."

            st.write("Recommendations:")
            st.write(f"  - Skewness: {skew_recommendation}")
            st.write(f"  - Kurtosis: {kurt_recommendation}")

            # Plot distribution
            fig, ax = plt.subplots()
            ax.hist(data[col], bins=30, color='c', edgecolor='k', alpha=0.65)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            st.write("---")

        for col in features:
            sk = skew(data[col].dropna())
            ku = kurtosis(data[col].dropna())
            recommendations_skew_kurt(col, sk, ku)

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

    dup_count = data.duplicated().sum()
    if dup_count > 0:
        st.write(f"Duplicate records found: {dup_count}")
        if st.radio("Do you want to drop duplicate rows?", ("Yes", "No")) == "Yes":
            data = data.drop_duplicates()
            st.write(f"Duplicates dropped. New data shape: {data.shape}")
    else:
        st.write("No duplicate records found.")

    missing_info = data.isnull().mean() * 100
    missing_info = missing_info[missing_info > 0]
    if not missing_info.empty:
        st.write("Columns with missing values (percentage):")
        st.write(missing_info)

        action = st.radio("Choose how to handle missing values:", ("Drop rows with missing values", "Impute missing values", "Do nothing"))
        if action == "Drop rows with missing values":
            data = data.dropna()
            st.write(f"Rows with missing values dropped. New shape: {data.shape}")
        elif action == "Impute missing values":
            for col in missing_info.index:
                st.write(f"Impute missing values for column: {col}")
                impute_method = st.selectbox(
                    f"Choose imputation method for {col}",
                    ("Mean", "Median", "Mode"),
                    key=col
                )
                if impute_method == "Mean":
                    value = data[col].mean()
                elif impute_method == "Median":

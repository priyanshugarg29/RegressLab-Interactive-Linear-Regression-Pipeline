import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

st.title("RegressLab: Interactive Linear Regression Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.header("Initial Data Preview and Quality Checks")
    st.write("First five rows of your data:")
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
        st.warning(f"Selected non-numeric columns: {non_numeric}. Please pick numeric columns only.")
    else:
        st.success("Numeric columns selected correctly.")

        st.header("Feature Analysis, Outlier Treatment & Transformation")

        def plot_hist(col):
            fig, ax = plt.subplots()
            ax.hist(data[col], bins=30, color='c', edgecolor='k', alpha=0.65)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        def plot_boxplot(col, title):
            fig, ax = plt.subplots()
            ax.boxplot(data[col], vert=False)
            ax.set_title(title)
            ax.set_xlabel(col)
            st.pyplot(fig)

        def treat_outliers(col):
            st.subheader(f"Outliers in {col}")
            plot_boxplot(col, "Original Boxplot with Outliers")

            treatment = st.radio(f"Outlier treatment for {col}", ("None", "Remove", "Clip"), key=f"outlier_{col}")
            if treatment == "Remove":
                Q1 = np.percentile(data[col], 25)
                Q3 = np.percentile(data[col], 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                removed = ((data[col] < lower) | (data[col] > upper)).sum()
                data[col] = data[col].where((data[col] >= lower) & (data[col] <= upper), np.nan)
                st.write(f"Removed {removed} outliers (set as NaN).")
            elif treatment == "Clip":
                low_perc = st.slider(f"Lower percentile clip for {col}", 0, 20, 5, key=f"low_{col}")
                high_perc = st.slider(f"Upper percentile clip for {col}", 80, 100, 95, key=f"high_{col}")
                low_val = np.percentile(data[col], low_perc)
                high_val = np.percentile(data[col], high_perc)
                data[col] = np.clip(data[col], low_val, high_val)
                st.write(f"Clipped to [{low_val:.3f}, {high_val:.3f}].")

            if treatment != "None":
                plot_boxplot(col, "Boxplot After Outlier Treatment")

        def apply_transformations(col):
            st.subheader(f"Transformation options for {col}")
            plot_hist(col)

            choice = st.selectbox(f"Choose transformation for {col}", ("None", "Square root", "Log (log1p)"), key=f"transform_{col}")

            if choice == "Square root":
                if (data[col] < 0).any():
                    st.warning("Negative values present, sqrt transform skipped")
                else:
                    data[col] = np.sqrt(data[col])
                    st.write(f"Applied square root transform on {col}")
            elif choice == "Log (log1p)":
                if (data[col] < 0).any():
                    st.warning("Negative values present, log1p transform skipped")
                else:
                    data[col] = np.log1p(data[col])
                    st.write(f"Applied log1p transform on {col}")
            else:
                st.write("No transformation applied.")
            plot_hist(col)

        for col in features:
            st.markdown(f"### Feature: {col}")
            treat_outliers(col)
            apply_transformations(col)

        st.header("Final Feature Distribution & Statistics")
        for col in features:
            s, k = skew(data[col].dropna()), kurtosis(data[col].dropna())
            st.write(f"Feature: {col}")
            st.write(f"Skewness: {s:.3f}, Kurtosis: {k:.3f}")
            plot_hist(col)
            st.write("---")

        st.header("Target Variable Analysis")
        st.write(f"Target: {target}")

        plot_hist(target)

        target_choice = st.selectbox("Transform target variable", ("None", "Square root", "Log (log1p)"), key="target_transform")

        if target_choice == "Square root":
            if (data[target] < 0).any():
                st.warning("Negative values present, sqrt transform skipped for target")
            else:
                data[target] = np.sqrt(data[target])
                st.write("Applied square root transform on target")
        elif target_choice == "Log (log1p)":
            if (data[target] < 0).any():
                st.warning("Negative values present, log1p transform skipped for target")
            else:
                data[target] = np.log1p(data[target])
                st.write("Applied log1p transform on target")
        else:
            st.write("No transformation applied on target")

        plot_hist(target)

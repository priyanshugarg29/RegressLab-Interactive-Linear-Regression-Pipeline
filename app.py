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
                    value = data[col].median()
                else:
                    value = data[col].mode().iloc[0]
                data[col] = data[col].fillna(value)
                st.write(f"Missing values in {col} imputed with {impute_method} ({value})")
    else:
        st.write("No missing values found.")

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
        st.write("----------------------------UNIVARIATE ANALYSIS TYPICAL FOR LINEAR REGRESSION-------------------------------------")
        st.write("[Checking skewness and kurtosis of features]")

        def recommendations_skew_kurt(col, skew_val, kurt_val):
            st.write(f"Feature: {col}")
            st.write(f"Skewness: {skew_val:.3f}")
            st.write(f"Kurtosis: {kurt_val:.3f}")

            if abs(skew_val) < 0.5:
                st.write("  - Skewness near 0: data is approximately symmetric.")
                skew_recommendation = "No transformation needed."
            elif 0.5 <= abs(skew_val) < 1:
                st.write("  - Moderate skewness detected.")
                skew_recommendation = "Consider mild transformations such as square root or cube root."
            else:
                st.write("  - High skewness detected.")
                skew_recommendation = "Consider log or Box-Cox/Yeo-Johnson transformation."

            if 2 < kurt_val < 4:
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

            fig, ax = plt.subplots()
            ax.hist(data[col], bins=30, color='c', edgecolor='k', alpha=0.65)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            st.write("[Outlier detection and treatment]")
            def plot_boxplot(data_col, title):
                fig, ax = plt.subplots()
                ax.boxplot(data_col, vert=False)
                ax.set_title(title)
                ax.set_xlabel(col)
                st.pyplot(fig)

            plot_boxplot(data[col].dropna(), "Original Distribution with Outliers")

            treatment = st.radio(
                f"Choose outlier treatment for {col}",
                ("None", "Remove outliers", "Clip outliers at percentiles"),
                key=f"outlier_{col}"
            )

            if treatment == "Remove outliers":
                Q1 = np.percentile(data[col], 25)
                Q3 = np.percentile(data[col], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                removed_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                data[col] = data[col].where((data[col] >= lower_bound) & (data[col] <= upper_bound), np.nan)
                st.write(f"Removed {removed_count} outlier values by setting them as NaN.")
            elif treatment == "Clip outliers at percentiles":
                lower_percentile = st.slider(f"Select lower percentile clip for {col}", 0, 20, 5, key=f"low_clip_{col}")
                upper_percentile = st.slider(f"Select upper percentile clip for {col}", 80, 100, 95, key=f"high_clip_{col}")
                lower_bound = np.percentile(data[col], lower_percentile)
                upper_bound = np.percentile(data[col], upper_percentile)
                data[col] = np.clip(data[col], lower_bound, upper_bound)
                st.write(f"Clipped values outside {lower_percentile}th and {upper_percentile}th percentiles.")
            else:
                st.write("No outlier treatment applied.")

            if treatment != "None":
                plot_boxplot(data[col].dropna(), "Updated Distribution After Outlier Treatment")

            st.write("---")

        for col in features:
            sk = skew(data[col].dropna())
            ku = kurtosis(data[col].dropna())
            recommendations_skew_kurt(col, sk, ku)

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, probplot
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("RegressLab: Interactive Linear Regression Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    if st.checkbox("Use default dataset (advertising.csv)"):
        default_path = os.path.join(os.path.dirname(__file__), "advertising.csv")
        if os.path.exists(default_path):
            data_original = pd.read_csv(default_path)
            st.success("Loaded default dataset: advertising.csv")
        else:
            st.error("Default dataset advertising.csv not found in app directory.")
else:
    data_original = pd.read_csv(uploaded_file)
    st.success("Uploaded CSV loaded successfully.")

if 'data_original' in locals():
    data = data_original.copy()

    st.header("Initial Data Preview and Quality Checks")
    st.write(data.head())

    dup_count = data.duplicated().sum()
    if dup_count > 0:
        with st.expander("Duplicate Records Detected"):
            st.write(f"Duplicate records found: {dup_count}")
            drop_dups = st.radio("Drop duplicate rows?", ("Yes", "No"))
            if drop_dups == "Yes":
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
            missing_action = st.radio("Handle missing values:", ("Drop rows", "Impute", "Do nothing"))
            if missing_action == "Drop rows":
                data = data.dropna()
                st.write(f"Rows with missing values dropped. New shape: {data.shape}")
            elif missing_action == "Impute":
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
            plt.close(fig)

        def analyze_column(col):
            st.subheader(f"Analysis for {col}")
            col_data = data[col].copy()

            orig_skew = skew(col_data.dropna())
            orig_kurt = kurtosis(col_data.dropna())
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
                Q1 = np.percentile(col_data, 25)
                Q3 = np.percentile(col_data, 75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                before_rows = len(col_data)
                col_data = col_data.where((col_data >= lower) & (col_data <= upper))
                removed = before_rows - col_data.dropna().shape[0]
                st.write(f"Removed {removed} outliers (set to NaN).")
            elif treatment == "Clip outliers":
                low_pct = st.slider(f"Lower percentile clip for {col}", 0, 20, 5, key=f"lowclip_{col}")
                high_pct = st.slider(f"Upper percentile clip for {col}", 80, 100, 95, key=f"highclip_{col}")
                low_val = np.percentile(col_data.dropna(), low_pct)
                high_val = np.percentile(col_data.dropna(), high_pct)
                col_data = np.clip(col_data, low_val, high_val)
                st.write(f"Clipped values outside [{low_val:.3f}, {high_val:.3f}].")

            transformation = st.selectbox(f"Transformation for {col}", ("None", "Square root", "Log (log1p)"), key=f"trans_{col}")

            if transformation == "Square root":
                if (col_data < 0).any():
                    st.warning("Negative values present, sqrt transform skipped.")
                else:
                    col_data = np.sqrt(col_data)
                    st.write("Applied square root transformation.")
            elif transformation == "Log (log1p)":
                if (col_data < 0).any():
                    st.warning("Negative values present, log1p transform skipped.")
                else:
                    col_data = np.log1p(col_data)
                    st.write("Applied log1p transformation.")
            else:
                st.write("No transformation applied.")

            data[col] = col_data

            final_skew = skew(col_data.dropna())
            final_kurt = kurtosis(col_data.dropna())
            st.write(f"Final skewness: {final_skew:.3f}")
            st.write(f"Final kurtosis: {final_kurt:.3f}")

            plot_distributions(col)

        for feature in features:
            analyze_column(feature)

        st.header("Target Variable Analysis")
        analyze_column(target)

        st.header("Correlation Matrix of Numeric Columns")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            corr = data[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough numeric columns to show correlation matrix.")

        st.header("Manual Gradient Descent for Linear Regression")

        reg_type = st.selectbox(
            "Choose regularization:",
            ("None", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net")
        )

        if reg_type == "Elastic Net":
            alpha = st.number_input("Alpha (Regularization strength)", min_value=0.0, value=0.1, step=0.01)
        else:
            alpha = 0.0

        learning_rate = st.number_input("Learning rate", min_value=0.0001, value=0.01, step=0.001)
        iterations = st.number_input("Number of iterations", min_value=10, value=50, step=10)

        # Prepare data matrix X and target vector y for gradient descent
        X = np.column_stack((np.ones(data.shape[0]), data[features].values))
        y = data[target].values

        theta = np.zeros(X.shape[1])

        st.write(f"Starting manual gradient descent for {iterations} iterations with {reg_type} regularization...")
        if reg_type == "None":
            st.write("No regularization will be applied.")
        elif reg_type == "L1 (Lasso)":
            st.write("L1 regularization will be applied, promoting sparsity in coefficients.")
        elif reg_type == "L2 (Ridge)":
            st.write("L2 regularization will be applied, shrinking coefficients towards zero.")
        else:
            st.write(f"Elastic Net regularization with alpha={alpha} and mixing parameter 0.5 will be applied.")

        def compute_cost(X, y, theta, reg_type, alpha):
            m = len(y)
            predictions = X.dot(theta)
            errors = predictions - y
            mse = (1/(2*m)) * np.sum(errors ** 2)

            if reg_type == "None":
                reg_term = 0
            elif reg_type == "L1 (Lasso)":
                reg_term = alpha * np.sum(np.abs(theta[1:])) / m
            elif reg_type == "L2 (Ridge)":
                reg_term = (alpha/(2*m)) * np.sum(theta[1:] ** 2)
            else:  # Elastic Net
                l1_ratio = 0.5
                l1 = np.sum(np.abs(theta[1:]))
                l2 = np.sum(theta[1:] ** 2)
                reg_term = alpha * (l1_ratio * l1 + (1 - l1_ratio) * l2/2) / m
            return mse + reg_term

        def gradient_descent_step(X, y, theta, learning_rate, reg_type, alpha):
            m = len(y)
            predictions = X.dot(theta)
            errors = predictions - y
            grad = (1/m) * X.T.dot(errors)

            if reg_type == "None":
                reg_grad = np.zeros(theta.shape)
            elif reg_type == "L1 (Lasso)":
                reg_grad = np.concatenate(([0], alpha * np.sign(theta[1:]) / m))
            elif reg_type == "L2 (Ridge)":
                reg_grad = np.concatenate(([0], (alpha/m) * theta[1:]))
            else:  # Elastic Net
                l1_ratio = 0.5
                l1_grad = np.sign(theta[1:])
                l2_grad = theta[1:]
                reg_grad_tail = alpha / m * (l1_ratio * l1_grad + (1 - l1_ratio) * l2_grad)
                reg_grad = np.concatenate(([0], reg_grad_tail))

            grad += reg_grad
            theta_new = theta - learning_rate * grad
            return theta_new, grad

        for i in range(1, int(iterations)+1):
            cost = compute_cost(X, y, theta, reg_type, alpha)
            st.write(f"Iteration {i}: Cost = {cost:.6f}")
            st.write(f"Current parameters: {theta}")
            theta, grad = gradient_descent_step(X, y, theta, learning_rate, reg_type, alpha)
            st.write(f"Gradient: {grad}")
            st.write(f"Updated parameters: {theta}")
            st.write("We update parameters in the direction opposite to the gradient to minimize the cost function.\n")

        st.write("Gradient descent complete.")
        st.write(f"Final parameters: {theta}")

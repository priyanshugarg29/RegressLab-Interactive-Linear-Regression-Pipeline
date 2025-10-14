import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, probplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split

st.title("RegressLab: Interactive Linear Regression Pipeline")

st.write("""
Welcome to RegressLab!

This interactive app walks you through the entire linear regression process — from data upload and cleaning, through exploratory and statistical analyses, to manually training models using gradient descent with customizable regularization and feature scaling.

You can explore how the regression hyperplane fits your data visually in 3D for any pair of features, and evaluate your model’s predictive performance on training and test splits with detailed metrics.

Designed for students and researchers, RegressLab makes learning linear regression transparent, hands-on, and intuitive.
""")

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

def manual_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def manual_minmax_scaler(X):
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    return (X - min_) / (max_ - min_), min_, max_

def manual_robust_scaler(X):
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return (X - median) / iqr, median, iqr

def evaluate_regression(y_true, y_pred):
    n = len(y_true)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    mse = ss_res / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    mape = np.mean(np.abs(residuals / y_true)) * 100 if np.all(y_true != 0) else np.nan
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE (%)': mape
    }

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

        # New: Train-Test Split option
        split_ratio = st.slider("Train-Test Split Ratio (Train %)", 0.1, 0.9, 0.8, 0.05)
        train_df, test_df = train_test_split(data, train_size=split_ratio, random_state=42)

        X_train_raw = train_df[features].values
        y_train = train_df[target].values
        X_test_raw = test_df[features].values
        y_test = test_df[target].values

        # Feature scaling method selection with default "Standardization (Z-score)"
        st.header("Feature Scaling Options")
        scaling_method = st.selectbox(
            "Choose a feature scaling method:",
            ("None", "Standardization (Z-score)", "Min-Max Scaling", "Robust Scaling"),
            index=1
        )

        if scaling_method == "None":
            X_train_scaled = X_train_raw
            X_test_scaled = X_test_raw
        elif scaling_method == "Standardization (Z-score)":
            mean = np.mean(X_train_raw, axis=0)
            std = np.std(X_train_raw, axis=0)
            X_train_scaled = (X_train_raw - mean) / std
            X_test_scaled = (X_test_raw - mean) / std
        elif scaling_method == "Min-Max Scaling":
            min_ = np.min(X_train_raw, axis=0)
            max_ = np.max(X_train_raw, axis=0)
            X_train_scaled = (X_train_raw - min_) / (max_ - min_)
            X_test_scaled = (X_test_raw - min_) / (max_ - min_)
        else:
            median = np.median(X_train_raw, axis=0)
            q1 = np.percentile(X_train_raw, 25, axis=0)
            q3 = np.percentile(X_train_raw, 75, axis=0)
            iqr = q3 - q1
            X_train_scaled = (X_train_raw - median) / iqr
            X_test_scaled = (X_test_raw - median) / iqr

        X_train = np.column_stack((np.ones(len(train_df)), X_train_scaled))
        X_test = np.column_stack((np.ones(len(test_df)), X_test_scaled))

        st.header("Manual Gradient Descent for Linear Regression")
        reg_type = st.selectbox(
            "Choose regularization:",
            ("None", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net"),
            index=0
        )

        if reg_type == "Elastic Net":
            alpha = st.number_input("Alpha (Regularization strength)", min_value=0.0, value=0.1, step=0.01, format="%.10f")
        else:
            alpha = 0.0

        recommended_lr = 0.01
        st.info(f"Recommended learning rate is about {recommended_lr}.")

        learning_rate = st.number_input("Learning rate", min_value=1e-10, value=recommended_lr, step=1e-7, format="%e")

        convergence_threshold = max(learning_rate * 1e-3, 1e-10)
        st.write(f"Automatically set convergence threshold = {convergence_threshold:.12f} (proportional to learning rate)")

        max_iterations = st.number_input("Maximum iterations", min_value=10, value=1000, step=10)

        theta = np.zeros(X_train.shape[1])

        st.write(f"Starting manual gradient descent for up to {max_iterations} iterations with {reg_type} regularization...")

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
            else:
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
            else:
                l1_ratio = 0.5
                l1_grad = np.sign(theta[1:])
                l2_grad = theta[1:]
                reg_grad_tail = alpha / m * (l1_ratio * l1_grad + (1 - l1_ratio) * l2_grad)
                reg_grad = np.concatenate(([0], reg_grad_tail))

            grad += reg_grad
            theta_new = theta - learning_rate * grad
            return theta_new, grad

        previous_cost = float('inf')
        for i in range(1, int(max_iterations)+1):
            cost = compute_cost(X_train, y_train, theta, reg_type, alpha)

            if i % 10 == 0 or i == int(max_iterations):
                st.write(f"--- Iteration {i} ---")
                st.write(f"Training cost (loss): {cost:.12f}")

                param_desc = [f"Intercept (bias): {theta[0]:.6f}"]
                param_desc += [f"Weight '{feat}': {theta[j+1]:.6f}" for j, feat in enumerate(features)]
                st.write("Parameters:")
                for desc in param_desc:
                    st.write(f"- {desc}")

                theta, grad = gradient_descent_step(X_train, y_train, theta, learning_rate, reg_type, alpha)

                grad_desc = [f"Gradient for Intercept (bias): {grad[0]:.6f}"]
                grad_desc += [f"Gradient for '{feat}': {grad[j+1]:.6f}" for j, feat in enumerate(features)]
                st.write("Gradient:")
                for desc in grad_desc:
                    st.write(f"- {desc}")

                st.write("Parameters updated by moving opposite the gradient to minimize the cost.\n")
            else:
                theta, grad = gradient_descent_step(X_train, y_train, theta, learning_rate, reg_type, alpha)

            cost_change = abs(previous_cost - cost)
            if cost_change < convergence_threshold:
                st.write(f"Convergence achieved: cost change {cost_change:.12f} < threshold {convergence_threshold:.12f}")
                st.write(f"Stopping gradient descent at iteration {i}.")
                break
            previous_cost = cost

        st.success("Gradient descent complete.")
        
        train_preds = X_train.dot(theta)
        test_preds = X_test.dot(theta)

        train_df['predicted_target'] = train_preds
        test_df['predicted_target'] = test_preds

        st.header("Training Set Evaluation Metrics")
        train_metrics = evaluate_regression(y_train, train_preds)
        for metric, value in train_metrics.items():
            st.write(f"{metric}: {value:.6f}")

        st.header("Test Set Evaluation Metrics")
        test_metrics = evaluate_regression(y_test, test_preds)
        for metric, value in test_metrics.items():
            st.write(f"{metric}: {value:.6f}" if not np.isnan(value) else f"{metric}: Undefined (zero values in target)")

        st.write("------")
        
        st.header("3D Visualization of Regression Plane and Data Points (Training Data)")

        if len(features) < 2:
            st.info("Select at least two features for 3D visualization.")
        else:
            feature_pair_1 = st.selectbox("Select feature 1 for 3D plot", features, index=0)
            other_features = [f for f in features if f != feature_pair_1]
            feature_pair_2 = st.selectbox("Select feature 2 for 3D plot", other_features, index=0)

            f1_idx = features.index(feature_pair_1)
            f2_idx = features.index(feature_pair_2)

            fixed_features_means = {f: 0.0 for f in features}
            fixed_features_means.pop(feature_pair_1)
            fixed_features_means.pop(feature_pair_2)

            f1_vals = np.linspace(X_train_scaled[:, f1_idx].min(), X_train_scaled[:, f1_idx].max(), 30)
            f2_vals = np.linspace(X_train_scaled[:, f2_idx].min(), X_train_scaled[:, f2_idx].max(), 30)
            xx, yy = np.meshgrid(f1_vals, f2_vals)

            grid_shape = xx.shape
            X_grid = np.ones((xx.size, X_train.shape[1]))
            X_grid[:, f1_idx + 1] = xx.ravel()
            X_grid[:, f2_idx + 1] = yy.ravel()

            for feat, val in fixed_features_means.items():
                idx = features.index(feat) + 1
                X_grid[:, idx] = val

            zz = X_grid.dot(theta).reshape(grid_shape)

            scatter3

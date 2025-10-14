import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, probplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objs as go

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

        st.header("Feature Scaling Options")
        scaling_method = st.selectbox(
            "Choose a feature scaling method:",
            ("None", "Standardization (Z-score)", "Min-Max Scaling", "Robust Scaling")
        )

        X_raw = data[features].values

        if scaling_method == "None":
            st.write("""
            **Pros:** No change to original data; preserves interpretability.  
            **Cons:** Can cause slow or unstable convergence with gradient descent; regularization may be biased if features scale differently.
            """)
            X_scaled = X_raw

        elif scaling_method == "Standardization (Z-score)":
            st.write("""
            **Pros:** Centers data around zero with unit variance; improves gradient descent convergence.  
            **Cons:** Sensitive to outliers; assumes Gaussian-like distributions.
            """)
            mean = np.mean(X_raw, axis=0)
            std = np.std(X_raw, axis=0)
            X_scaled = (X_raw - mean) / std

        elif scaling_method == "Min-Max Scaling":
            st.write("""
            **Pros:** Scales features to [0,1]; useful when fixed range is important.  
            **Cons:** Sensitive to outliers; may distort feature distributions.
            """)
            min_ = np.min(X_raw, axis=0)
            max_ = np.max(X_raw, axis=0)
            X_scaled = (X_raw - min_) / (max_ - min_)

        else:  # Robust Scaling
            st.write("""
            **Pros:** Uses median and IQR; robust to outliers.  
            **Cons:** Data not always centered; sometimes less efficient for well-behaved data.
            """)
            median = np.median(X_raw, axis=0)
            q1 = np.percentile(X_raw, 25, axis=0)
            q3 = np.percentile(X_raw, 75, axis=0)
            iqr = q3 - q1
            X_scaled = (X_raw - median) / iqr

        X = np.column_stack((np.ones(len(data)), X_scaled))
        y = data[target].values

        st.header("Manual Gradient Descent for Linear Regression")

        reg_type = st.selectbox(
            "Choose regularization:",
            ("None", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net")
        )

        if reg_type == "Elastic Net":
            alpha = st.number_input("Alpha (Regularization strength)", min_value=0.0, value=0.1, step=0.01, format="%.10f")
        else:
            alpha = 0.0

        recommended_lr = 0.01
        st.info(f"Recommended learning rate is about {recommended_lr} to avoid divergence or slow learning.")

        learning_rate = st.number_input("Learning rate", min_value=1e-10, value=recommended_lr, step=1e-7, format="%e")

        convergence_threshold = max(learning_rate * 1e-3, 1e-10)
        st.write(f"Automatically set convergence threshold = {convergence_threshold:.12f} (proportional to learning rate)")

        max_iterations = st.number_input("Maximum iterations", min_value=10, value=1000, step=10)

        theta = np.zeros(X.shape[1])

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

        previous_cost = float('inf')
        for i in range(1, int(max_iterations)+1):
            cost = compute_cost(X, y, theta, reg_type, alpha)
            
            # Print only every 10th iteration and last iteration
            if i % 10 == 0 or i == int(max_iterations):
                st.write(f"--- Iteration {i} ---")
                st.write(f"Current cost (loss): {cost:.12f}")

                param_desc = [f"Intercept (bias): {theta[0]:.6f}"]
                param_desc += [f"Weight of feature '{feat}': {theta[j+1]:.6f}" for j, feat in enumerate(features)]
                st.write("Parameters:")
                for desc in param_desc:
                    st.write(f"- {desc}")

                theta, grad = gradient_descent_step(X, y, theta, learning_rate, reg_type, alpha)

                grad_desc = [f"Gradient for Intercept (bias): {grad[0]:.6f}"]
                grad_desc += [f"Gradient for feature '{feat}': {grad[j+1]:.6f}" for j, feat in enumerate(features)]
                st.write("Gradient:")
                for desc in grad_desc:
                    st.write(f"- {desc}")

                st.write("Parameters updated by moving opposite the gradient to minimize the cost.\n")
            else:
                theta, grad = gradient_descent_step(X, y, theta, learning_rate, reg_type, alpha)

            cost_change = abs(previous_cost - cost)
            if cost_change < convergence_threshold:
                st.write(f"Convergence achieved: cost change {cost_change:.12f} < threshold {convergence_threshold:.12f}")
                st.write(f"Stopping gradient descent at iteration {i}.")
                break
            previous_cost = cost

        st.success("Gradient descent complete.")
        st.write("Final learned parameters:")
        param_desc = [f"Intercept (bias): {theta[0]:.6f}"] + [f"Weight of feature '{feat}': {theta[j+1]:.6f}" for j, feat in enumerate(features)]
        for desc in param_desc:
            st.write(f"- {desc}")

        st.write("------")
        # 3D Visualization for 2 features only
        if len(features) == 2:
            st.header("3D Visualization of Regression Plane and Data Points")

            x_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 30)
            y_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 30)
            xx, yy = np.meshgrid(x_range, y_range)

            zz = theta[0] + theta[1] * xx + theta[2] * yy

            scatter = go.Scatter3d(
                x=X_scaled[:, 0],
                y=X_scaled[:, 1],
                z=y,
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Data Points'
            )

            plane = go.Surface(
                x=xx,
                y=yy,
                z=zz,
                colorscale='Viridis',
                opacity=0.6,
                name='Regression Plane'
            )

            layout = go.Layout(
                scene=dict(
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    zaxis_title=target,
                ),
                width=800,
                height=600,
                margin=dict(l=0, r=0, b=0, t=0)
            )

            fig = go.Figure(data=[scatter, plane], layout=layout)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("3D visualization requires exactly 2 features selected.")

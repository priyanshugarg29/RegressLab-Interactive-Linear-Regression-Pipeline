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

# Other manual scalers omitted for brevity, use only Z-score default here for simplicity

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

        # Default to Z-score scaling
        st.header("Feature Scaling (default: Standardization / Z-score)")
        mean = np.mean(data[features].values, axis=0)
        std = np.std(data[features].values, axis=0)
        X_scaled = (data[features].values - mean) / std

        X = np.column_stack((np.ones(len(data)), X_scaled))
        y = data[target].values

        st.header("Manual Gradient Descent for Linear Regression")
        reg_type = "None"  # fixed, no regularization for simplicity here
        alpha = 0.0

        recommended_lr = 0.01
        st.info(f"Recommended learning rate is about {recommended_lr}.")

        learning_rate = st.number_input("Learning rate", min_value=1e-10, value=recommended_lr, step=1e-7, format="%e")
        convergence_threshold = max(learning_rate * 1e-3, 1e-10)
        st.write(f"Automatically set convergence threshold = {convergence_threshold:.12f} (proportional to learning rate)")
        max_iter = st.number_input("Maximum iterations", min_value=10, value=1000, step=10)

        theta = np.zeros(X.shape[1])

        def compute_cost(X, y, theta):
            m = len(y)
            errors = X.dot(theta) - y
            return (1/(2*m)) * np.sum(errors ** 2)

        def gradient_step(X, y, theta, lr):
            m = len(y)
            errors = X.dot(theta) - y
            grad = (1/m) * X.T.dot(errors)
            return theta - lr * grad, grad

        previous_cost = float('inf')
        for i in range(1, int(max_iter)+1):
            cost = compute_cost(X, y, theta)
            if i % 10 == 0 or i == int(max_iter):
                st.write(f"--- Iteration {i} --- Cost: {cost:.12f}")
                param_desc = [f"Intercept: {theta[0]:.6f}"] + [f"Weight '{feat}': {theta[j+1]:.6f}" for j, feat in enumerate(features)]
                st.write("Parameters:")
                for p in param_desc:
                    st.write(f"- {p}")
            theta, grad = gradient_step(X, y, theta, learning_rate)
            cost_change = abs(previous_cost - cost)
            if cost_change < convergence_threshold:
                st.write(f"Converged after {i} iterations (cost change {cost_change:.12f} < threshold {convergence_threshold:.12f})")
                break
            previous_cost = cost

        st.success("Gradient descent complete.")
        st.write("Final parameters:")
        for p in param_desc:
            st.write(f"- {p}")

        # Choose two features for 3D visualization
        st.header("3D Visualization of Hyperplane and Data Points")
        if len(features) < 2:
            st.info("Select at least two features for 3D visualization.")
        else:
            feature_pair = st.selectbox("Select feature 1 for 3D plot", features, index=0)
            features_exclude_1 = [f for f in features if f != feature_pair]
            feature_pair_2 = st.selectbox("Select feature 2 for 3D plot", features_exclude_1, index=0)

            # Fix other features at mean scaled values
            fixed_features_means = {f:0 for f in features}  # since scaled (Z-score), mean=0
            fixed_features_means.pop(feature_pair)
            fixed_features_means.pop(feature_pair_2)

            # Create grid for selected features
            f1_idx = features.index(feature_pair)
            f2_idx = features.index(feature_pair_2)

            f1_vals = np.linspace(X_scaled[:, f1_idx].min(), X_scaled[:, f1_idx].max(), 30)
            f2_vals = np.linspace(X_scaled[:, f2_idx].min(), X_scaled[:, f2_idx].max(), 30)
            xx, yy = np.meshgrid(f1_vals, f2_vals)

            # Prepare input matrix for prediction with fixed other features = 0 (mean)
            grid_shape = xx.shape
            X_grid = np.ones((xx.size, X.shape[1]))
            # set feature1 and feature2 column ( +1 for intercept column)
            X_grid[:, f1_idx + 1] = xx.ravel()
            X_grid[:, f2_idx + 1] = yy.ravel()

            # Other features fixed at 0 (mean after scaling)
            for feat, val in fixed_features_means.items():
                idx = features.index(feat) + 1
                X_grid[:, idx] = val

            zz = X_grid.dot(theta).reshape(grid_shape)

            # Scatter points for selected features and target
            scatter3d = go.Scatter3d(
                x = X_scaled[:, f1_idx],
                y = X_scaled[:, f2_idx],
                z = y,
                mode = 'markers',
                marker=dict(size=5, color='red'),
                name='Data points'
            )

            surface3d = go.Surface(
                x = xx,
                y = yy,
                z = zz,
                colorscale = 'Viridis',
                opacity=0.6,
                name='Regression Plane'
            )

            layout = go.Layout(
                scene = dict(
                    xaxis_title=feature_pair,
                    yaxis_title=feature_pair_2,
                    zaxis_title=target,
                ),
                width=800,
                height=600,
                margin=dict(l=0,r=0,b=0,t=0)
            )

            fig = go.Figure(data=[scatter3d, surface3d], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

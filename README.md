RegressLab: Interactive Linear Regression Pipeline

RegressLab is an interactive Streamlit web application designed to guide students, researchers, and practitioners through the end-to-end process of building and analyzing a linear regression model from tabular data.

Deployed Dashboard URL: https://regresslabinteractivelinearregressionpipeline.streamlit.app/

Features

- Data Upload & Quality Checks: Easily upload CSV datasets, with automatic detection and handling of missing values and duplicates.
- Exploratory Data Analysis: Visualize distributions, check skewness and kurtosis, and analyze correlations with intuitive plots.
- Outlier Treatment & Transformation: Interactive options to remove or clip outliers and apply transformations (log, square root) on selected variables.
- Customizable Feature Scaling: Multiple scaling options including Standardization (Z-score), Min-Max, and Robust scaling.
- Manual Gradient Descent: Core linear regression training implemented manually with support for L1, L2, Elastic Net regularization and convergence criteria.
- Train-Test Split & Evaluation: Split data into customizable training and testing sets with comprehensive manual evaluation metrics (MSE, RMSE, MAE, R², MAPE).
- Interactive Residual Analysis: Visualize predicted vs actual values with intuitive residual color mapping for both training and test sets.
- 3D Visualization: Explore the learned regression hyperplane visually for any pair of selected features, alongside actual data points from the training set.

Installation

1. Clone this repository:

git clone https://github.com/yourusername/regresslab.git
cd regresslab

2. Install required dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py

Usage

- Upload your dataset or choose the default sample dataset.
- Select your target and feature variables.
- Perform exploratory data analysis and apply transformations as needed.
- Select feature scaling and configure train-test split and training parameters.
- Train the model via manual gradient descent while monitoring the training loss and parameters.
- Visualize prediction performance with residual plots for both training and test data.
- Inspect the regression hyperplane fitted to training data through interactive 3D plots.

Notes

- This app is ideal for educational purposes and gaining an intuitive understanding of linear regression workflows.
- The codebase purposefully implements key ML components manually for transparency and learning.
- For production or large-scale applications, consider using optimized ML libraries.

Citation

If you use or refer to this project in your research or work, please cite it as:

Priyanshu Garg, RegressLab: Interactive Linear Regression Pipeline, GitHub repository, 2025.
https://github.com/priyanshugarg29/RegressLab-Interactive-Linear-Regression-Pipeline/

License

This project is licensed under the MIT License.

import streamlit as st
import pandas as pd

st.title("RegressLab: Interactive Linear Regression Pipeline")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("First five rows of your data:")
    st.write(data.head())

st.write("DATASET SUMMARY")
columns = data.columns.to_list()
st.write("[dataset columns extracted to list]")
st.write(f"THE DATASET HAS {len(columns)} COLUMNS: {columns}")

st.write("[target and feature selection"])
target = st.selectbox('Select target variable', columns)
features = st.multiselect('Select feature columns', columns, default=[col for col in columns if col != target])

st.write(f"Selected target variable: {target}, Data_type: {type(target)} (should be <str>)")
st.write(f"Selected {len(features)} feature columns: {features}, Data_type: {type(features)} (should be <list>)")


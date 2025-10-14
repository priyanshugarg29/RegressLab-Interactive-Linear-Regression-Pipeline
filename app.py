import streamlit as st
import pandas as pd

st.title("RegressLab: Interactive Linear Regression Pipeline")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("First five rows of your data:")
    st.write(data.head())

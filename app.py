import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------
# Load Files
# -------------------------------
customer_model = pickle.load(open("customer_segment_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
similarity_df = pickle.load(open("product_similarity.pkl", "rb"))
cluster_labels = pickle.load(open("cluster_labels.pkl", "rb"))
df = pd.read_csv("cleaned_online_retail.csv")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.title("🛒 Shopper Spectrum")

    st.subheader("Customer Segmentation and Product Recommendation System")

    st.write("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", df["CustomerID"].nunique())
    col2.metric("Products", df["Description"].nunique())
    col3.metric("Transactions", df["InvoiceNo"].nunique())

    st.write("---")

    st.markdown("""
### Project Overview

This project analyzes customer purchasing behaviour using the Online Retail dataset.

### Objectives

- Customer Segmentation using RFM Analysis
- K-Means Clustering
- Product Recommendation using Collaborative Filtering
- Interactive Streamlit Application
""")

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    recency = st.number_input("Recency (Days)", min_value=0.0)

    frequency = st.number_input("Frequency (Purchases)", min_value=0.0)

    monetary = st.number_input("Monetary (Total Spend)", min_value=0.0)

    if st.button("Predict Customer Segment"):

        data = np.array([[recency, frequency, monetary]])

        scaled = scaler.transform(data)

        cluster = customer_model.predict(scaled)[0]

        segment = cluster_labels[cluster]

        st.success(f"Customer Segment : **{segment}**")

# =====================================================
# PRODUCT RECOMMENDATION
# =====================================================

else:

    st.title("🛍 Product Recommendation")

    product = st.text_input("Enter Product Name")

    if st.button("Recommend Products"):

        product = product.upper()

        if product in similarity_df.index:

            recommendations = similarity_df[product].sort_values(
                ascending=False
            )[1:6]

            st.subheader("Recommended Products")

            for item in recommendations.index:
                st.write("✅", item)

        else:

            st.error("Product not found.")

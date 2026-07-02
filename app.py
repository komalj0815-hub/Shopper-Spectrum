import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ----------------------------------------------------
# Load Files
# ----------------------------------------------------

customer_model = pickle.load(open("customer_segment_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
cluster_labels = pickle.load(open("cluster_labels.pkl", "rb"))

df = pd.read_csv("streamlit_data.csv")

# ----------------------------------------------------
# Create Similarity Matrix Automatically
# ----------------------------------------------------

customer_product = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

product_matrix = customer_product.T

similarity = cosine_similarity(product_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=product_matrix.index,
    columns=product_matrix.index
)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Product Recommendation"
    ]
)

# ====================================================
# HOME
# ====================================================

if page == "Home":

    st.title("🛒 Shopper Spectrum")

    st.subheader("Customer Segmentation and Product Recommendation")

    st.write("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", df["CustomerID"].nunique())
    col2.metric("Products", df["Description"].nunique())
    col3.metric("Transactions", df["InvoiceNo"].nunique())

    st.write("---")

    st.markdown("""
## Project Overview

This project analyzes customer purchasing behaviour using the Online Retail Dataset.

### Objectives

- Customer Segmentation using RFM Analysis
- K-Means Clustering
- Product Recommendation using Collaborative Filtering
- Interactive Streamlit Application
""")

# ====================================================
# CUSTOMER SEGMENTATION
# ====================================================

elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write("Enter the customer's RFM values.")

    recency = st.number_input(
        "Recency (Days)",
        min_value=0.0,
        value=10.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0.0,
        value=5.0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        values = np.array([[recency, frequency, monetary]])

        scaled_values = scaler.transform(values)

        cluster = customer_model.predict(scaled_values)[0]

        segment = cluster_labels[cluster]

        st.success(f"Predicted Segment: **{segment}**")

# ====================================================
# PRODUCT RECOMMENDATION
# ====================================================

else:

    st.title("🛍 Product Recommendation")

    product = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):

        product = product.upper().strip()

        products_upper = {
            p.upper(): p for p in similarity_df.index
        }

        if product in products_upper:

            original_name = products_upper[product]

            recommendations = similarity_df.loc[original_name] \
                .sort_values(ascending=False)[1:6]

            st.subheader("Recommended Products")

            for item in recommendations.index:
                st.write("✅", item)

        else:

            st.error("Product not found.")

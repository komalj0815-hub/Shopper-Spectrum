# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendation

## 📌 Project Overview

Shopper Spectrum is a machine learning project that analyzes customer purchasing behavior using the Online Retail dataset. The project performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. It also recommends similar products using an item-based collaborative filtering approach.

An interactive Streamlit web application is developed to allow users to predict customer segments and receive product recommendations.

---

## 🎯 Objectives

- Clean and preprocess retail transaction data
- Perform Exploratory Data Analysis (EDA)
- Generate RFM features for customer analysis
- Segment customers using K-Means Clustering
- Build a product recommendation system using Cosine Similarity
- Develop an interactive Streamlit web application

---

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Shopper_Spectrum/
│
├── app.py
├── requirements.txt
├── streamlit_data.csv
├── customer_segment_model.pkl
├── scaler.pkl
├── cluster_labels.pkl
└── README.md
```

---

## 📊 Machine Learning Techniques

### Customer Segmentation
- RFM Analysis
- Feature Scaling using StandardScaler
- K-Means Clustering
- Elbow Method
- Silhouette Score

### Product Recommendation
- Customer-Product Matrix
- Item-Based Collaborative Filtering
- Cosine Similarity

---

## 📈 Exploratory Data Analysis

The following analyses were performed:

- Transaction volume by country
- Top-selling products
- Monthly sales trend
- Daily sales trend
- Transaction amount distribution
- Quantity distribution
- Unit price distribution
- Top customers by spending
- Revenue by country
- Correlation heatmap

---

## 💻 Streamlit Application

The application contains three modules:

### 🏠 Home
- Project overview
- Dataset statistics
- Project objectives

### 👥 Customer Segmentation
Users enter:
- Recency
- Frequency
- Monetary

The application predicts the customer segment:

- High-Value
- Regular
- Occasional
- At-Risk

### 🛍 Product Recommendation

Users enter a product name.

The application recommends the five most similar products using collaborative filtering.

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/komalj0815-hub/Shopper_Spectrum.git
```

Move into the project folder:

```bash
cd Shopper_Spectrum
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📌 Dataset

Dataset: Online Retail Dataset

The dataset contains retail transaction records including:

- Invoice Number
- Product Description
- Quantity
- Unit Price
- Customer ID
- Country
- Invoice Date

---

## 📷 Results

The project successfully:

- Cleaned and preprocessed the dataset
- Performed customer segmentation
- Identified customer purchasing patterns
- Generated personalized product recommendations
- Developed an interactive Streamlit application

---

## 👨‍💻 Author

**Komal**

Aspiring Data Analyst

Skills:
- Python
- SQL
- Excel
- Power BI
- Machine Learning
- Streamlit

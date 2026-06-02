import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    layout="wide"
)

st.title("📊 Predictive Analytics Dashboard")

st.markdown("""
### Online Retail Sales & Customer Analytics

This dashboard provides insights into:
- Revenue Performance
- Customer Behaviour
- Product Performance
- Predictive Analytics
""")

#DATA CLEANING

df = pd.read_excel(
    "Online Retail Data 2009 To 2011.xlsb",
    engine="pyxlsb"
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    unit="D",
    origin="1899-12-30"
)

# df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df["Revenue"] = df["Quantity"] * df["Price"]

# SIDEBAR

st.sidebar.markdown("## 🎛️ Filter Data")
st.sidebar.markdown("---")

# Country Filter

selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["Country"].unique().tolist())
)

# Date Filter

start_date = st.sidebar.date_input(
    "Start Date",
    df["InvoiceDate"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    df["InvoiceDate"].max().date()
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
    Records: {len(df):,}
    
    Countries: {df['Country'].nunique()}
    """
)

# Apply Country Filter

if selected_country != "All":
    df = df[df["Country"] == selected_country]

# Apply Date Filter

df = df[
    (df["InvoiceDate"].dt.date >= start_date)
    &
    (df["InvoiceDate"].dt.date <= end_date)
]

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
Records: {len(df):,}

Countries: {df['Country'].nunique()}

Revenue: £{df['Revenue'].sum():,.0f}
"""
) 
# KPI CALCULATIONS

total_revenue = df["Revenue"].sum()

total_orders = df["Invoice"].nunique()

total_customers = df["Customer ID"].nunique()

avg_order_value = total_revenue / total_orders

# KPI DISPLAY

with st.container():

    st.markdown("## 📊 Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "💰 Revenue",
    f"£{total_revenue:,.0f}"
)

col2.metric(
    "🛒 Orders",
    f"{total_orders:,}"
)

col3.metric(
    "👥 Customers",
    f"{total_customers:,}"
)

col4.metric(
    "📦 Avg Order",
    f"£{avg_order_value:,.2f}"
)
monthly_sales = (
    df.groupby(
        pd.Grouper(
            key="InvoiceDate",
            freq="ME"
        )
    )["Revenue"]
    .sum()
    .reset_index()
)
st.header("📈 Sales Analysis")

plt.figure(figsize=(12,6))

plt.plot(
    monthly_sales["InvoiceDate"],
    monthly_sales["Revenue"]
)

st.markdown("## 📈 Monthly Revenue Trend")
plt.title(
    "Monthly Revenue Trend",
    fontsize=16
)
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.grid(True)

st.pyplot(plt.gcf())
plt.close()
st.divider()

#TOP PRODUCTS

st.markdown("## 🏆 Top Products")
top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

top_products.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Product")

st.pyplot(plt.gcf())
plt.close()

#TOP CUSTOMERS

st.subheader("👥 Top Customers")
top_customers = (
    df.groupby("Customer ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


plt.figure(figsize=(10,6))

top_customers.plot(
    kind="bar"
)

plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)

st.pyplot(plt.gcf())
plt.close()

# Revenue By Country
st.markdown("## 🌍 Revenue by Country")
country_sales = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)


plt.figure(figsize=(12,6))

country_sales.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 Countries by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Country")

st.pyplot(plt.gcf())
plt.close()

# CUSTOMER REVENUE PREDICTION

customer_data = (
    df.groupby("Customer ID")
    .agg({
        "Revenue": "sum",
        "Quantity": "sum",
        "Invoice": "nunique"
    })
    .reset_index()
)

customer_data.columns = [
    "CustomerID",
    "Revenue",
    "Quantity",
    "Orders"
]


X = customer_data[
    ["Quantity", "Orders"]
]

y = customer_data["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")

st.pyplot(plt.gcf())
plt.close()

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})


plt.figure(figsize=(6,4))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance")
plt.ylabel("Importance Score")

st.pyplot(plt.gcf())
plt.close()

clv = (
    df.groupby("Customer ID")
    .agg({
        "Revenue":"sum",
        "Invoice":"nunique"
    })
    .reset_index()
)

clv["Average_Order_Value"] = (
    clv["Revenue"] /
    clv["Invoice"]
)


plt.figure(figsize=(10,6))

plt.hist(
    clv["Revenue"],
    bins=50
)

plt.title("Customer Lifetime Value Distribution")
plt.xlabel("Revenue (£)")
plt.ylabel("Number of Customers")

st.pyplot(plt.gcf())
plt.close()


forecast_data = (
    df.groupby(
        pd.Grouper(
            key="InvoiceDate",
            freq="ME"
        )
    )["Revenue"]
    .sum()
    .reset_index()
)

forecast_data.columns = [
    "ds",
    "y"
]

import matplotlib.pyplot as plt

corr_df = df[
    ["Quantity", "Price", "Revenue"]
].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr_df,
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

st.pyplot(plt.gcf())
plt.close()

customer_revenue = (
    df.groupby("Customer ID")["Revenue"]
    .sum()
    .reset_index()
)

customer_revenue["Segment"] = pd.qcut(
    customer_revenue["Revenue"],
    q=3,
    labels=[
        "Low Value",
        "Medium Value",
        "High Value"
    ]
)

customer_revenue["Segment"].value_counts().plot(
    kind="bar"
)

plt.title("Customer Segments")

st.pyplot(plt.gcf())
plt.close()


customer_revenue.to_excel(
    "customer_segmentation.xlsx",
    index=False
)

clv.to_excel(
    "customer_lifetime_value.xlsx",
    index=False
)

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_predictions)

models = ["Linear Regression", "Random Forest"]
scores = [lr_r2, r2]

plt.figure(figsize=(6,4))
plt.bar(models, scores)

plt.ylabel("R² Score")
plt.title("Model Comparison")
st.pyplot(plt.gcf())
plt.close()
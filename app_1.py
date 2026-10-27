import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Page configuration
st.set_page_config(page_title="Sales Analysis Dashboard", page_icon=":bar_chart:", layout="wide")

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Data Loading with Caching
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
    return df

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Filter Options")

city = st.sidebar.multiselect(
    "Select the City:",
    options=df["City"].unique(),
    default=df["City"].unique()
)

customer_type = st.sidebar.multiselect(
    "Select Customer Type:",
    options=df["Customer type"].unique(),
    default=df["Customer type"].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

# Apply Filters
df_selection = df.query(
    "City == @city & `Customer type` == @customer_type & Gender == @gender"
)

# Date Filter in Sidebar
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Further filter by date
df_selection = df_selection[
    (df_selection["Date"].dt.date >= start_date) & 
    (df_selection["Date"].dt.date <= end_date)
]

# --- MAINPAGE ---
st.title(":bar_chart: Sales Analysis Dashboard")
st.markdown("##")

# TOP KPI's
total_sales = int(df_selection["Total"].sum())
average_rating = round(df_selection["Rating"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_sale_by_transaction = round(df_selection["Total"].mean(), 2)
total_items_sold = int(df_selection["Quantity"].sum())

left_column, middle_column, right_column, extra_column = st.columns(4)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
with middle_column:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")
with right_column:
    st.subheader("Avg Transaction:")
    st.subheader(f"US $ {average_sale_by_transaction}")
with extra_column:
    st.subheader("Total Items Sold:")
    st.subheader(f"{total_items_sold:,}")

st.markdown("""---""")

# SALES BY PRODUCT LINE [BAR CHART]
sales_by_product_line = (
    df_selection.groupby("Product line")["Total"]
    .sum()
    .sort_values()
)

# SALES BY HOUR [LINE CHART]
sales_by_hour = (
    df_selection.groupby("Hour")[["Total"]]
    .sum()
)

# PLOTTING WITH MATPLOTLIB/SEABORN
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("### Sales by Product Line")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=sales_by_product_line.values,
        y=sales_by_product_line.index,
        palette="viridis",
        ax=ax
    )
    ax.set_xlabel("Total Sales (USD)")
    ax.set_ylabel("Product Line")
    st.pyplot(fig)

with fig_col2:
    st.markdown("### Sales Trends by Hour")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=sales_by_hour.index,
        y=sales_by_hour["Total"],
        marker='o',
        color='#2ecc71',
        ax=ax
    )
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total Sales (USD)")
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# ADDITIONAL BREAKDOWNS
st.markdown("### Demographic & Categorical Breakdowns")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Gender Distribution")
    fig, ax = plt.subplots()
    gender_counts = df_selection["Gender"].value_counts()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)

with col2:
    st.markdown("#### Customer Type")
    fig, ax = plt.subplots()
    cust_counts = df_selection["Customer type"].value_counts()
    ax.pie(cust_counts, labels=cust_counts.index, autopct='%1.1f%%', startangle=140, colors=['#99ff99','#ffcc99'])
    ax.axis('equal')
    st.pyplot(fig)

with col3:
    st.markdown("#### Payment Method")
    fig, ax = plt.subplots()
    pay_counts = df_selection["Payment"].value_counts()
    ax.pie(pay_counts, labels=pay_counts.index, autopct='%1.1f%%', startangle=140, colors=['#c2c2f0','#ffb3e6', '#c4ffc4'])
    ax.axis('equal')
    st.pyplot(fig)

# EXPORT ENGINE
st.markdown("---")
st.subheader("Data Export")
csv = df_selection.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv',
)

# Display raw data if requested
if st.checkbox("Show Raw Filtered Data"):
    st.dataframe(df_selection)

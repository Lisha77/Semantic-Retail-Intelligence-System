import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Retail Insights Pro", page_icon="📈", layout="wide")

# Professional Styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #007bff; }
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_and_clean_data():
    # Attempt to read local file, else use the generated one
    try:
        df = pd.read_csv("supermarket_sales.csv")
    except:
        st.error("Dataset not found. Please run the data generator first.")
        st.stop()
    
    # Ensure standard Kaggle types
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
    return df

df = load_and_clean_data()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=100)
st.sidebar.title("Control Panel")

with st.sidebar.expander("🌍 Geographic & Customer Filters", expanded=True):
    city = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
    customer = st.sidebar.multiselect("Customer Type", options=df["Customer type"].unique(), default=df["Customer type"].unique())
    gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique(), default=df["Gender"].unique())

with st.sidebar.expander("📅 Timeframe"):
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    start_date, end_date = st.sidebar.date_input("Date Range", [min_d, max_d])

# Filter logic
mask = (df["City"].isin(city)) & \
       (df["Customer type"].isin(customer)) & \
       (df["Gender"].isin(gender)) & \
       (df["Date"].dt.date >= start_date) & \
       (df["Date"].dt.date <= end_date)

df_selection = df.loc[mask]

# --- MAIN HEADER ---
st.title("📊 Retail Insights Pro: Strategic Dashboard")
st.markdown(f"**Data Scope:** {start_date} to {end_date} | **Active Records:** {len(df_selection)}")

# --- TOP METRICS (DYNAMIC) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Revenue", f"${df_selection['Total'].sum():,.2f}")
kpi2.metric("Gross Profit", f"${df_selection['Gross income'].sum():,.2f}", delta=f"{df_selection['Gross margin percentage'].iloc[0]:.2f}%")
kpi3.metric("Avg Rating", f"{df_selection['Rating'].mean():.1f} / 10")
kpi4.metric("Unit Volume", f"{df_selection['Quantity'].sum():,}")

st.markdown("---")

# --- NOVEL FEATURE: DYNAMIC BUSINESS QUESTIONS ---
st.header("🧠 Smart Analysis Playground")
st.info("Select a business question below to generate dynamic insights and recommendations.")

question = st.selectbox(
    "What would you like to investigate?",
    [
        "Product Performance: Which lines are driving volume?",
        "Payment Insights: How do customers prefer to pay?",
        "Gender Dynamics: Is there a gender bias in product categories?",
        "Branch Efficiency: Revenue vs Gross Income comparison",
        "Customer Satisfaction: Rating analysis for improvement",
        "Financial Health: Profit vs Loss (COGS) review"
    ]
)

q_col1, q_col2 = st.columns([1, 1])

if question == "Product Performance: Which lines are driving volume?":
    data = df_selection.groupby("Product line")["Quantity"].sum().sort_values(ascending=False)
    with q_col1:
        st.subheader("Analysis: Quantity Sold by Category")
        chart_type = st.radio("Choose Visualization (Recommended: Bar Chart)", ["Bar Chart", "Pie Chart"])
        if chart_type == "Bar Chart":
            fig, ax = plt.subplots()
            sns.barplot(x=data.values, y=data.index, palette="magma", ax=ax)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=sns.color_palette("magma"))
            st.pyplot(fig)
    with q_col2:
        st.success("**Business Conclusion:**")
        top_cat = data.index[0]
        st.write(f"The **{top_cat}** category is currently your volume leader with {data.max()} units sold.")
        st.write("👉 *Recommendation:* Consider bundling lower-volume items with these leaders to clear inventory.")

elif question == "Payment Insights: How do customers prefer to pay?":
    data = df_selection["Payment"].value_counts()
    with q_col1:
        st.subheader("Analysis: Payment Method Share")
        chart_type = st.radio("Choose Visualization (Recommended: Pie Chart)", ["Pie Chart", "Bar Chart"])
        if chart_type == "Pie Chart":
            fig, ax = plt.subplots()
            ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=["#ff9999","#66b3ff","#99ff99"])
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.barplot(x=data.index, y=data.values, palette="Set2", ax=ax)
            st.pyplot(fig)
    with q_col2:
        st.success("**Business Conclusion:**")
        pref_pay = data.index[0]
        st.write(f"Customers primarily use **{pref_pay}**.")
        st.write(f"This accounts for {data.max()/data.sum()*100:.1f}% of all transactions.")

elif question == "Gender Dynamics: Is there a gender bias in product categories?":
    data = df_selection.groupby(["Product line", "Gender"])["Total"].count().unstack()
    with q_col1:
        st.subheader("Gender Distribution per Category")
        fig, ax = plt.subplots(figsize=(10, 6))
        data.plot(kind='barh', stacked=True, ax=ax, color=['#ffb3e6', '#c2c2f0'])
        st.pyplot(fig)
    with q_col2:
        st.success("**Business Conclusion:**")
        st.write("This visualization highlights target demographics for specific product lines.")
        st.write("👉 *Action:* Use this to optimize your marketing spend. If 'Fashion accessories' is skewed female, target ads accordingly.")

elif question == "Branch Efficiency: Revenue vs Gross Income comparison":
    data = df_selection.groupby("Branch")[["Total", "Gross income"]].sum()
    with q_col1:
        st.subheader("Revenue vs Profit by Branch")
        fig, ax = plt.subplots()
        data.plot(kind='bar', ax=ax)
        st.pyplot(fig)
    with q_col2:
        st.success("**Business Conclusion:**")
        best_branch = data['Gross income'].idxmax()
        st.write(f"Branch **{best_branch}** is the most profitable in this selection.")
        st.write(f"Margin Efficiency: {(data.loc[best_branch, 'Gross income'] / data.loc[best_branch, 'Total'])*100:.2f}%")

elif question == "Customer Satisfaction: Rating analysis for improvement":
    avg_rating = df_selection.groupby("Product line")["Rating"].mean().sort_values()
    with q_col1:
        st.subheader("Avg Rating by Product Line")
        fig, ax = plt.subplots()
        sns.barplot(x=avg_rating.values, y=avg_rating.index, palette="RdYlGn", ax=ax)
        ax.set_xlim(0, 10)
        st.pyplot(fig)
    with q_col2:
        st.warning("**Improvement Area:**")
        worst_cat = avg_rating.index[0]
        st.write(f"The **{worst_cat}** category has the lowest satisfaction ({avg_rating.min():.2f}).")
        st.write("👉 *Action:* Review supplier quality or customer feedback for this specific category immediately.")

elif question == "Financial Health: Profit vs Loss (COGS) review":
    total_rev = df_selection["Total"].sum()
    total_cogs = df_selection["COGS"].sum()
    with q_col1:
        st.subheader("Cost of Goods Sold vs Revenue")
        fig, ax = plt.subplots()
        ax.pie([total_cogs, total_rev - total_cogs], labels=["COGS (Costs)", "Gross Profit"], autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'])
        st.pyplot(fig)
    with q_col2:
        st.success("**Financial Overview:**")
        st.write(f"Total Costs: ${total_cogs:,.2f}")
        st.write(f"Total Profit: ${(total_rev - total_cogs):,.2f}")
        st.write("The business is maintaining a healthy margin according to the Kaggle standard (approx 4.76%).")

st.markdown("---")

# --- FOOTER & EXPORT ---
col_f1, col_f2 = st.columns(2)
with col_f1:
    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df_selection)
with col_f2:
    csv = df_selection.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Report (CSV)", data=csv, file_name='retail_report.csv', mime='text/csv')
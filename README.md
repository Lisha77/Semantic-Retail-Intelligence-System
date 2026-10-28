🧠 Semantic Retail Intelligence Engine

Overview

Traditional business intelligence (BI) dashboards suffer from static rigidity: users are confined to pre-defined filters and fixed charts, and the moment a column is renamed, the dashboard breaks.

The Semantic Retail Intelligence Engine is an AI-augmented, web-based system that bridges raw retail data and non-technical stakeholders. It translates natural-language questions (e.g., "Which city has the highest gross income?") into precise pandas operations and professional Plotly visualizations—without requiring any SQL or hard-coded column references.

The engine performs runtime schema discovery on any structured retail CSV, utilising a hybrid of exact whole-word matching, character n-gram TF-IDF cosine similarity, and NLTK WordNet synonym expansion to interpret user intent.

✨ Key Features

Natural Language Querying: Ask business questions in plain English.

Zero Hard-coding (Schema Discovery): Automatically classifies columns as metric, dimension, or temporal data at load time.

Explainable AI & Auditability:

Logic Trace: Displays the resolved metric, aggregation, and filters.

Evidence Trace: Shows similarity scores for top matched labels.

Mathematical Proof: Renders literal arithmetic via LaTeX for absolute transparency.

Dynamic Visualizations: Automatically recommends and renders the best chart type (Bar, Line, Pie, Histogram, Scatter, Box, or Bee-swarm).

Comparison Mode: Splits "X versus Y" queries into two independent sub-questions and computes percentage-delta comparisons.

Robust Data Ingestion: Load datasets seamlessly via KaggleHub, direct URL, or local CSV upload.

Export Capabilities: One-click CSV export of filtered data and analysis outputs.

🛠️ Technology Stack

Frontend Framework: Streamlit

Data Manipulation: Pandas, NumPy

Visualizations: Plotly Express, Matplotlib, Seaborn

Text Mining & NLP: scikit-learn (TF-IDF Vectorizer, Cosine Similarity), NLTK (WordNet)

Data Fetching: KaggleHub

⚙️ Architecture Pipeline

The backend relies on a clear, five-stage analytical pipeline:

Stage 0 (Data Layer): Data load and automatic schema discovery.

Stage 1 (Reasoning): Hybrid text mining, thesaurus expansion, and search index building.

Stage 2 (Reasoning): Intent, aggregation, and chart hint mining to form a logical query plan.

Stage 3 & 4 (Analytics): Pandas execution, runnable code generation, LaTeX proofing, and chart rendering.

Stage 5 (Presentation): Plain-text interpretation and conclusion generation.

🚀 Getting Started

Follow these steps to run the application locally on your machine.

Prerequisites

Python 3.13 or compatible version installed.

Installation & Execution

Clone the repository:

git clone https://github.com/Lisha77/Semantic-Retail-Intelligence-System.git
cd Semantic-Retail-Intelligence-System


Create and activate a virtual environment:

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app.py


📊 Example Queries to Try

Once the dashboard is running, try typing these queries into the input box:

"total sales by product line" (Renders sum of Total per product line as a Bar chart)

"how many different payment methods are used in yangon" (Renders distinct count as a Pie chart)

"sales of yangon versus mandalay" (Triggers comparison mode with % delta)

"average rating by branch" (Renders average Rating per branch)

Developed as a Python Programming Mini Project at Chandigarh University.

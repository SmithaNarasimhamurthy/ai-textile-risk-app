import streamlit as st
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
st.markdown(
    """
    <style>
    /* Very light blue background for whole app */
    .stApp {
        background-color: #dbe9ff;  /* very light blue */
        color: black;
    }

    /* Sidebar background pink with border */
    .css-1d391kg {
        background-color: #ffe6f0 !important;  /* very light pink */
        border: 2px solid #ff66b2;  /* pink border */
        border-radius: 10px;
        padding: 10px;
    }

    /* Buttons with black text and subtle background */
    .stButton>button {
        color: black;
        background-color: #f0f5ff;
        font-weight: bold;
    }

    /* Headers and text in black */
    h1, h2, h3, h4, p, div, span {
        color: black !important;
    }

    /* Dataframe text color */
    .stDataFrame div[data-testid="stMarkdownContainer"] {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- SETTINGS ---
SUPPLIERS = {
    "Welspun Living Limited": ["welspun"],
    "Teejay Lanka PLC": ["teejay"],
    "Arvind Limited": ["arvind"],
    "Caleres, Inc.": ["caleres"],
    "Interloop Limited": ["interloop"],
    "Kitex Garments Limited": ["kitex"],
    "ThredUp Inc.": ["thredup"],
    "G-III Apparel Group, Ltd.": ["g-iii", "giii"],
    "Mint Velvet": ["mint velvet"],
    "White Stuff Limited": ["white stuff"]
}

RISK_CATEGORIES = {
    "Geopolitical and Regulatory": ["tariff", "ban", "sanction", "regulation", "license", "policy", "compliance", "export", "import", "law"],
    "Agricultural and Environmental":["climate", "drought", "flood", "fire", "emission", "emissions", "pollution", "pollutant", "weather","wildfire", "deforestation", "environment", "environmental", "ecology"],
    "Financial and Operational": ["loss", "bankruptcy", "strike", "shutdown", "layoff", "debt", "lawsuit", "cost", "profit", "finance"],
    "Supply Chain and Logistics": ["delay", "transport", "logistics", "shipping", "port", "customs", "shortage", "warehouse", "distribution"],
    "Market and Competitive": ["price", "competitor", "demand", "trend", "fashion", "revenue", "market", "sale"]
}


# --- FUNCTIONS ---
@st.cache_data
def load_data():
    with open("Cleaned_Articles_2023_2024.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def match_supplier(article_text):
    matches = []
    for name, aliases in SUPPLIERS.items():
        for keyword in aliases:
            if re.search(rf"\b{re.escape(keyword)}\b", article_text, re.IGNORECASE):
                matches.append(name)
                break
    return matches

def get_sentiment_direction(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity >= 0 else "Negative"

@st.cache_data(show_spinner=False)
def train_risk_classifier(df):
    mlb = MultiLabelBinarizer(classes=list(RISK_CATEGORIES.keys()))
    Y = mlb.fit_transform(df['Risk_Tags'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['Full_Article'], Y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000)))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)
    print("Classification Report:\n", report)

    return pipeline, mlb

@st.cache_data
def load_categorization_data():
    with open("categorization.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def extract_risks_ml(text, pipeline, mlb):
    pred = pipeline.predict([text])
    return list(np.array(mlb.classes_)[pred[0] == 1])

# --- MAIN APP ---
# 1. --- START SCREEN LOGIC ---
if 'start_app' not in st.session_state:
    st.session_state.start_app = False

if not st.session_state.start_app:
    st.image("images/textile.jpeg", use_container_width=True)
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 5px;'>
            <h1 style='color: #1f77b4;'>Textile Supply Chain Risk Analyzer</h1>
            <p style='color: #333;'> using <b>Machine Learning</b> and <b>Natural Language processing</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("🚀 Start Analysis"):
        st.session_state.start_app = True
    st.stop()

# 2. --- MAIN APP CONTENT (after start button is clicked) ---
st.sidebar.title("Navigation")


# --- MAIN TITLE ---
st.title("📊 Textile Supplier Risk Classifier (2023-2024)")
st.markdown("Analyzing news to assess supply chain risks using machine learning")


df = load_data()
df['Supplier'] = df['Full_Article'].apply(match_supplier)
df = df[df['Supplier'].map(len) > 0].explode("Supplier")
df['published_datetime_utc'] = pd.to_datetime(df['published_datetime_utc'], errors='coerce')
df['Year'] = df['published_datetime_utc'].dt.year

st.markdown("""
    <h1 style='text-align: center; color: #1f4e79;'>Textile Supply Chain Risk Analyzer</h1>
    <h4 style='text-align: center;'>Using Machine Learning and Natural Language Processing</h4>
    <hr>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
view_mode = st.sidebar.radio("Choose View Mode:", ["Overview", "Analyze Supplier", "AI Categorization"], key="view_mode_radio")

supplier_filter = st.sidebar.multiselect(
    "Select Supplier(s):", sorted(df['Supplier'].unique()), 
    disabled=(view_mode == "Overview"),
    key="supplier_multiselect"
)

risk_filter = st.sidebar.multiselect(
    "Select Risk Category(s):", list(RISK_CATEGORIES.keys()), 
    help="Optional risk category filter",
    key="risk_multiselect"
)

st.sidebar.markdown("---")

# --- ANALYSIS ---

def extract_risks_keyword(text):
    risks = []
    for category, keywords in RISK_CATEGORIES.items():
        if any(re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE) for k in keywords):
            risks.append(category)
    return risks

def match_supplier_in_text(text):
    """Check if article text mentions any suppliers and return list of matched suppliers"""
    matches = []
    for name, aliases in SUPPLIERS.items():
        for keyword in aliases:
            if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                matches.append(name)
                break
    return matches

df['Risk_Tags'] = df['Full_Article'].apply(extract_risks_keyword)
df = df[df['Risk_Tags'].map(len) > 0]
df = df.explode("Risk_Tags")
df_grouped = df.groupby(['title', 'published_datetime_utc', 'Full_Article', 'Supplier', 'Year'])['Risk_Tags'].apply(list).reset_index()

with st.spinner("Training risk classification model..."):
    risk_pipeline, mlb = train_risk_classifier(df_grouped)

df_grouped['Risk_Tags_Predicted'] = df_grouped['Full_Article'].apply(lambda x: extract_risks_ml(x, risk_pipeline, mlb))
df_expanded = df_grouped.explode('Risk_Tags_Predicted')
df_expanded['Risk_Direction'] = df_expanded['Full_Article'].apply(get_sentiment_direction)

if risk_filter:
    df_expanded = df_expanded[df_expanded['Risk_Tags_Predicted'].isin(risk_filter)]

if view_mode == "AI Categorization":
    # Load categorization data
    categorization_df = load_categorization_data()
    
    # Add supplier detection to categorization data
    categorization_df['suppliers_mentioned'] = categorization_df['article_title'].apply(match_supplier_in_text)
    
    # Apply supplier filter if selected
    if supplier_filter:
        categorization_df = categorization_df[categorization_df['suppliers_mentioned'].apply(
            lambda x: any(supplier in x for supplier in supplier_filter)
        )]
    
    # Explode risk categories to have one row per category (for visualizations)
    categorization_expanded = categorization_df.explode("risk_categories")
    categorization_expanded = categorization_expanded[categorization_expanded['risk_categories'].notna()]
    categorization_expanded = categorization_expanded[categorization_expanded['risk_categories'] != '']
    
    # Apply risk filter if selected
    filtered_cat_data = categorization_expanded.copy()
    if risk_filter:
        filtered_cat_data = filtered_cat_data[filtered_cat_data['risk_categories'].isin(risk_filter)]
    
    # For display: show unique articles with their risk categories as a string
    display_df = categorization_df.copy()
    if risk_filter:
        # Filter to only show articles that have at least one of the selected risk categories
        display_df = display_df[display_df['risk_categories'].apply(
            lambda x: any(cat in risk_filter for cat in (x if isinstance(x, list) else []))
        )]
    
    # Convert risk_categories list to comma-separated string for display
    display_df = display_df.copy()
    display_df['risk_categories_str'] = display_df['risk_categories'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else ''
    )
    display_df['suppliers_str'] = display_df['suppliers_mentioned'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) and len(x) > 0 else 'No suppliers mentioned'
    )
    
    st.markdown(f"### 📌 AI Categorized Articles: {len(display_df)}")
    st.dataframe(display_df[['article_title', 'suppliers_str', 'risk_categories_str']].rename(
        columns={'article_title': 'Article Title', 'suppliers_str': 'Suppliers Mentioned', 'risk_categories_str': 'Risk Categories'}
    ))

    st.markdown("### 📈 Risk Category Distribution")
    risk_counts = filtered_cat_data['risk_categories'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Count",
        color="Risk Category",
        height=400,
        title="Risk Category Distribution in AI Categorized Articles"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Add supplier distribution chart if suppliers are mentioned
    if len(categorization_df[categorization_df['suppliers_mentioned'].map(len) > 0]) > 0:
        st.markdown("### 🏢 Supplier Mentions Distribution")
        supplier_expanded = categorization_df.explode('suppliers_mentioned')
        supplier_expanded = supplier_expanded[supplier_expanded['suppliers_mentioned'].notna()]
        supplier_expanded = supplier_expanded[supplier_expanded['suppliers_mentioned'] != '']
        
        supplier_counts = supplier_expanded['suppliers_mentioned'].value_counts().reset_index()
        supplier_counts.columns = ['Supplier', 'Count']
        fig_suppliers = px.bar(
            supplier_counts,
            x="Supplier",
            y="Count",
            color="Supplier",
            height=400,
            title="Supplier Mentions in AI Categorized Articles"
        )
        fig_suppliers.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_suppliers, use_container_width=True)

    st.markdown("### 🔁 Risk Category Coverage")
    # Show percentage of articles in each category
    total_articles = len(categorization_df)
    coverage_data = []
    for category in RISK_CATEGORIES.keys():
        count = len(categorization_expanded[categorization_expanded['risk_categories'] == category])
        percentage = (count / total_articles) * 100 if total_articles > 0 else 0
        coverage_data.append({'Risk Category': category, 'Articles Count': count, 'Coverage %': percentage})
    
    coverage_df = pd.DataFrame(coverage_data)
    fig_coverage = px.bar(
        coverage_df,
        x='Risk Category',
        y='Coverage %',
        color='Risk Category',
        title="Risk Category Coverage Percentage",
        height=350
    )
    fig_coverage.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_coverage, use_container_width=True)

    st.markdown("### 📊 Risk Categories Co-occurrence")
    # Create a matrix showing how often categories appear together
    article_categories = categorization_df['risk_categories'].apply(lambda x: x if isinstance(x, list) else [])
    
    # Create co-occurrence matrix
    categories = list(RISK_CATEGORIES.keys())
    cooccurrence_matrix = pd.DataFrame(0, index=categories, columns=categories)
    
    for cats in article_categories:
        if isinstance(cats, list) and len(cats) > 1:
            for i, cat1 in enumerate(cats):
                for cat2 in cats[i+1:]:
                    if cat1 in categories and cat2 in categories:
                        cooccurrence_matrix.loc[cat1, cat2] += 1
                        cooccurrence_matrix.loc[cat2, cat1] += 1
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=cooccurrence_matrix.values,
        x=cooccurrence_matrix.columns,
        y=cooccurrence_matrix.index,
        colorscale='Blues',
        hoverongaps=False,
        text=cooccurrence_matrix.values,
        texttemplate="%{text}"
    ))
    fig_heatmap.update_layout(
        title="Risk Categories Co-occurrence Matrix",
        xaxis_title="Risk Category",
        yaxis_title="Risk Category",
        height=450
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
# --- ANALYZE SUPPLIER VIEW ---
elif view_mode == "Analyze Supplier":
    filtered = df_expanded.copy()
    if supplier_filter:
        filtered = filtered[filtered['Supplier'].isin(supplier_filter)]

    st.markdown(f"### 📌 Filtered Articles: {len(filtered)}")
    st.dataframe(filtered[['title', 'published_datetime_utc', 'Supplier', 'Risk_Tags_Predicted', 'Risk_Direction']].rename(
        columns={'Risk_Tags_Predicted': 'Risk_Tags'}
    ))

    st.markdown("### 📈 Risk Distribution by Supplier")
    fig = px.histogram(
        filtered,
        x="Supplier",
        color="Risk_Tags_Predicted",
        barmode="group",
        height=400,
        labels={"Supplier": "Supplier", "count": "Number of Articles", "Risk_Tags_Predicted": "Risk Category"},
        title="Risk Distribution by Supplier"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔁 Risk Direction Breakdown")
    risk_dir_counts = filtered['Risk_Direction'].value_counts().reset_index()
    risk_dir_counts.columns = ['Risk Direction', 'Count']
    fig_dir = px.bar(
        risk_dir_counts,
        x='Risk Direction',
        y='Count',
        color='Risk Direction',
        title="Risk Direction Breakdown",
        height=350
    )
    st.plotly_chart(fig_dir, use_container_width=True)

    st.markdown("### 📅 Yearly Risk Category Trend (Stacked Area)")
    all_years = list(range(df_expanded['Year'].min(), df_expanded['Year'].max() + 1))
    risk_trends = filtered.groupby(['Year', 'Risk_Tags_Predicted']).size().reset_index(name='Count')
    pivot_risks = risk_trends.pivot(index='Year', columns='Risk_Tags_Predicted', values='Count').reindex(all_years, fill_value=0)

    fig_area = go.Figure()
    for col in pivot_risks.columns:
        fig_area.add_trace(go.Scatter(
            x=pivot_risks.index,
            y=pivot_risks[col],
            stackgroup='one',
            name=col
        ))

    fig_area.update_layout(
        title="Stacked Area Chart: Risk Category Frequency Over Years",
        xaxis_title="Year",
        yaxis_title="Number of Articles",
        height=450
    )
    st.plotly_chart(fig_area, use_container_width=True)

# --- OVERVIEW MODE ---
elif view_mode == "Overview":
    st.markdown("### 🌐 All Suppliers Overview")
    heatmap_data = df_expanded.groupby(['Supplier', 'Risk_Tags_Predicted']).size().unstack(fill_value=0)

    st.markdown("#### 🔥 Heatmap: Supplier vs Risk Category")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        hoverongaps=False,
        text=heatmap_data.values,
        texttemplate="%{text}"
    ))
    fig_heatmap.update_layout(
        title="Heatmap of Risk Categories by Supplier",
        xaxis_title="Risk Category",
        yaxis_title="Supplier",
        height=450
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("#### 📊 Overall Risk Category Counts")
    risk_counts = df_expanded['Risk_Tags_Predicted'].value_counts().reset_index()
    risk_counts.columns = ['Risk Category', 'Count']
    fig_risk_counts = px.bar(
        risk_counts,
        x='Risk Category',
        y='Count',
        color='Risk Category',
        title="Overall Risk Category Counts",
        height=350
    )
    st.plotly_chart(fig_risk_counts, use_container_width=True)

    st.markdown("#### 💬 Sentiment Overview")
    sentiment_counts = df_expanded['Risk_Direction'].value_counts().reset_index()
    sentiment_counts.columns = ['Risk Direction', 'Count']
    fig_sentiment = px.bar(
        sentiment_counts,
        x='Risk Direction',
        y='Count',
        color='Risk Direction',
        title="Sentiment Overview",
        height=350
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.markdown("#### 📈 Year-wise Supplier Risk Trend (Smoothed)")

    yearly_risk_trend = df_expanded.groupby(['Year', 'Supplier']).size().reset_index(name='Risk_Article_Count')
    pivot_supplier = yearly_risk_trend.pivot(index='Year', columns='Supplier', values='Risk_Article_Count').fillna(0)

    fig_line = go.Figure()
    for supplier in pivot_supplier.columns:
        fig_line.add_trace(go.Scatter(
            x=pivot_supplier.index,
            y=pivot_supplier[supplier].rolling(window=1).mean(),
            mode='lines+markers',
            name=supplier
        ))
    fig_line.update_layout(
        title="Year-wise Risk Article Count by Supplier",
        xaxis_title="Year",
        yaxis_title="Number of Risk Articles",
        height=450
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("#### 📊 Stacked Bar Chart: Risk Categories by Supplier")

    # Prepare data for stacked bar chart: count of each risk category per supplier
    stacked_data = df_expanded.groupby(['Supplier', 'Risk_Tags_Predicted']).size().reset_index(name='Count')

    fig_stacked = px.bar(
        stacked_data,
        x='Supplier',
        y='Count',
        color='Risk_Tags_Predicted',
        title="Stacked Bar Chart of Risk Categories by Supplier",
        labels={"Count": "Number of Articles", "Supplier": "Supplier", "Risk_Tags_Predicted": "Risk Category"},
        height=450
    )
    fig_stacked.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_stacked, use_container_width=True)

st.markdown("---")
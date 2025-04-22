import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="🎬 IMDb Movie Recommender", layout="wide")
st.title("🍿 IMDb Movie Explorer + Recommender System")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("IMDb_Data_final.csv")  # Change this if your CSV file has a different name
    df.dropna(subset=['Title', 'Director', 'Stars', 'Category'], inplace=True)
    df['combined_features'] = (
        df['Director'].astype(str) + ' ' +
        df['Stars'].astype(str) + ' ' +
        df['Category'].astype(str)
    )
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["📊 Visualizations", "🎯 Movie Recommendation"])

# --- Visualizations ---
if section == "📊 Visualizations":
    st.subheader("🎥 Top Directors with Most High Score Movies")

    top_directors = df['Director'].value_counts().head(10).reset_index()
    top_directors.columns = ['Director', 'MovieCount']

    fig = px.bar(
        top_directors,
        x='Director',
        y='MovieCount',
        color='Director',
        title="Top 10 Directors with Most High-Score Movies",
        text_auto=True,
        width=1000,
        height=600,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig)

    st.subheader("📈 IMDb Ratings Distribution")
    fig2 = px.histogram(df, x="IMDb-Rating", nbins=20, title="IMDb Rating Histogram", color_discrete_sequence=["#E45756"])
    st.plotly_chart(fig2)

# --- Content-Based Recommender ---
elif section == "🎯 Movie Recommendation":
    st.subheader("🎯 Content-Based Movie Recommendation System")

    # --- FILTERS FOR CATEGORY AND RELEASE YEAR ---
    st.sidebar.markdown("### 🎛️ Filter Movies")

    # Category filter
    category_options = df['Category'].dropna().unique().tolist()
    selected_category = st.sidebar.selectbox("🎭 Filter by Category", ["All"] + sorted(category_options))

    # Release Year filter
    if 'ReleaseYear' in df.columns:
        year_options = sorted(df['ReleaseYear'].dropna().unique())
        selected_year = st.sidebar.selectbox("📅 Filter by Release Year", ["All"] + [str(int(year)) for year in year_options])
    else:
        selected_year = "All"

    # Apply filters to dropdown options
    filtered_df = df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['ReleaseYear'] == int(selected_year)]

    # Vectorization and similarity matrix
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df['combined_features'].str.lower().str.replace(' ', ''))
    cosine_sim = cosine_similarity(matrix)

    def recommend_movie(title, top_n=5):
        if title not in df['Title'].values:
            return []
        idx = df[df['Title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        return df.iloc[[i[0] for i in sim_scores]]['Title'].tolist()

    movie = st.selectbox("🎬 Choose a movie to get similar recommendations:", filtered_df['Title'].unique())
    top_n = st.slider("🔢 Number of recommendations", 1, 10, 5)

    if st.button("Recommend 🎬"):
        results = recommend_movie(movie, top_n)
        st.success(f"Movies similar to **{movie}**:")
        for i, rec in enumerate(results, 1):
            st.markdown(f"**{i}. {rec}**")

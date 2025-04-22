# app.py

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
# Optional filters: Genre and Language
st.sidebar.markdown("### 🎯 Optional Filters")

# Genre filter
if 'Category' in df.columns:
    genre_options = df['Category'].dropna().unique().tolist()
    selected_genre = st.sidebar.selectbox("🎭 Filter by Genre", ["All"] + sorted(genre_options))
else:
    selected_genre = "All"

# Language filter
if 'Language' in df.columns:
    language_options = df['Language'].dropna().unique().tolist()
    selected_language = st.sidebar.selectbox("🗣️ Filter by Language", ["All"] + sorted(language_options))
else:
    selected_language = "All"

# Apply filters to the movie list for the dropdown
filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['Category'] == selected_genre]
if selected_language != "All":
    filtered_df = filtered_df[filtered_df['Language'] == selected_language]

# Overwrite movie options (only this line is affected)
selected_movie = st.sidebar.selectbox("🎬 Select a movie:", filtered_df['Title'].dropna().unique())

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

    movie = st.selectbox("Choose a movie to get similar recommendations:", df['Title'].unique())
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend 🎬"):
        results = recommend_movie(movie, top_n)
        st.success(f"Movies similar to **{movie}**:")
        for i, rec in enumerate(results, 1):
            st.markdown(f"**{i}. {rec}**")

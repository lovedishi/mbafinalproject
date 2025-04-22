import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import random

st.set_page_config(page_title="🎬 IMDb Movie Recommender", layout="wide")
st.title("🍿 IMDb Movie Explorer + Recommender System")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("IMDb_Data_final.csv")
    df.dropna(subset=['Title', 'Director', 'Stars', 'Category'], inplace=True)
    df['combined_features'] = (
        df['Director'].astype(str) + ' ' +
        df['Stars'].astype(str) + ' ' +
        df['Category'].astype(str)
    )
    df['Tag'] = df.apply(lambda row: 
                         '🎖️ Top IMDb-rated' if row['IMDb-Rating'] >= 8.5 else
                         '💎 Hidden Gem' if row['IMDb-Rating'] > 7.5 and row['ReleaseYear'] < 2010 else
                         '🎬 Classic' if row['ReleaseYear'] < 1990 else '⭐ Popular', axis=1)
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to", ["📊 Visualizations", "🎯 Movie Recommendation", "📊 Compare Movies", "🎲 Random Spinner"])

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

    st.subheader("🎨 Genre Frequency")
    genre_count = df['Category'].value_counts().reset_index()
    genre_count.columns = ['Genre', 'Count']
    fig3 = px.bar(genre_count, x='Genre', y='Count', title="🎨 Genre Frequency", color='Count')
    st.plotly_chart(fig3)

    st.subheader("📂 Filter by Tags")
    tag_option = st.selectbox("Choose a Tag to Filter Movies", df['Tag'].unique())
    tagged_df = df[df['Tag'] == tag_option]
    st.dataframe(tagged_df[['Title', 'IMDb-Rating', 'Tag']])
    st.download_button("📥 Download Filtered Movies", data=tagged_df.to_csv(index=False), file_name="tagged_movies.csv", mime="text/csv")

# --- Movie Recommendation ---
elif section == "🎯 Movie Recommendation":
    st.subheader("🎯 Content-Based Movie Recommendation System")

    # --- Genre Filter Mode Selection ---
    filter_mode = st.sidebar.radio("Select Genre Filter Mode", ["Single Genre", "Multiple Genres"])

    # --- Genre Filter based on selected mode ---
    if filter_mode == "Single Genre":
        category_filter = st.sidebar.selectbox("Filter by Category (Genre):", options=sorted(df['Category'].unique()), index=0)  # Dropdown for single genre
    else:  # Multiple genres mode
        category_filter = st.sidebar.multiselect("Filter by Categories (Genres):", options=sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))

    year_filter = st.sidebar.slider("Select Release Year Range:", int(df['ReleaseYear'].min()), int(df['ReleaseYear'].max()), (2000, 2023))

    # Filter Data based on genre selection
    if filter_mode == "Single Genre":
        filtered_df = df[(df['Category'] == category_filter) & (df['ReleaseYear'].between(year_filter[0], year_filter[1]))]
    else:
        filtered_df = df[(df['Category'].isin(category_filter)) & (df['ReleaseYear'].between(year_filter[0], year_filter[1]))]

    # --- Clean and Preprocess combined_features ---
    filtered_df['combined_features'] = filtered_df['combined_features'].fillna('')

    # Remove rows where the combined_features is still empty after filling NaN
    filtered_df = filtered_df[filtered_df['combined_features'].str.strip() != '']

    # Clean the text: remove spaces between words (optional) and convert to lowercase
    filtered_df['combined_features'] = filtered_df['combined_features'].str.replace(' ', '', regex=True).str.lower()

    # Check if we have enough data to proceed
    if filtered_df['combined_features'].empty:
        st.error("No valid data found after filtering. Please adjust your filters.")
        st.stop()

    # Vectorization and similarity matrix
    vectorizer = CountVectorizer(stop_words='english')  # Remove stop words
    try:
        matrix = vectorizer.fit_transform(filtered_df['combined_features'])
        cosine_sim = cosine_similarity(matrix)

        def recommend_movie(title, top_n=5):
            if title not in filtered_df['Title'].values:
                return []
            idx = filtered_df[filtered_df['Title'] == title].index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            return filtered_df.iloc[[i[0] for i in sim_scores]]['Title'].tolist()

        movie = st.selectbox("Choose a movie to get similar recommendations:", filtered_df['Title'].unique())
        top_n = st.slider("Number of recommendations", 1, 10, 5)

        if st.button("Recommend 🎬"):
            results = recommend_movie(movie, top_n)
            st.success(f"Movies similar to **{movie}**:")
            for i, rec in enumerate(results, 1):
                st.markdown(f"**{i}. {rec}**")

    except ValueError as e:
        st.error(f"An error occurred: {e}. This could be due to issues with the 'combined_features' column data.")
        st.stop()

# --- Movie Comparison ---
elif section == "📊 Compare Movies":
    st.subheader("📊 Compare Two Movies Side-by-Side")
    col1, col2 = st.columns(2)
    with col1:
        movie1 = st.selectbox("🎥 Select Movie 1", df['Title'], key="movie1")
    with col2:
        movie2 = st.selectbox("🎞️ Select Movie 2", df['Title'], key="movie2")

    compare_df = df[df['Title'].isin([movie1, movie2])]
    st.table(compare_df[['Title', 'IMDb-Rating', 'Director', 'Category', 'ReleaseYear', 'Duration']])

# --- Random Movie Spinner ---
elif section == "🎲 Random Spinner":
    st.subheader("🎲 Feeling Lucky? Spin & Get a Random Movie!")
    if st.button("🎯 Spin the Movie Picker"):
        random_movie = df.sample(1).iloc[0]
        st.success(f"🍿 Watch: **{random_movie['Title']}** ({random_movie['ReleaseYear']}) - Rated: {random_movie['IMDb-Rating']}")

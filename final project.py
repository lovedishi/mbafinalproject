import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import random
import time

st.set_page_config(page_title="🎬 IMDb Movie Recommender", layout="wide")
st.title("🍿 MOVIE MENTOR: A PERSONALIZED MOVIE RECOMMENDER")

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

    # Filter Sidebar - Category (Genre)
    category_filter = st.sidebar.multiselect("Filter by Category (Genre):", options=sorted(df['Category'].unique()), default=sorted(df['Category'].unique()))

    # Filter Sidebar - Year Range (Independent of Genre Filter)
    year_filter = st.sidebar.slider("Select Release Year Range:", int(df['ReleaseYear'].min()), int(df['ReleaseYear'].max()), (2000, 2023))

    # Filtered DataFrame (Independent filters)
    filtered_df = df[(df['Category'].isin(category_filter)) & (df['ReleaseYear'].between(year_filter[0], year_filter[1]))]

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(filtered_df['combined_features'].str.lower().str.replace(' ', ''))
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

    # Search Bar for Movie Information
    st.markdown("---")
    st.subheader("🔍 Search for a Movie")
    search_input = st.text_input("Enter movie name to get details")
    if search_input:
        matched = df[df['Title'].str.lower() == search_input.lower()]
        if not matched.empty:
            st.success("✅ Movie Found:")
            st.write(matched.T)
        else:
            st.warning("❌ Oops! No movie found.")

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

    spin_col, result_col = st.columns([1, 2])

    with spin_col:
        st.markdown("### 🎡 Movie Spinner")

    with result_col:
        placeholder = st.empty()

    if st.button("🎯 Spin the Movie Picker"):
        spinner_anim = ["🔁", "🔃", "🔄"]
        with st.spinner("Spinning the wheel..."):
            for _ in range(6):
                placeholder.markdown(f"### {random.choice(spinner_anim)}")
                time.sleep(0.5)

        random_movie = df.sample(1).iloc[0]
        placeholder.markdown(f"""<div style='text-align:center; font-size:26px; font-weight:bold; color:#2C3E50;'>
        🎬 <u>Watch this movie:</u><br>
        <span style='color:#E74C3C;'>{random_movie['Title']}</span><br>
        ⭐ IMDb: {random_movie['IMDb-Rating']} | 📅 {random_movie['ReleaseYear']} | 🎭 {random_movie['Category']}
        </div>""", unsafe_allow_html=True)

import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import random
import time

# Language Dictionary for English, Hindi, and Telugu
language_dict = {
    "en": {
        "title": "🍿 MOVIE MENTOR: A PERSONALIZED MOVIE RECOMMENDER",
        "sidebar_title": "🔍 Navigation",
        "visualizations": "📊 Visualizations",
        "movie_recommendation": "🎯 Movie Recommendation",
        "compare_movies": "📊 Compare Movies",
        "random_spinner": "🎲 Random Spinner",
        "choose_movie": "Choose a movie to get similar recommendations:",
        "recommend": "Recommend 🎬",
        "mood_label": "What's your mood?",
        "happy": "Happy",
        "sad": "Sad",
        "romantic": "Romantic",
        "watch_movie": "Watch this movie:",
        "search_movie": "🔍 Search for a Movie",
        "search_button": "Search",
        "no_movie_found": "❌ Oops! No movie found.",
        "movie_comparison": "📊 Compare Two Movies Side-by-Side",
        "random_movie": "🎲 Feeling Lucky? Spin & Get a Random Movie!",
        "spin_button": "🎯 Spin the Movie Picker",
        "download_button": "⬇️ Download Filtered Movies",
    },
    "hi": {
        "title": "🍿 मूवी मेंटर: एक व्यक्तिगत मूवी अनुशंसा प्रणाली",
        "sidebar_title": "🔍 नेविगेशन",
        "visualizations": "📊 विज़ुअलाइजेशन",
        "movie_recommendation": "🎯 मूवी अनुशंसा",
        "compare_movies": "📊 मूवी तुलना",
        "random_spinner": "🎲 रैंडम स्पिनर",
        "choose_movie": "समान अनुशंसा प्राप्त करने के लिए एक मूवी चुनें:",
        "recommend": "अनुशंसा करें 🎬",
        "mood_label": "आपका मूड कैसा है?",
        "happy": "खुश",
        "sad": "उदास",
        "romantic": "रोमांटिक",
        "watch_movie": "यह मूवी देखें:",
        "search_movie": "🔍 मूवी खोजें",
        "search_button": "खोजें",
        "no_movie_found": "❌ ओह! कोई मूवी नहीं मिली।",
        "movie_comparison": "📊 दो मूवीज़ की तुलना करें",
        "random_movie": "🎲 किस्मत आज़माएँ? एक रैंडम मूवी स्पिन करें!",
        "spin_button": "🎯 मूवी स्पिनर घुमाएं",
        "download_button": "⬇️ फ़िल्टर की गई मूवी डाउनलोड करें",
    },
    "te": {
        "title": "🍿 మూవీ మెంటార్: ఒక వ్యక్తిగత మూవీ సిఫారసు వ్యవస్థ",
        "sidebar_title": "🔍 నావిగేషన్",
        "visualizations": "📊 విజువలైజేషన్స్",
        "movie_recommendation": "🎯 మూవీ సిఫారసు",
        "compare_movies": "📊 మూవీస్ పోల్చండి",
        "random_spinner": "🎲 రాండమ్ స్పిన్నర్",
        "choose_movie": "సమానమైన సిఫారసు పొందడానికి ఒక సినిమా ఎంచుకోండి:",
        "recommend": "సిఫారసు చేయండి 🎬",
        "mood_label": "మీ మూడ్ ఏమిటి?",
        "happy": "సంతోషంగా",
        "sad": "ఊహించలేము",
        "romantic": "ప్రేమిక",
        "watch_movie": "ఈ సినిమా చూడండి:",
        "search_movie": "🔍 మూవీ అన్వేషించండి",
        "search_button": "అన్వేషించండి",
        "no_movie_found": "❌ ఓహ్! సినిమా లభించలేదు.",
        "movie_comparison": "📊 రెండు మూవీలను పోల్చండి",
        "random_movie": "🎲 లక్కీ ఫీల్! ఒక రాండమ్ మూవీ స్ఫిన్ చేయండి!",
        "spin_button": "🎯 మూవీ స్పిన్నర్ తిప్పండి",
        "download_button": "⬇️ ఫిల్టర్ చేసిన సినిమాను డౌన్‌లోడ్ చేయండి",
    }
}

# Language Selection
selected_language = st.sidebar.selectbox("Select Language", ["English", "हिन्दी", "తెలుగు"])
lang_map = {"English": "en", "हिन्दी": "hi", "తెలుగు": "te"}
lang_code = lang_map[selected_language]

# ✅ Display Title
st.title(language_dict[lang_code]["title"])

# Load Data
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

# Sidebar Navigation
st.sidebar.title(language_dict[lang_code]["sidebar_title"])
section = st.sidebar.radio("Go to", [language_dict[lang_code]["visualizations"], 
                                    language_dict[lang_code]["movie_recommendation"], 
                                    language_dict[lang_code]["compare_movies"],
                                    language_dict[lang_code]["random_spinner"]])

# Visualizations
if section == language_dict[lang_code]["visualizations"]:
    st.subheader("🎥 Top Directors with Most High Score Movies")
    top_directors = df['Director'].value_counts().head(10).reset_index()
    top_directors.columns = ['Director', 'MovieCount']
    fig = px.bar(top_directors, x='Director', y='MovieCount', color='Director',
                 title="Top 10 Directors with Most High-Score Movies", text_auto=True,
                 width=1000, height=600, color_discrete_sequence=px.colors.qualitative.Safe)
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
    
    # ✅ Download button for filtered movies
    st.download_button(
        label=language_dict[lang_code]["download_button"],
        data=tagged_df.to_csv(index=False),
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

# Movie Recommendation
elif section == language_dict[lang_code]["movie_recommendation"]:
    st.subheader(language_dict[lang_code]["movie_recommendation"])
    year_filter = st.sidebar.slider("Select Release Year Range:", int(df['ReleaseYear'].min()), int(df['ReleaseYear'].max()), (2000, 2023))
    filtered_df = df[df['ReleaseYear'].between(year_filter[0], year_filter[1])]
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

    movie = st.selectbox(language_dict[lang_code]["choose_movie"], filtered_df['Title'].unique())
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button(language_dict[lang_code]["recommend"]):
        results = recommend_movie(movie, top_n)
        st.success(f"Movies similar to **{movie}**:")
        for i, rec in enumerate(results, 1):
            st.markdown(f"**{i}. {rec}**")

    st.subheader(language_dict[lang_code]["search_movie"])
    search_query = st.text_input(language_dict[lang_code]["search_movie"])
    if st.button(language_dict[lang_code]["search_button"]):
        result = df[df['Title'].str.lower() == search_query.lower()]
        if not result.empty:
            st.write(result[['Title', 'Director', 'Stars', 'IMDb-Rating', 'Category', 'Duration', 'ReleaseYear']])
        else:
            st.warning(language_dict[lang_code]["no_movie_found"])

# Movie Comparison
elif section == language_dict[lang_code]["compare_movies"]:
    st.subheader(language_dict[lang_code]["movie_comparison"])
    col1, col2 = st.columns(2)
    with col1:
        movie1 = st.selectbox("🎥 Select Movie 1", df['Title'], key="movie1")
    with col2:
        movie2 = st.selectbox("🎞️ Select Movie 2", df['Title'], key="movie2")

    compare_df = df[df['Title'].isin([movie1, movie2])]
    st.table(compare_df[['Title', 'IMDb-Rating', 'Director', 'Category', 'ReleaseYear', 'Duration']])

# Random Spinner
elif section == language_dict[lang_code]["random_spinner"]:
    st.subheader(language_dict[lang_code]["random_movie"])
    spin_col, result_col = st.columns([1, 2])

    with spin_col:
        st.markdown("### 🎡 Movie Spinner")
    with result_col:
        placeholder = st.empty()

    if st.button(language_dict[lang_code]["spin_button"]):
        spinner_anim = ["🔁", "🔃", "🔄"]
        with st.spinner("Spinning the wheel..."):
            for _ in range(6):
                placeholder.markdown(f"### {random.choice(spinner_anim)}")
                time.sleep(0.5)

        random_movie = df.sample(1).iloc[0]
        placeholder.markdown(f"""<div style='text-align:center; font-size:26px; font-weight:bold; color:#2C3E50;'>
        🎬 <u>{language_dict[lang_code]['watch_movie']}</u><br>
        <span style='color:#E74C3C;'>{random_movie['Title']}</span><br>
        ⭐ IMDb: {random_movie['IMDb-Rating']} | 📅 {random_movie['ReleaseYear']} | 🎭 {random_movie['Category']}
        </div>""", unsafe_allow_html=True)

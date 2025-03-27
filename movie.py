import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("C:/Users/91620/OneDrive/Desktop/New 2025/imdb_movie_dataset.csv")

# Fill missing values
df['Genre'] = df['Genre'].fillna('')

# Convert genres into a structured format
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genre_matrix = vectorizer.fit_transform(df['Genre'])

# Compute cosine similarity
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Function to get movie recommendations
def recommend_movies(title, df=df, cosine_sim=cosine_sim):
    if title not in df['Title'].values:
        return pd.DataFrame(columns=['Title', 'Genre'])
    
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return df[['Title', 'Genre']].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get similar recommendations based on genre.")

selected_movie = st.selectbox("Choose a movie", df['Title'].values)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(selected_movie)
    if not recommendations.empty:
        st.write("### Recommended Movies:")
        st.dataframe(recommendations)
    else:
        st.write("Movie not found. Please check the title.")

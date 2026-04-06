import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Finder")

try:
    df = pd.read_csv("movies.csv")
    df.columns = df.columns.str.strip()

    st.write("Data loaded:")
    st.write(df)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["description"])

    query = st.text_input("What kind of movie are you looking for?")

    if st.button("Find Movies"):
        if query:
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

            df["similarity"] = similarities
            top = df.sort_values("similarity", ascending=False).head(3)

            for _, row in top.iterrows():
                st.subheader(row["title"])
                st.write(row["description"])
                st.markdown(f"[View on Letterboxd]({row['letterboxd_url']})")
                st.write(f"Score: {row['similarity']:.2f}")

except Exception as e:
    st.error(f"Error: {e}")
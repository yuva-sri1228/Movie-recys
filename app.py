import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")


@st.cache_resource
def load():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    model = joblib.load("model.joblib")
    return movies, ratings, model


movies, ratings, model = load()
user_ids, movie_ids = model["user_ids"], model["movie_ids"]
user_f, item_f, user_mean = model["user_f"], model["item_f"], model["user_mean"]
title_of = movies.set_index("movieId")["title"]
mid_to_col = {m: i for i, m in enumerate(movie_ids)}

mode = st.radio("Recommend by", ["Existing user", "Movies I like"], horizontal=True)
n = st.slider("Number of recommendations", 3, 15, 5)


def top_n(scores, exclude_ids):
    order = np.argsort(-scores)
    out = []
    for j in order:
        mid = movie_ids[j]
        if mid in exclude_ids:
            continue
        out.append((title_of.get(mid, str(mid)), float(np.clip(scores[j], 0.5, 5))))
        if len(out) == n:
            break
    return out


if mode == "Existing user":
    uid = st.selectbox("User ID", user_ids)
    if st.button("Recommend Movies"):
        i = int(np.where(user_ids == uid)[0][0])
        scores = user_f[i] @ item_f + user_mean[i]
        seen = set(ratings.loc[ratings.userId == uid, "movieId"])
        st.subheader("Recommended Movies")
        for title, s in top_n(scores, seen):
            st.success(f"🎞️ {title} — predicted rating {s:.2f}")
else:
    liked = st.multiselect("Pick movies you enjoyed", movies["title"].tolist())
    if st.button("Recommend Movies"):
        if not liked:
            st.warning("Pick at least one movie.")
        else:
            ids = movies.loc[movies.title.isin(liked), "movieId"]
            cols = [mid_to_col[m] for m in ids if m in mid_to_col]
            if not cols:
                st.warning("Those movies have no ratings in the dataset.")
            else:
                # item-item similarity in the learned latent space
                profile = item_f[:, cols].mean(axis=1, keepdims=True).T
                sims = cosine_similarity(profile, item_f.T).ravel()
                st.subheader("Because you liked these")
                for title, s in top_n(sims, set(ids)):
                    st.success(f"🎞️ {title} — similarity {s:.2f}")

st.caption(f"Model: SVD collaborative filtering, k={model['k']}, test RMSE={model['rmse']:.3f}. For educational use.")

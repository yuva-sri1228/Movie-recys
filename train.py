"""Train the recommender and save it to model.joblib.

Dataset: MovieLens "ml-latest-small" (https://grouplens.org/datasets/movielens/latest/)
Put movies.csv and ratings.csv in the same folder, then run:  python train.py
"""
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")


def build_matrix(df, user_ids, movie_ids):
    u = df["userId"].map({v: i for i, v in enumerate(user_ids)})
    m = df["movieId"].map({v: i for i, v in enumerate(movie_ids)})
    return csr_matrix((df["rating"], (u, m)), shape=(len(user_ids), len(movie_ids)))


def fit_svd(df, user_ids, movie_ids, k):
    mat = build_matrix(df, user_ids, movie_ids)
    # mean-center each user's ratings (only where rated) so SVD learns taste, not rating habit
    counts = np.maximum(mat.getnnz(axis=1), 1)
    user_mean = np.asarray(mat.sum(axis=1)).ravel() / counts
    centered = mat.copy().astype(float)
    centered.data -= np.repeat(user_mean, mat.getnnz(axis=1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    user_f = svd.fit_transform(centered)
    return user_f, svd.components_, user_mean


def rmse(df, user_ids, movie_ids, user_f, item_f, user_mean):
    uidx = {v: i for i, v in enumerate(user_ids)}
    midx = {v: i for i, v in enumerate(movie_ids)}
    df = df[df["userId"].isin(uidx) & df["movieId"].isin(midx)]
    u = df["userId"].map(uidx).values
    m = df["movieId"].map(midx).values
    pred = (user_f[u] * item_f[:, m].T).sum(axis=1) + user_mean[u]
    pred = np.clip(pred, 0.5, 5)
    return float(np.sqrt(np.mean((pred - df["rating"].values) ** 2)))


train, test = train_test_split(ratings, test_size=0.2, random_state=42)
user_ids = np.sort(ratings["userId"].unique())
movie_ids = np.sort(ratings["movieId"].unique())

# Pick number of latent factors using the held-out test set
best = None
for k in (10, 20, 30, 50):
    uf, itf, um = fit_svd(train, user_ids, movie_ids, k)
    score = rmse(test, user_ids, movie_ids, uf, itf, um)
    print(f"k={k:>3}  test RMSE = {score:.4f}")
    if best is None or score < best[0]:
        best = (score, k)

print(f"Best k = {best[1]} (RMSE {best[0]:.4f})")

# Final model: retrain on ALL ratings with best k
user_f, item_f, user_mean = fit_svd(ratings, user_ids, movie_ids, best[1])
joblib.dump(
    {"user_ids": user_ids, "movie_ids": movie_ids, "user_f": user_f,
     "item_f": item_f, "user_mean": user_mean, "rmse": best[0], "k": best[1]},
    "model.joblib", compress=3,
)
print("Saved model.joblib")

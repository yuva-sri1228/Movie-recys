# Movie Recommendation System
1. Download ml-latest-small from https://grouplens.org/datasets/movielens/latest/ and copy `movies.csv`, `ratings.csv` here.
2. `pip install -r requirements.txt`
3. `python train.py`        # trains, prints RMSE, saves model.joblib
4. `streamlit run app.py`   # run locally
Deploy: push all files (incl. model.joblib, movies.csv, ratings.csv) to GitHub -> share.streamlit.io -> New app -> app.py

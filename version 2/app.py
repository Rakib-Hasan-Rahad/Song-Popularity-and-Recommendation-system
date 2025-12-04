# app.py
"""
Hybrid Audio-Semantic Recommender — interactive app (Streamlit + Colab widgets)
- Expects a prepared dataframe CSV named 'df4_prepared.csv' with:
  Title, Artist, Genre, Year, numeric features, artist_tfidf_*, genre_tfidf_*, Cluster
- Provides sliders for artist & genre weights and a Recommend button.
- Streamlit app: run `streamlit run app.py`
- Colab: import this file and run run_colab_ui()
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# --------- Configuration ----------
DF_PATH = "df4_prepared.csv"   #  path 
PCA_COMPONENTS = 25
DEFAULT_N_RECS = 5
# ---------------------------------

def load_df(path=DF_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Place your df4 CSV there.")
    df = pd.read_csv(path)
    # make sure Title column exists
    if 'Title' not in df.columns:
        raise ValueError("df does not contain 'Title' column.")
    return df

def detect_tfidf_columns(df):
    artist_cols = [c for c in df.columns if c.startswith("artist_tfidf_")]
    genre_cols = [c for c in df.columns if c.startswith("genre_tfidf_")]
    return artist_cols, genre_cols

def get_rec_features(df, artist_cols, genre_cols):
    numeric_features = [
        'BPM','Energy','Danceability','Loudness',
        'Liveness','Valence','Acousticness',
        'Speechiness','Length','Popularity'
    ]
    # keep only existing features
    numeric = [c for c in numeric_features if c in df.columns]
    rec_features = numeric + artist_cols + genre_cols
    return rec_features, numeric

def build_feature_matrix(df, rec_features, artist_cols, genre_cols, artist_weight=0.2, genre_weight=0.4):
    """
    Build feature matrix from df given weights.
    - TF-IDF columns in df assumed already normalized between 0..1 or similar.
    """
    df_local = df.copy()
    # apply weights to tfidf columns (we avoid mutating original by copying)
    if artist_cols:
        df_local[artist_cols] = df_local[artist_cols] * float(artist_weight)
    if genre_cols:
        df_local[genre_cols] = df_local[genre_cols] * float(genre_weight)
    # ensure rec_features exist
    missing = [f for f in rec_features if f not in df_local.columns]
    if missing:
        raise ValueError(f"Missing features in df: {missing}")
    feature_matrix = df_local[rec_features].values.astype(float)
    return feature_matrix, df_local

def compute_pca_similarity(feature_matrix, n_components=PCA_COMPONENTS):
    pca = PCA(n_components=min(n_components, feature_matrix.shape[1]), random_state=42)
    features_pca = pca.fit_transform(feature_matrix)
    sim = cosine_similarity(features_pca)
    return sim, pca, features_pca

def recommend_songs_from_similarity(df_local, sim_matrix, title, rec_features, top_n=DEFAULT_N_RECS):
    if title not in df_local['Title'].values:
        raise ValueError(f"Song '{title}' not found in dataset.")
    idx = int(df_local[df_local['Title'] == title].index[0])
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_indices = [i for i, s in scores]
    sim_scores = [s for (_, s) in scores]
    recommended = df_local.loc[rec_indices, ['Title','Artist','Genre','Year','Cluster']].copy()
    recommended['Similarity Score'] = sim_scores
    # explanation: diff in raw rec_features
    base_vec = df_local.loc[idx, rec_features].values.astype(float)
    rec_vecs = df_local.loc[rec_indices, rec_features].values.astype(float)
    explanation_df = pd.DataFrame(rec_vecs - base_vec, columns=[f"{f} diff" for f in rec_features], index=recommended.index)
    return recommended.reset_index(drop=True), explanation_df

# ---------------------------
# Streamlit App
# ---------------------------
def run_streamlit_app():
    try:
        import streamlit as st
    except Exception as e:
        print("Streamlit is not installed. Install with `pip install streamlit` and re-run.")
        raise

    st.set_page_config(page_title="Hybrid Audio-Semantic Recommender", layout="wide")
    st.title("🎧 Hybrid Audio–Semantic Recommender (TF-IDF + PCA)")

    # Load data
    try:
        df = load_df(DF_PATH)
    except Exception as e:
        st.error(str(e))
        return

    artist_cols, genre_cols = detect_tfidf_columns(df)
    rec_features, numeric = get_rec_features(df, artist_cols, genre_cols)

    # Sidebar controls
    st.sidebar.header("Settings")
    artist_w = st.sidebar.slider("Artist weight", min_value=0.0, max_value=2.0, value=0.20, step=0.01)
    genre_w  = st.sidebar.slider("Genre weight",  min_value=0.0, max_value=2.0, value=0.40, step=0.01)
    n_components = st.sidebar.slider("PCA components",  min_value=5, max_value=min(100, len(rec_features)), value=min(PCA_COMPONENTS, len(rec_features)), step=1)
    n_clusters = st.sidebar.slider("KMeans clusters (for evaluation)", min_value=2, max_value=50, value=8, step=1)
    top_n = st.sidebar.slider("Number of recommendations", min_value=1, max_value=20, value=5, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Data info:")
    st.sidebar.write(f"Samples: **{df.shape[0]}**")
    st.sidebar.write(f"Rec features: **{len(rec_features)}**")
    st.sidebar.write(f"Artist TF-IDF cols: **{len(artist_cols)}**, Genre TF-IDF cols: **{len(genre_cols)}**")

    # Song selector with autocomplete-ish behavior
    song = st.selectbox("Select a song title (type to search):", options=df['Title'].tolist())

    if st.button("Generate recommendations"):
        with st.spinner("Computing recommendations..."):
            # Build weighted features and compute PCA + similarity
            feature_matrix, df_local = build_feature_matrix(df, rec_features, artist_cols, genre_cols, artist_w, genre_w)
            sim_matrix, pca, features_pca = compute_pca_similarity(feature_matrix, n_components=n_components)

            # Optionally recompute clusters for evaluation
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            try:
                df_local['Cluster'] = kmeans.fit_predict(features_pca)
            except Exception:
                # if kmeans fails (rare), fall back to previous cluster column if exists
                if 'Cluster' not in df_local.columns:
                    df_local['Cluster'] = -1

            # Make recommendations
            try:
                recommended, explanation = recommend_songs_from_similarity(df_local, sim_matrix, song, rec_features, top_n=top_n)
            except Exception as e:
                st.error(str(e))
                return

            # Show results
            st.subheader("🎵 Top Recommendations")
            st.table(recommended)

            st.subheader("Feature differences")
            # show only top numeric + a few tfidf columns to avoid giant tables
            show_cols = [c for c in explanation.columns if any(n in c for n in numeric)]  # numeric diffs
            # add first 10 tfidf diffs (artist + genre) for context
            tfidf_cols = [c for c in explanation.columns if c not in show_cols]
            show_cols = show_cols + tfidf_cols[:10]
            st.dataframe(explanation[show_cols].round(3))

            # Evaluation metrics
            base_row = df_local[df_local['Title'] == song].iloc[0]

            # Genre match rate
            genre_match_rate = (recommended['Genre'] == base_row['Genre']).mean() if not recommended.empty else np.nan

            # Cluster match rate
            cluster_match_rate = None
            if 'Cluster' in recommended.columns:
                cluster_match_rate = (recommended['Cluster'] == base_row.get('Cluster', -999)).mean() if not recommended.empty else np.nan

            # Average feature distance
            if not recommended.empty:
                
                # Extract feature values
                reco_matrix = df_local.loc[
                    df_local['Title'].isin(recommended['Title']),
                    rec_features
                ].astype(float).values  # ensure float type
                reco_matrix = np.atleast_2d(reco_matrix)

                base_vector = base_row[rec_features].astype(float).values  # ensure float type
                base_vector = np.atleast_2d(base_vector)

                # Handle the single-row, single-feature edge case
                if reco_matrix.shape[0] == 1 and reco_matrix.shape[1] == 1:
                    distances = np.array([abs(reco_matrix[0,0] - base_vector[0,0])])
                else:
                    distances = np.linalg.norm(reco_matrix - base_vector, axis=1)

                avg_dist = float(np.mean(distances))
            else:
                avg_dist = np.nan
           

            # Display in Streamlit
            st.subheader("Evaluation")
            st.write({
                "Genre Match Rate": round(float(genre_match_rate), 3) if not np.isnan(genre_match_rate) else "N/A",
                "Cluster Match Rate": round(float(cluster_match_rate), 3) if cluster_match_rate is not None and not np.isnan(cluster_match_rate) else "N/A",
                "Average Feature Distance": round(avg_dist, 4) if not np.isnan(avg_dist) else "N/A"
            })


            # Allow download of recommendations CSV
            csv = recommended.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations (CSV)", data=csv, file_name="recommendations.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("Tips:\n- Increase/decrease artist & genre weights to prioritize those signals. \n- Use PCA components to tradeoff speed/precision. \n- Run multiple songs to evaluate qualitative results.")

# ---------------------------
# Colab / Jupyter UI
# ---------------------------
def run_colab_ui():
    """
    Interactive UI using ipywidgets — paste this into a Colab cell:
    from app import run_colab_ui
    run_colab_ui()
    """
    try:
        from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, Button, HBox, VBox, Output
    except Exception:
        print("ipywidgets not installed. Install with `pip install ipywidgets` and enable in Jupyter/Colab.")
        return

    df = load_df(DF_PATH)
    artist_cols, genre_cols = detect_tfidf_columns(df)
    rec_features, numeric = get_rec_features(df, artist_cols, genre_cols)

    artist_slider = FloatSlider(value=0.2, min=0.0, max=2.0, step=0.01, description='Artist W:')
    genre_slider = FloatSlider(value=0.4, min=0.0, max=2.0, step=0.01, description='Genre W:')
    ncomp_slider = IntSlider(value=min(PCA_COMPONENTS, len(rec_features)), min=5, max=min(100, len(rec_features)), step=1, description='PCA:')
    nclusters_slider = IntSlider(value=8, min=2, max=50, step=1, description='Clusters:')
    topn_slider = IntSlider(value=5, min=1, max=20, step=1, description='Top N:')
    song_dropdown = Dropdown(options=df['Title'].tolist(), description='Song:')

    out = Output()

    def on_button_clicked(b):
        with out:
            out.clear_output()
            print("Computing...")
            feature_matrix, df_local = build_feature_matrix(df, rec_features, artist_cols, genre_cols, artist_slider.value, genre_slider.value)
            sim_matrix, pca, features_pca = compute_pca_similarity(feature_matrix, n_components=ncomp_slider.value)
            kmeans = KMeans(n_clusters=nclusters_slider.value, random_state=42)
            try:
                df_local['Cluster'] = kmeans.fit_predict(features_pca)
            except:
                df_local['Cluster'] = df_local.get('Cluster', -1)
            try:
                recommended, explanation = recommend_songs_from_similarity(df_local, sim_matrix, song_dropdown.value, rec_features, top_n=topn_slider.value)
            except Exception as e:
                print("Error:", e)
                return
            display(recommended)
            print("\nExplanation (numeric diffs + top tfidf diffs):")
            show_cols = [c for c in explanation.columns if any(n in c for n in numeric)]
            tfidf_cols = [c for c in explanation.columns if c not in show_cols]
            display(explanation[show_cols + tfidf_cols[:10]].round(3))
    btn = Button(description="Recommend")
    btn.on_click(on_button_clicked)

    ui = VBox([HBox([artist_slider, genre_slider, ncomp_slider, nclusters_slider, topn_slider]), song_dropdown, btn, out])
    display(ui)

# ---------------------------
# CLI fallback
# ---------------------------
def run_cli():
    print("Running in CLI mode.")
    df = load_df(DF_PATH)
    artist_cols, genre_cols = detect_tfidf_columns(df)
    rec_features, numeric = get_rec_features(df, artist_cols, genre_cols)
    print(f"Loaded df with {df.shape[0]} rows. Found {len(artist_cols)} artist tfidf cols and {len(genre_cols)} genre tfidf cols.")
    artist_w = float(input("Artist weight [0.2]: ") or 0.2)
    genre_w  = float(input("Genre weight [0.4]: ") or 0.4)
    song = input("Song title (exact): ")
    feature_matrix, df_local = build_feature_matrix(df, rec_features, artist_cols, genre_cols, artist_w, genre_w)
    sim_matrix, pca, features_pca = compute_pca_similarity(feature_matrix)
    recommended, explanation = recommend_songs_from_similarity(df_local, sim_matrix, song, rec_features, top_n=5)
    print(recommended)
    # save CSV
    recommended.to_csv("recommendations_cli.csv", index=False)
    print("Saved recommendations_cli.csv")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # If Streamlit runs this file, Streamlit will call run_streamlit_app() automatically.
    # But to support `python app.py` we detect if running under streamlit context.
    if "streamlit" in sys.modules:
        # when streamlit imports this file, the script continues and __name__ == "__main__"
        run_streamlit_app()
    else:
        # If user runs `python app.py`, open CLI prompt and tell them how to launch streamlit
        print("To run the Streamlit app: streamlit run app.py")
        print("Or run CLI mode now.")
        run_cli()

import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Mobile Phone Recommender (CBF)", layout="wide")
st.title("📱 Mobile Phone Recommender (CBF)")

# ---- load artifacts ----
phones_df = joblib.load("cleaned_phone_data.joblib")
cosine_sim = joblib.load("cosine_sim.joblib")

# ---- build a UNIQUE display name and mapping ----
phones_df["display_name"] = phones_df["Brand"].astype(str).str.strip() + " - " + phones_df["Model"].astype(str).str.strip()

# Drop duplicate display names, keep the first occurrence so we map to ONE row per label
phones_df = phones_df.drop_duplicates(subset=["display_name"]).reset_index(drop=True)

# Map display_name -> row index (guaranteed unique now)
indices = phones_df.reset_index().set_index("display_name")["index"]  # Series: label -> int

def get_recs(display_name: str, n: int = 10) -> pd.DataFrame:
    """Return top-n similar phones for a given display_name ('Brand - Model')."""
    idx = indices.get(display_name, None)
    if idx is None:
        return pd.DataFrame()

    # ensure scalar int (paranoia guard)
    if isinstance(idx, (pd.Series, list, tuple)):
        idx = int(pd.Series(idx).iloc[0])
    else:
        idx = int(idx)

    # one row of cosine similarities (1-D)
    row = cosine_sim[idx]
    if hasattr(row, "ravel"):
        row = row.ravel()

    # sort, skip itself, take top-n
    sim_scores = list(enumerate(row))
    sim_scores.sort(key=lambda t: t[1], reverse=True)
    top = [(i, s) for i, s in sim_scores if i != idx][:n]
    ids = [i for i, _ in top]

    cols = ["Brand","Model","Price","RAM","Storage","Screen Size","Battery Capacity","main_camera_mp"]
    return phones_df.iloc[ids][cols].reset_index(drop=True)

# ---- UI ----
choice = st.selectbox("Choose a model", sorted(phones_df["display_name"].unique()))
if st.button("Find similar"):
    recs = get_recs(choice, n=10)
    if recs.empty:
        st.warning("No recommendations found (duplicate label or index mismatch). Try another model.")
    else:
        st.dataframe(recs)

# (optional) quick sanity
st.caption(f"Rows: {phones_df.shape[0]}  |  Cosine shape: {getattr(cosine_sim, 'shape', None)}")

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Mobile Phone Recommender (CBF)", layout="wide")
st.title("📱 Mobile Phone Recommender (CBF)")

# ---- load artifacts ----
phones_df = joblib.load("cleaned_phone_data.joblib")
cosine_sim = joblib.load("cosine_sim.joblib")

# ---- build a UNIQUE display name but KEEP the original index ----
phones_df["display_name"] = (
    phones_df["Brand"].astype(str).str.strip() + " - " + phones_df["Model"].astype(str).str.strip()
)

# Mapping display_name → original row index
indices = phones_df["display_name"].reset_index().set_index("display_name")["index"]

def get_recs(display_name: str, n: int = 10) -> pd.DataFrame:
    """Return top-n similar phones for a given display_name ('Brand - Model')."""
    if display_name not in indices:
        return pd.DataFrame()

    idx = int(indices[display_name])

    # fetch similarity row
    row = cosine_sim[idx]
    if hasattr(row, "ravel"):
        row = row.ravel()

    # sort, skip itself, take top-n
    sim_scores = list(enumerate(row))
    sim_scores.sort(key=lambda t: t[1], reverse=True)
    top = [(i, s) for i, s in sim_scores if i != idx][:n]
    ids = [i for i, _ in top]

    cols = ["Brand", "Model", "Price", "RAM", "Storage", "Screen Size", "Battery Capacity", "main_camera_mp"]
    return phones_df.iloc[ids][cols].reset_index(drop=True)

# ---- UI ----
choice = st.selectbox("Choose a model", sorted(phones_df["display_name"].unique()))
if st.button("Find similar"):
    recs = get_recs(choice, n=10)
    if recs.empty:
        st.warning("No recommendations found. Try another model.")
    else:
        st.dataframe(recs)

# Debug sanity check
st.caption(f"phones_df: {phones_df.shape} | cosine_sim: {cosine_sim.shape}")

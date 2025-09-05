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
indices = phones_df.reset_index().set_index("display_name")["index"]

def get_recs(display_name: str, n: int = 10) -> pd.DataFrame:
    """Return top-n similar phones for a given display_name ('Brand - Model')."""
    # Resolve display_name -> one row index (handle duplicates gracefully)
    if display_name not in indices.index:
        return pd.DataFrame()

    idx_match = indices.loc[display_name]  # can be scalar int or a Series of ints
    if isinstance(idx_match, pd.Series):
        idx = int(idx_match.iloc[0])       # pick the first occurrence
    else:
        idx = int(idx_match)

    # safety: ensure within bounds
    if idx < 0 or idx >= cosine_sim.shape[0]:
        return pd.DataFrame()

    row = cosine_sim[idx]
    row = row.ravel() if hasattr(row, "ravel") else row

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
        st.warning("No recommendations found. Try another model.")
    else:
        st.dataframe(recs)

# Debug sanity check
st.caption(f"phones_df: {phones_df.shape} | cosine_sim: {cosine_sim.shape}")


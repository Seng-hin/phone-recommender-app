import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Mobile Phone Recommender (CBF)", layout="wide")
st.title("📱 Mobile Phone Recommender (CBF)")

# ---- load artifacts (already in your repo) ----
phones_df = joblib.load("cleaned_phone_data.joblib")
cosine_sim = joblib.load("cosine_sim.joblib")

# ---- build a unique display key and the index map ----
phones_df["display_name"] = phones_df["Brand"].astype(str) + " - " + phones_df["Model"].astype(str)
indices = pd.Series(phones_df.index, index=phones_df["display_name"]).drop_duplicates()

def get_recs(display_name: str, n: int = 10) -> pd.DataFrame:
    """Return top-n similar phones for a given display_name ('Brand - Model')."""
    idx = indices.get(display_name, None)
    if idx is None:
        return pd.DataFrame()

    idx = int(idx)                          # ensure scalar int
    row = cosine_sim[idx].ravel()           # 1-D similarity row
    sim_scores = list(enumerate(row))
    sim_scores.sort(key=lambda t: t[1], reverse=True)

    # skip itself and take top-n
    top = [(i, s) for i, s in sim_scores if i != idx][:n]
    ids = [i for i, _ in top]

    cols = ["Brand","Model","Price","RAM","Storage","Screen Size","Battery Capacity","main_camera_mp"]
    return phones_df.iloc[ids][cols].reset_index(drop=True)

# ---- UI ----
choice = st.selectbox("Choose a model", sorted(phones_df["display_name"].unique()))
if st.button("Find similar"):
    try:
        recs = get_recs(choice, n=10)
        if recs.empty:
            st.warning("No recommendations found (index mismatch or single-item brand). Try another model.")
        else:
            st.dataframe(recs)
    except Exception as e:
        st.exception(e)

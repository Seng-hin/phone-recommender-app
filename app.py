# app.py (minimal)
import streamlit as st, joblib, pandas as pd

st.set_page_config(page_title="Phone Recommender", layout="wide")
st.title("📱 Mobile Phone Recommender (CBF)")

phones_df = joblib.load("models/cleaned_phone_data.joblib")
cosine_sim = joblib.load("models/cosine_sim.joblib")
indices = pd.Series(phones_df.index, index=phones_df["Model"]).drop_duplicates()

def get_recs(model_name, n=10):
    if model_name not in indices: return pd.DataFrame()
    idx = indices[model_name]
    sim = list(enumerate(cosine_sim[idx]))
    sim = sorted(sim, key=lambda x: x[1], reverse=True)[1:n+1]
    ids = [i for i,_ in sim]
    cols = ["Brand","Model","Price","RAM","Storage","Screen Size","Battery Capacity","main_camera_mp"]
    return phones_df.iloc[ids][cols].reset_index(drop=True)

choice = st.selectbox("Choose a model", sorted(phones_df["Model"].unique()))
if st.button("Find similar"):
    recs = get_recs(choice, n=10)
    if recs.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(recs)

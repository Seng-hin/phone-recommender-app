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
# ---- UI (search + dropdown + filters) ----

import difflib

def ranked_options(query: str, options: list[str], topk: int = 20) -> list[str]:
    """Return options ranked by how well they match query (case-insensitive)."""
    if not query:
        return options[:topk]
    q = query.lower().strip()

    # 1) strong boost for startswith / substring
    scored = []
    for opt in options:
        o = opt.lower()
        score = 0
        if o.startswith(q): score += 3
        if q in o:          score += 2
        # 2) fuzzy similarity (difflib)
        score += difflib.SequenceMatcher(a=q, b=o).ratio()
        scored.append((score, opt))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [opt for _, opt in scored[:topk]]

# all labels (keep original index alignment!)
all_labels = list(phones_df["display_name"])

st.subheader("Choose a model")
col_search, col_n = st.columns([3, 1], vertical_alignment="bottom")

with col_search:
    query = st.text_input(
        "Type to search (brand or model). Case-insensitive. e.g. `sam` → Samsung",
        value="",
        placeholder="Search… (e.g., samsung, note, 13 pro, g50)"
    )
with col_n:
    topn = st.number_input("Top-N", min_value=5, max_value=50, value=10, step=1)

# live, case-insensitive suggestions (ranked)
suggestions = ranked_options(query, all_labels, topk=50)
selected_label = st.selectbox(
    "Matches",
    options=suggestions if suggestions else ["— no matches —"],
    index=0 if suggestions else None,
    disabled=not suggestions
)

# optional: let users browse the full list via expander (shows a ▸/▾ arrow)
with st.expander("▸ Browse all models"):
    browse_choice = st.selectbox("All models", options=sorted(all_labels))
    # prefer the typed selection; if user opens expander and chooses one, override
    if browse_choice and (not query or st.button("Use selection above", key="use_browse")):
        selected_label = browse_choice

# run
run = st.button("Find similar")

if run:
    if not suggestions and selected_label == "— no matches —":
        st.warning("No matches for your search. Try a different keyword.")
    else:
        recs = get_recs(selected_label, n=int(topn))

        if recs.empty:
            st.warning("No recommendations found. Try another model.")
        else:
            # ---- filters (sidebar) ----
            st.sidebar.header("Filters")
            # brand
            brands = sorted(recs["Brand"].unique().tolist())
            f_brands = st.sidebar.multiselect("Brand", brands, default=brands)

            # price
            pmin, pmax = float(recs["Price"].min()), float(recs["Price"].max())
            f_price = st.sidebar.slider("Price ($)", pmin, pmax, (pmin, pmax))

            # RAM / Storage (treat as numeric)
            rmin, rmax = int(recs["RAM"].min()), int(recs["RAM"].max())
            smin, smax = int(recs["Storage"].min()), int(recs["Storage"].max())
            f_ram = st.sidebar.slider("RAM (GB)", rmin, rmax, (rmin, rmax))
            f_storage = st.sidebar.slider("Storage (GB)", smin, smax, (smin, smax))

            # Screen / Battery / Camera
            scmin, scmax = float(recs["Screen Size"].min()), float(recs["Screen Size"].max())
            bcmin, bcmax = int(recs["Battery Capacity"].min()), int(recs["Battery Capacity"].max())
            cammin, cammax = float(recs["main_camera_mp"].min()), float(recs["main_camera_mp"].max())
            f_screen = st.sidebar.slider("Screen Size (in)", scmin, scmax, (scmin, scmax))
            f_batt   = st.sidebar.slider("Battery (mAh)", bcmin, bcmax, (bcmin, bcmax))
            f_cam    = st.sidebar.slider("Main Camera (MP, total)", cammin, cammax, (cammin, cammax))

            # apply filters
            fr = recs[
                (recs["Brand"].isin(f_brands)) &
                (recs["Price"].between(*f_price)) &
                (recs["RAM"].between(*f_ram)) &
                (recs["Storage"].between(*f_storage)) &
                (recs["Screen Size"].between(*f_screen)) &
                (recs["Battery Capacity"].between(*f_batt)) &
                (recs["main_camera_mp"].between(*f_cam))
            ].reset_index(drop=True)

            st.success(f"Recommendations for: **{selected_label}**")
            if fr.empty:
                st.info("No results after filtering. Loosen the filters in the sidebar.")
            else:
                st.dataframe(
                    fr,
                    use_container_width=True
                )






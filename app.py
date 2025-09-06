import streamlit as st
import joblib
import pandas as pd
from difflib import SequenceMatcher

st.set_page_config(page_title="Mobile Phone Recommender", layout="wide")
st.title("📱 Mobile Phone Recommender")

# --- Load artifacts ---
phones_df = joblib.load("cleaned_phone_data.joblib")
cosine_sim = joblib.load("cosine_sim.joblib")

# --- Build display label (keep original index order!) ---
phones_df["display_name"] = (
    phones_df["Brand"].astype(str).str.strip() + " - " + phones_df["Model"].astype(str).str.strip()
)

# Unique mapping: display_name -> first occurrence row index
label_to_index = (
    phones_df.reset_index()                              # has 'index' = original row id
             .drop_duplicates(subset=["display_name"])   # pick first row for each label
             .set_index("display_name")["index"]         # Series: label -> int index
)

# Use the **index** (labels, strings) for the suggestions list
all_labels = list(label_to_index.index)

# ---------------- utils ----------------
def ranked_options(query: str, options, topk: int = 30):
    """Rank options by startswith, substring, and fuzzy similarity (case-insensitive)."""
    # force everything to string to prevent .lower() crashes
    options = [str(opt) for opt in options]
    if not query:
        # de-dup while preserving order
        return list(dict.fromkeys(options[:topk]))
    q = str(query).lower().strip()
    scored = []
    for opt in options:
        o = opt.lower()
        score = 0
        if o.startswith(q): score += 3
        if q in o:          score += 2
        score += SequenceMatcher(a=q, b=o).ratio()
        scored.append((score, opt))
    scored.sort(key=lambda t: t[0], reverse=True)
    # de-dup while preserving rank
    return list(dict.fromkeys([opt for _, opt in scored[:topk]]))

def safe_slider(label, lo, hi, default=None, step=None, fmt=None, key=None):
    """Always returns a (lo, hi) tuple; shows fixed text if lo==hi (Streamlit slider guard)."""
    if default is None:
        default = (lo, hi)
    if lo == hi:
        shown = f"{lo}" if fmt is None else fmt.format(lo)
        st.sidebar.write(f"{label}: {shown} (fixed)")
        return (lo, hi)
    return st.sidebar.slider(label, lo, hi, default, step=step, key=key)

# ---------------- recommender ----------------
def get_recs(display_name: str, n: int = 10) -> pd.DataFrame:
    """Top-n unique Brand–Model results for a given display_name."""
    if display_name not in label_to_index.index:
        return pd.DataFrame()
    idx = int(label_to_index[display_name])

    row = cosine_sim[idx]
    row = row.ravel() if hasattr(row, "ravel") else row

    sim_scores = list(enumerate(row))
    sim_scores.sort(key=lambda t: t[1], reverse=True)

    picked_ids, seen_labels = [], set()
    for i, _ in sim_scores:
        if i == idx:
            continue
        lbl = phones_df["display_name"].iat[i]
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        picked_ids.append(i)
        if len(picked_ids) == n:
            break

    cols = ["Brand","Model","Price","RAM","Storage","Screen Size","Battery Capacity","main_camera_mp"]
    return phones_df.iloc[picked_ids][cols].reset_index(drop=True)

# ---------------- UI ----------------
st.subheader("Choose a model")
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")

with c1:
    query = st.text_input(
        "Type to search",
        value=st.session_state.get("q", ""),
        key="q",
        placeholder="Search brand/model…",
    )
with c2:
    topn = st.number_input("Top-N", min_value=5, max_value=50, value=10, step=1, key="k_topn")

suggestions = ranked_options(query, all_labels, topk=50)
selected_label_ui = st.selectbox(
    "Matches",
    options=suggestions if suggestions else ["— no matches —"],
    index=0 if suggestions else None,
    disabled=not suggestions,
    key="suggest_select"
)

if st.button("Find similar", type="primary"):
    if suggestions and selected_label_ui != "— no matches —":
        st.session_state.selected_label = selected_label_ui
        st.session_state.recs = get_recs(selected_label_ui, n=int(topn))
    else:
        st.warning("No matches for your search. Try another keyword.")

# ---------------- results + AUTO filters ----------------
if st.session_state.get("recs") is not None and not st.session_state.recs.empty:
    recs = st.session_state.recs

    st.sidebar.header("Filters")
    brands = sorted(recs["Brand"].unique().tolist())
    f_brands = st.sidebar.multiselect("Brand", brands, default=brands)

    pmin, pmax = float(recs["Price"].min()), float(recs["Price"].max())
    f_price = safe_slider("Price ($)", pmin, pmax)

    rmin, rmax = int(recs["RAM"].min()), int(recs["RAM"].max())
    smin, smax = int(recs["Storage"].min()), int(recs["Storage"].max())
    f_ram = safe_slider("RAM (GB)", rmin, rmax)
    f_storage = safe_slider("Storage (GB)", smin, smax)

    scmin, scmax = float(recs["Screen Size"].min()), float(recs["Screen Size"].max())
    bcmin, bcmax = int(recs["Battery Capacity"].min()), int(recs["Battery Capacity"].max())
    cammin, cammax = float(recs["main_camera_mp"].min()), float(recs["main_camera_mp"].max())
    f_screen = safe_slider("Screen Size (in)", scmin, scmax)
    f_batt   = safe_slider("Battery (mAh)", bcmin, bcmax)
    f_cam    = safe_slider("Main Camera (MP total)", cammin, cammax)

    fr = recs[
        (recs["Brand"].isin(f_brands)) &
        (recs["Price"].between(*f_price)) &
        (recs["RAM"].between(*f_ram)) &
        (recs["Storage"].between(*f_storage)) &
        (recs["Screen Size"].between(*f_screen)) &
        (recs["Battery Capacity"].between(*f_batt)) &
        (recs["main_camera_mp"].between(*f_cam))
    ].reset_index(drop=True)

    st.success(f"Recommendations for **{st.session_state.selected_label}**")
    if fr.empty:
        st.info("No results after filtering. Loosen the filters in the sidebar.")
    else:
        fr.index = range(1, len(fr) + 1)   # start table at 1
        st.dataframe(fr, use_container_width=True)

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

# ===================== SMART PICKER + PERSISTENT STATE + FILTERS =====================
# ===================== SEARCH + AUTO FILTERS (no Apply) =====================
import difflib

def ranked_options(query: str, options: list[str], topk: int = 30) -> list[str]:
    """Rank options by startswith, substring and fuzzy similarity (case-insensitive)."""
    if not query:
        return options[:topk]
    q = query.lower().strip()
    scored = []
    for opt in options:
        o = opt.lower()
        score = 0
        if o.startswith(q): score += 3
        if q in o:          score += 2
        score += difflib.SequenceMatcher(a=q, b=o).ratio()
        scored.append((score, opt))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [opt for _, opt in scored[:topk]]

def safe_slider(label, lo, hi, default=None, step=None, fmt=None, key=None):
    """Return (lo, hi); if lo==hi show a fixed text and return the single value range."""
    if default is None:
        default = (lo, hi)
    if lo == hi:
        shown = f"{lo}" if fmt is None else fmt.format(lo)
        st.sidebar.write(f"{label}: {shown} (fixed)")
        return (lo, hi)
    return st.sidebar.slider(label, lo, hi, default, step=step, key=key)

# ---------- state ----------
if "selected_label" not in st.session_state:
    st.session_state.selected_label = None
if "recs" not in st.session_state:
    st.session_state.recs = None

all_labels = list(phones_df["display_name"])

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

# live suggestions
suggestions = ranked_options(query, all_labels, topk=50)
selected_label_ui = st.selectbox(
    "Matches", options=suggestions if suggestions else ["— no matches —"],
    index=0 if suggestions else None, disabled=not suggestions, key="suggest_select"
)

# run
if st.button("Find similar", type="primary", key="run_btn"):
    if suggestions and selected_label_ui != "— no matches —":
        st.session_state.selected_label = selected_label_ui
        st.session_state.recs = get_recs(selected_label_ui, n=int(topn))
    else:
        st.warning("No matches for your search. Try another keyword.")

# ---------- results + AUTO filters ----------
if st.session_state.recs is not None and not st.session_state.recs.empty:
    recs = st.session_state.recs

    # sidebar filters (auto-apply each change)
    st.sidebar.header("Filters")

    brands = sorted(recs["Brand"].unique().tolist())
    f_brands = st.sidebar.multiselect("Brand", brands, default=brands, key="f_brands")

    pmin, pmax = float(recs["Price"].min()), float(recs["Price"].max())
    f_price = safe_slider("Price ($)", pmin, pmax, key="f_price")

    rmin, rmax = int(recs["RAM"].min()), int(recs["RAM"].max())
    smin, smax = int(recs["Storage"].min()), int(recs["Storage"].max())
    f_ram = safe_slider("RAM (GB)", rmin, rmax, key="f_ram")
    f_storage = safe_slider("Storage (GB)", smin, smax, key="f_storage")

    scmin, scmax = float(recs["Screen Size"].min()), float(recs["Screen Size"].max())
    bcmin, bcmax = int(recs["Battery Capacity"].min()), int(recs["Battery Capacity"].max())
    cammin, cammax = float(recs["main_camera_mp"].min()), float(recs["main_camera_mp"].max())
    f_screen = safe_slider("Screen Size (in)", scmin, scmax, key="f_screen")
    f_batt   = safe_slider("Battery (mAh)", bcmin, bcmax, key="f_batt")
    f_cam    = safe_slider("Main Camera (MP total)", cammin, cammax, key="f_cam")

    # apply immediately (reactive)
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
        st.dataframe(fr, use_container_width=True)










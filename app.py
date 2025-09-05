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
import difflib

# ---------- helpers ----------
def ranked_options(query: str, options: list[str], topk: int = 30) -> list[str]:
    """Rank options by startswith, substring and fuzzy match (case-insensitive)."""
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
    """Return (lo, hi) even when lo==hi; avoids Streamlit slider crash."""
    if default is None:
        default = (lo, hi)
    if lo == hi:
        st.sidebar.write(f"{label}: {lo if fmt is None else fmt.format(lo)} (fixed)")
        return (lo, hi)
    return st.sidebar.slider(label, lo, hi, default, step=step, key=key)

# ---------- state ----------
if "selected_label" not in st.session_state:
    st.session_state.selected_label = None
if "recs" not in st.session_state:
    st.session_state.recs = None
if "filtered" not in st.session_state:
    st.session_state.filtered = None

all_labels = list(phones_df["display_name"])

st.subheader("Choose a model")

c1, c2 = st.columns([3, 1], vertical_alignment="bottom")

with c1:
    query = st.text_input(
        "Type to search (e.g. `sam` → Samsung). Case-insensitive.",
        value=st.session_state.get("q", ""),
        key="q",
        placeholder="Search brand/model…"
    )
with c2:
    topn = st.number_input("Top-N", min_value=5, max_value=50, value=10, step=1, key="k_topn")

# live suggestions (ranked)
suggestions = ranked_options(query, all_labels, topk=50)
suggest_choice = st.selectbox(
    "Matches",
    options=suggestions if suggestions else ["— no matches —"],
    index=0 if suggestions else None,
    disabled=not suggestions,
    key="suggest_select"
)

# browse all — now **wired** to update the selection
with st.expander("▸ Browse all models"):
    browse_choice = st.selectbox("All models", options=sorted(all_labels), key="browse_select")
    if st.button("Use this model", key="use_browse_btn"):
        st.session_state.selected_label = browse_choice
        st.session_state.q = ""  # clear search to avoid confusion
        suggest_choice = browse_choice

# use suggest choice if set
if suggestions and suggest_choice != "— no matches —":
    chosen_now = suggest_choice
else:
    chosen_now = st.session_state.selected_label  # fallback to previous

# main action
if st.button("Find similar", type="primary", key="run_btn"):
    if chosen_now and chosen_now in all_labels:
        st.session_state.selected_label = chosen_now
        st.session_state.recs = get_recs(chosen_now, n=int(st.session_state.k_topn))
        st.session_state.filtered = None
    else:
        st.warning("Please pick a model from matches or the browser.")

# ---------- results + filters (persist across reruns) ----------
if st.session_state.recs is not None and not st.session_state.recs.empty:
    recs = st.session_state.recs

    # FILTERS in a form so they only apply on submit (prevents instant reruns nuking the view)
    with st.sidebar.form("filters_form", clear_on_submit=False):
        st.header("Filters")

        # Brand
        brands = sorted(recs["Brand"].unique().tolist())
        f_brands = st.multiselect("Brand", brands, default=brands, key="f_brands")

        # Price
        pmin, pmax = float(recs["Price"].min()), float(recs["Price"].max())
        f_price = safe_slider("Price ($)", pmin, pmax, key="f_price")

        # RAM / Storage
        rmin, rmax = int(recs["RAM"].min()), int(recs["RAM"].max())
        smin, smax = int(recs["Storage"].min()), int(recs["Storage"].max())
        f_ram = safe_slider("RAM (GB)", rmin, rmax, key="f_ram")
        f_storage = safe_slider("Storage (GB)", smin, smax, key="f_storage")

        # Screen / Battery / Camera
        scmin, scmax = float(recs["Screen Size"].min()), float(recs["Screen Size"].max())
        bcmin, bcmax = int(recs["Battery Capacity"].min()), int(recs["Battery Capacity"].max())
        cammin, cammax = float(recs["main_camera_mp"].min()), float(recs["main_camera_mp"].max())
        f_screen = safe_slider("Screen Size (in)", scmin, scmax, key="f_screen")
        f_batt   = safe_slider("Battery (mAh)", bcmin, bcmax, key="f_batt")
        f_cam    = safe_slider("Main Camera (MP total)", cammin, cammax, key="f_cam")

        apply_filters = st.form_submit_button("Apply filters")

    # compute filtered view only when submitted (or reuse last)
    if apply_filters or st.session_state.filtered is None:
        fr = recs[
            (recs["Brand"].isin(st.session_state.f_brands if "f_brands" in st.session_state else brands)) &
            (recs["Price"].between(*(st.session_state.f_price if "f_price" in st.session_state else (pmin, pmax)))) &
            (recs["RAM"].between(*(st.session_state.f_ram if "f_ram" in st.session_state else (rmin, rmax)))) &
            (recs["Storage"].between(*(st.session_state.f_storage if "f_storage" in st.session_state else (smin, smax)))) &
            (recs["Screen Size"].between(*(st.session_state.f_screen if "f_screen" in st.session_state else (scmin, scmax)))) &
            (recs["Battery Capacity"].between(*(st.session_state.f_batt if "f_batt" in st.session_state else (bcmin, bcmax)))) &
            (recs["main_camera_mp"].between(*(st.session_state.f_cam if "f_cam" in st.session_state else (cammin, cammax))))
        ].reset_index(drop=True)
        st.session_state.filtered = fr
    else:
        fr = st.session_state.filtered

    st.success(f"Recommendations for **{st.session_state.selected_label}**")
    if fr.empty:
        st.info("No results after filtering. Loosen the filters in the sidebar.")
    else:
        st.dataframe(fr, use_container_width=True)








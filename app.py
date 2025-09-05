import streamlit as st
import joblib
import pandas as pd

# --- LOAD MODELS AND DATA ---
# Using st.cache_data to load the files only once for efficiency
@st.cache_data
def load_data():
    """Loads the recommender system's data files."""
    try:
        qualified_phones = joblib.load('qualified_phones.joblib')
        cosine_sim = joblib.load('cosine_sim.joblib')
        indices = pd.Series(qualified_phones.index, index=qualified_phones['model'])
        return qualified_phones, cosine_sim, indices
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'qualified_phones.joblib' and 'cosine_sim.joblib' are present.")
        return None, None, None

qualified_phones, cosine_sim, indices = load_data()

# --- RECOMMENDATION FUNCTION ---
def get_content_recommendations(model_name, n=10):
    """Generates content-based recommendations for a selected phone model."""
    if model_name not in indices:
        return f"Model '{model_name}' not found."

    idx = indices[model_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    phone_indices = [i[0] for i in sim_scores]
    
    columns_to_show = ['brand', 'model', 'price', 'performance', 'battery size', 'screen size', 'main camera', 'RAM']
    return qualified_phones[columns_to_show].iloc[phone_indices]

# --- STREAMLIT UI ---
st.set_page_config(page_title="Cellphone Recommender", layout="wide")

st.title('📱 Enhanced Cellphone Recommender')
st.markdown("Choose a phone you like, and we will recommend 10 others with similar features!")

if qualified_phones is not None and cosine_sim is not None:
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        phone_list = sorted(qualified_phones['model'].tolist())
        selected_phone = st.selectbox('Select a Phone Model:', phone_list, key='selected_phone')

    with col2:
        st.write("") 
        st.write("")
        find_button = st.button('Find Similar Phones', key='find_button')

    # --- Session State Management ---
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'phone_for_recs' not in st.session_state:
        st.session_state.phone_for_recs = ""

    if find_button:
        if selected_phone:
            with st.spinner(f'Finding recommendations similar to {selected_phone}...'):
                st.session_state.recommendations = get_content_recommendations(selected_phone, n=10)
                # Store the name of the phone for which we got recommendations
                st.session_state.phone_for_recs = selected_phone
        else:
            st.warning("Please select a phone from the list.")

    # --- Display Recommendations and Filters ---
    if st.session_state.recommendations is not None:
        recs_df = st.session_state.recommendations

        # --- Sidebar with Filters ---
        st.sidebar.header('Filter Recommendations')
        
        all_brands = sorted(recs_df['brand'].unique())
        selected_brands = st.sidebar.multiselect('Brand', all_brands, default=all_brands)
        
        min_price, max_price = int(recs_df['price'].min()), int(recs_df['price'].max())
        price_range = st.sidebar.slider('Price Range ($)', min_price, max_price, (min_price, max_price))

        min_perf, max_perf = float(recs_df['performance'].min()), float(recs_df['performance'].max())
        perf_range = st.sidebar.slider('Performance Score', min_perf, max_perf, (min_perf, max_perf), step=0.1)

        # Apply filters
        filtered_recs = recs_df[
            (recs_df['brand'].isin(selected_brands)) &
            (recs_df['price'].between(price_range[0], price_range[1])) &
            (recs_df['performance'].between(perf_range[0], perf_range[1]))
        ].reset_index(drop=True)

        # --- Display Filtered Results ---
        st.success(f"Here are your top recommendations for {st.session_state.phone_for_recs}:")

        if not filtered_recs.empty:
            # ================================================================
            # === THIS IS THE ONLY NEW LINE TO CHANGE THE INDEX TO START AT 1 ===
            filtered_recs.index = range(1, len(filtered_recs) + 1)
            # ================================================================

            st.dataframe(
                filtered_recs,
                column_config={
                    "price": st.column_config.NumberColumn("Price ($)", format="$ %d"),
                    "performance": st.column_config.NumberColumn("Performance", format="%.2f"),
                    "battery size": st.column_config.NumberColumn("Battery (mAh)", format="%d mAh"),
                    "screen size": st.column_config.NumberColumn("Screen (in)", format="%.1f in"),
                    "main camera": st.column_config.NumberColumn("Camera (MP)", format="%d MP"),
                    "RAM": st.column_config.NumberColumn("RAM (GB)", format="%d GB"),
                }
            )
        else:
            st.warning("No phones match your filter criteria. Try adjusting the filters in the sidebar.")

else:
    st.error("Application could not start. Please check the logs or ensure data files are available.")

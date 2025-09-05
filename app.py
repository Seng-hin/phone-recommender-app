import streamlit as st
import joblib
import pandas as pd

# --- LOAD MODELS AND DATA ---
@st.cache_data
def load_data():
    """Loads saved TF-IDF model data."""
    try:
        phones_df = joblib.load('models/cbf_train_df.joblib')
        similarity_matrix = joblib.load('models/cbf_matrix.joblib')
        indices = pd.Series(phones_df.index, index=phones_df['Model'])
        return phones_df, similarity_matrix, indices
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'cbf_train_df.joblib' and 'cbf_matrix.joblib' exist.")
        return None, None, None

phones_df, cosine_sim, indices = load_data()

# --- RECOMMENDATION FUNCTION ---
def get_content_recommendations(model_name, n=10):
    """Find similar phones using TF-IDF cosine similarity."""
    if model_name not in indices:
        return f"Model '{model_name}' not found."

    idx = indices[model_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    phone_indices = [i[0] for i in sim_scores]

    # Adjust these columns to match your actual dataframe
    columns_to_show = ['Brand', 'Model', 'Price ($)', 'RAM', 'Battery Capacity (mAh)', 
                       'Screen Size', 'main_camera_mp', 'Storage']
    
    return phones_df[columns_to_show].iloc[phone_indices]

# --- STREAMLIT UI ---
st.set_page_config(page_title="📱 TF-IDF Phone Recommender", layout="wide")
st.title('📱 Enhanced Mobile Recommender System')
st.markdown("Select a phone you like, and we'll suggest similar phones!")

if phones_df is not None and cosine_sim is not None:

    col1, col2 = st.columns([2, 1])

    with col1:
        phone_list = sorted(phones_df['Model'].unique())
        selected_phone = st.selectbox('Select a Phone Model:', phone_list)

    with col2:
        st.write("")
        st.write("")
        find_button = st.button('🔍 Find Similar Phones')

    # --- Session State ---
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'phone_for_recs' not in st.session_state:
        st.session_state.phone_for_recs = ""

    if find_button:
        if selected_phone:
            with st.spinner(f'Finding phones similar to {selected_phone}...'):
                st.session_state.recommendations = get_content_recommendations(selected_phone, n=10)
                st.session_state.phone_for_recs = selected_phone
        else:
            st.warning("Please select a phone model.")

    # --- Display Recommendations ---
    if st.session_state.recommendations is not None:
        recs_df = st.session_state.recommendations

        st.sidebar.header('🔧 Filter Recommendations')
        brands = sorted(recs_df['Brand'].unique())
        selected_brands = st.sidebar.multiselect('Brand', brands, default=brands)

        min_price, max_price = int(recs_df['Price ($)'].min()), int(recs_df['Price ($)'].max())
        price_range = st.sidebar.slider('Price Range ($)', min_price, max_price, (min_price, max_price))

        filtered_recs = recs_df[
            (recs_df['Brand'].isin(selected_brands)) &
            (recs_df['Price ($)'].between(price_range[0], price_range[1]))
        ].reset_index(drop=True)

        st.success(f"Top recommendations based on {st.session_state.phone_for_recs}:")

        if not filtered_recs.empty:
            filtered_recs.index = range(1, len(filtered_recs) + 1)

            st.dataframe(
                filtered_recs,
                column_config={
                    "Price ($)": st.column_config.NumberColumn("Price ($)", format="$ %d"),
                    "Battery Capacity (mAh)": st.column_config.NumberColumn("Battery", format="%d mAh"),
                    "Screen Size": st.column_config.NumberColumn("Screen Size", format="%.2f in"),
                    "main_camera_mp": st.column_config.NumberColumn("Camera", format="%d MP"),
                    "RAM": st.column_config.NumberColumn("RAM", format="%d GB"),
                    "Storage": st.column_config.NumberColumn("Storage", format="%d GB"),
                }
            )
        else:
            st.warning("No phones match the selected filters.")

else:
    st.error("🚫 Could not start application. Check if the model files are present.")

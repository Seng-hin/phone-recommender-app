import streamlit as st
import joblib
import pandas as pd

# ==============================================================================
#                      LOAD THE TRAINED MODEL AND DATA
# ==============================================================================
@st.cache_data
def load_data():
    """Loads saved model data and creates a display name for the dropdown."""
    try:
        phones_df = joblib.load('cleaned_phone_data.joblib')
        cosine_sim = joblib.load('tfidf_matrix.joblib')
        
        # --- NEW: Create a combined column for a better user experience ---
        phones_df['display_name'] = phones_df['Brand'] + " - " + phones_df['Model']
        
        # Create a mapping from the display name back to the original model name
        display_to_model_map = pd.Series(phones_df.Model.values, index=phones_df.display_name).to_dict()
        
        indices = pd.Series(phones_df.index, index=phones_df['Model'])
        return phones_df, cosine_sim, indices, display_to_model_map
    except FileNotFoundError:
        st.error("Model files not found! 😭 Ensure 'cleaned_phone_data.joblib' and 'tfidf_matrix.joblib' are in your GitHub repo.")
        return None, None, None, None

# Load the data when the app starts
phones_df, cosine_sim, indices, display_to_model_map = load_data()

# ==============================================================================
#                      RECOMMENDATION FUNCTION
# ==============================================================================
def get_content_recommendations(model_name, n=10):
    """Finds similar phones using the pre-computed cosine similarity matrix."""
    if model_name not in indices:
        return pd.DataFrame() 

    idx = indices[model_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n + 1]
    phone_indices = [i[0] for i in sim_scores]
    columns_to_show = ['Brand', 'Model', 'Price', 'RAM', 'Storage', 'Screen Size', 'Battery Capacity', 'main_camera_mp']
    return phones_df[columns_to_show].iloc[phone_indices]

# ==============================================================================
#                             STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Phone Recommender", layout="wide")
st.title('📱 Mobile Phone Recommender System')
st.markdown("Select a phone you like, and we'll suggest 10 others with similar features using our **TF-IDF model**!")

if phones_df is not None:
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # --- CHANGED: Use the 'display_name' for the selectbox ---
        phone_list = sorted(phones_df['display_name'].unique().tolist())
        selected_display_name = st.selectbox('Select a Phone Model from the list:', phone_list)

    with col2:
        st.write("") 
        st.write("")
        find_button = st.button('🔍 Find Similar Phones')

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'phone_for_recs' not in st.session_state:
        st.session_state.phone_for_recs = ""

    if find_button:
        # --- CHANGED: Map the display name back to the actual model name ---
        selected_model_name = display_to_model_map[selected_display_name]
        
        with st.spinner(f'Finding recommendations similar to {selected_model_name}...'):
            st.session_state.recommendations = get_content_recommendations(selected_model_name, n=10)
            st.session_state.phone_for_recs = selected_display_name

    if st.session_state.recommendations is not None:
        recs_df = st.session_state.recommendations

        # --- THIS IS THE FIX: Only show filters if there are recommendations ---
        if not recs_df.empty:
            st.sidebar.header('🔧 Filter Recommendations')
            all_brands = sorted(recs_df['Brand'].unique())
            selected_brands = st.sidebar.multiselect('Brand', all_brands, default=all_brands)
            
            min_price, max_price = int(recs_df['Price'].min()), int(recs_df['Price'].max())
            price_range = st.sidebar.slider('Price Range ($)', min_price, max_price, (min_price, max_price))

            filtered_recs = recs_df[
                (recs_df['Brand'].isin(selected_brands)) &
                (recs_df['Price'].between(price_range[0], price_range[1]))
            ].reset_index(drop=True)

            st.success(f"Here are your top recommendations based on the **{st.session_state.phone_for_recs}**:")

            if not filtered_recs.empty:
                filtered_recs.index = range(1, len(filtered_recs) + 1)
                st.dataframe(filtered_recs, column_config={
                        "Price": st.column_config.NumberColumn("Price ($)", format="$ %d"),
                        "RAM": st.column_config.NumberColumn("RAM (GB)", format="%d GB"),
                        "Storage": st.column_config.NumberColumn("Storage (GB)", format="%d GB"),
                        "Screen Size": st.column_config.NumberColumn("Screen (in)", format="%.2f in"),
                        "Battery Capacity": st.column_config.NumberColumn("Battery (mAh)", format="%d mAh"),
                        "main_camera_mp": st.column_config.NumberColumn("Camera (MP)", format="%d MP"),
                    })
            else:
                st.warning("No phones match your filter criteria. Try adjusting the filters in the sidebar!")
        else:
            st.warning("Could not find any recommendations for the selected phone.")```

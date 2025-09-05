import streamlit as st
import joblib
import pandas as pd

# ==============================================================================
#                      LOAD THE TRAINED MODEL AND DATA
# ==============================================================================
# Use st.cache_data to load the files only once, making the app faster.
@st.cache_data
def load_data():
    """Loads the saved TF-IDF model data from the 'models' directory."""
    try:
        # Load the dataframe, the similarity matrix, and create the indices series
        phones_df = joblib.load('models/cbf_train_df.joblib')
        cosine_sim = joblib.load('models/cbf_matrix.joblib')
        indices = pd.Series(phones_df.index, index=phones_df['Model'])
        return phones_df, cosine_sim, indices
    except FileNotFoundError:
        st.error(
            "Model files not found! 😭 Please make sure the 'models' directory "
            "with 'cbf_train_df.joblib' and 'cbf_matrix.joblib' is in your project folder."
        )
        return None, None, None

# Load the data when the app starts
phones_df, cosine_sim, indices = load_data()

# ==============================================================================
#                      RECOMMENDATION FUNCTION
# ==============================================================================
def get_content_recommendations(model_name, n=10):
    """Finds similar phones using the pre-computed cosine similarity matrix."""
    if model_name not in indices:
        return pd.DataFrame() # Return an empty dataframe if the model is not found

    # Get the index of the phone that matches the model name
    idx = indices[model_name]

    # Get the pairwise similarity scores of all phones with the selected phone
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar phones (excluding the phone itself)
    sim_scores = sim_scores[1:n + 1]

    # Get the phone indices from the similarity scores
    phone_indices = [i[0] for i in sim_scores]

    # Define the columns you want to show in the final recommendation table
    columns_to_show = ['Brand', 'Model', 'Price', 'RAM', 'Storage', 
                       'Screen Size', 'Battery Capacity', 'main_camera_mp']
    
    # Return the top N most similar phones as a DataFrame
    return phones_df[columns_to_show].iloc[phone_indices]

# ==============================================================================
#                             STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Phone Recommender", layout="wide")

st.title('📱 Mobile Phone Recommender System')
st.markdown("Select a phone you like, and we'll suggest 10 others with similar features using our **TF-IDF model**!")

# Only show the main UI if the data was loaded successfully
if phones_df is not None and cosine_sim is not None:
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a dropdown menu with a sorted list of all unique phone models
        phone_list = sorted(phones_df['Model'].unique().tolist())
        selected_phone = st.selectbox('Select a Phone Model from the list:', phone_list)

    with col2:
        # Add vertical space to better align the button with the dropdown
        st.write("") 
        st.write("")
        find_button = st.button('🔍 Find Similar Phones')

    # Use session state to store recommendations so they don't disappear when using filters
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'phone_for_recs' not in st.session_state:
        st.session_state.phone_for_recs = ""

    # Generate recommendations when the button is clicked
    if find_button:
        if selected_phone:
            with st.spinner(f'Finding recommendations similar to {selected_phone}...'):
                st.session_state.recommendations = get_content_recommendations(selected_phone, n=10)
                st.session_state.phone_for_recs = selected_phone # Remember which phone was selected
        else:
            st.warning("Please select a phone model from the list.")

    # --- Display Recommendations and Filtering Sidebar ---
    if st.session_state.recommendations is not None:
        recs_df = st.session_state.recommendations

        # Create a sidebar for filtering options
        st.sidebar.header('🔧 Filter Recommendations')
        
        # Filter by Brand
        all_brands = sorted(recs_df['Brand'].unique())
        selected_brands = st.sidebar.multiselect('Brand', all_brands, default=all_brands)
        
        # Filter by Price
        min_price, max_price = int(recs_df['Price'].min()), int(recs_df['Price'].max())
        price_range = st.sidebar.slider('Price Range ($)', min_price, max_price, (min_price, max_price))

        # Apply the selected filters to the recommendations DataFrame
        filtered_recs = recs_df[
            (recs_df['Brand'].isin(selected_brands)) &
            (recs_df['Price'].between(price_range[0], price_range[1]))
        ].reset_index(drop=True)

        # Display the filtered recommendations in the main area
        st.success(f"Here are your top recommendations based on the **{st.session_state.phone_for_recs}**:")

        if not filtered_recs.empty:
            # Set the table index to start from 1 instead of 0
            filtered_recs.index = range(1, len(filtered_recs) + 1)

            st.dataframe(
                filtered_recs,
                # Configure the columns for better readability
                column_config={
                    "Price": st.column_config.NumberColumn("Price ($)", format="$ %d"),
                    "RAM": st.column_config.NumberColumn("RAM (GB)", format="%d GB"),
                    "Storage": st.column_config.NumberColumn("Storage (GB)", format="%d GB"),
                    "Screen Size": st.column_config.NumberColumn("Screen (in)", format="%.2f in"),
                    "Battery Capacity": st.column_config.NumberColumn("Battery (mAh)", format="%d mAh"),
                    "main_camera_mp": st.column_config.NumberColumn("Camera (MP)", format="%d MP"),
                }
            )
        else:
            st.warning("No phones match your filter criteria. Try adjusting the filters in the sidebar!")

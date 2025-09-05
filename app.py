import streamlit as st
import joblib
import pandas as pd

# --- LOAD MODELS AND DATA ---
# Using st.cache_data to load the files only once for efficiency
@st.cache_data
def load_data():
    """Loads the recommender system's data files."""
    try:
        # Load the dataframe containing phone features and the similarity matrix
        qualified_phones = joblib.load('qualified_phones.joblib')
        cosine_sim = joblib.load('cosine_sim.joblib')
        
        # Recreate the indices Series required for mapping model names to index numbers
        indices = pd.Series(qualified_phones.index, index=qualified_phones['model'])
        return qualified_phones, cosine_sim, indices
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'qualified_phones.joblib' and 'cosine_sim.joblib' are present.")
        # Return None for all to prevent the app from running without data
        return None, None, None

# Load the data right at the start
qualified_phones, cosine_sim, indices = load_data()

# --- RECOMMENDATION FUNCTION ---
def get_content_recommendations(model_name, n=10):
    """
    Generates content-based recommendations for a selected phone model.
    """
    if model_name not in indices:
        return f"Model '{model_name}' not found."

    # Get the index of the phone that matches the model name
    idx = indices[model_name]

    # Get the pairwise similarity scores of all phones with the selected phone
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar phones (excluding the phone itself at index 0)
    sim_scores = sim_scores[1:n + 1]

    # Get the indices of the recommended phones
    phone_indices = [i[0] for i in sim_scores]

    # --- UPDATED ---
    # Return the full details for the top N most similar phones
    # We select more columns to display and use for filtering
    columns_to_show = ['brand', 'model', 'price', 'performance', 'battery size', 'screen size', 'main camera', 'RAM']
    return qualified_phones[columns_to_show].iloc[phone_indices]

# --- STREAMLIT UI ---

st.set_page_config(page_title="Cellphone Recommender", layout="wide")

st.title('📱 Enhanced Cellphone Recommender')
st.markdown("Choose a phone you like, and we will recommend 10 others with similar features!")

# Check if data was loaded successfully before building the rest of the UI
if qualified_phones is not None and cosine_sim is not None:
    
    # Main page layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a dropdown menu for the user to select a phone
        phone_list = sorted(qualified_phones['model'].tolist())
        selected_phone = st.selectbox(
            'Select a Phone Model:',
            phone_list,
            key='selected_phone'
        )

    with col2:
        # Add some vertical space to align the button better
        st.write("") 
        st.write("")
        # Create a button that triggers the recommendation process
        find_button = st.button('Find Similar Phones', key='find_button')

    # --- Session State to hold recommendations ---
    # This makes the recommendations persist so we can filter them
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    if find_button:
        if selected_phone:
            with st.spinner(f'Finding recommendations similar to {selected_phone}...'):
                # Store the generated recommendations in the session state
                st.session_state.recommendations = get_content_recommendations(selected_phone, n=10)
        else:
            st.warning("Please select a phone from the list.")

    # --- Display Recommendations and Filters ---
    if st.session_state.recommendations is not None:
        recs_df = st.session_state.recommendations

        # --- NEW: FILTERING LOGIC IN THE SIDEBAR ---
        st.sidebar.header('Filter Recommendations')
        
        # Filter by Brand (multiselect)
        all_brands = sorted(recs_df['brand'].unique())
        selected_brands = st.sidebar.multiselect('Brand', all_brands, default=all_brands)
        
        # Filter by Price (slider)
        min_price, max_price = int(recs_df['price'].min()), int(recs_df['price'].max())
        price_range = st.sidebar.slider('Price Range ($)', min_price, max_price, (min_price, max_price))

        # Filter by Performance (slider)
        min_perf, max_perf = float(recs_df['performance'].min()), float(recs_df['performance'].max())
        perf_range = st.sidebar.slider('Performance Score', min_perf, max_perf, (min_perf, max_perf), step=0.1)

        # Apply filters to the recommendation dataframe
        filtered_recs = recs_df[
            (recs_df['brand'].isin(selected_brands)) &
            (recs_df['price'].between(price_range[0], price_range[1])) &
            (recs_df['performance'].between(perf_range[0], perf_range[1]))
        ].reset_index(drop=True)

        # --- Display Filtered Results ---
        st.success(f"Here are your top recommendations for {st.session_state.selected_phone}:")

        if not filtered_recs.empty:
            # Using st.dataframe for better formatting and interactivity
            st.dataframe(
                filtered_recs,
                # Configure column formats for better readability
                column_config={
                    "price": st.column_config.NumberColumn(
                        "Price ($)",
                        format="$ %d",
                    ),
                    "performance": st.column_config.NumberColumn(
                        "Performance",
                        format="%.2f",
                    ),
                     "battery size": st.column_config.NumberColumn(
                        "Battery (mAh)",
                        format="%d mAh",
                    ),
                    "screen size": st.column_config.NumberColumn(
                        "Screen (in)",
                        format="%.1f in",
                    ),
                    "main camera": st.column_config.NumberColumn(
                        "Camera (MP)",
                        format="%d MP",
                    ),
                    "RAM": st.column_config.NumberColumn(
                        "RAM (GB)",
                        format="%d GB",
                    ),
                }
            )
        else:
            st.warning("No phones match your filter criteria. Try adjusting the filters in the sidebar.")

else:
    # This message shows if the initial data loading failed
    st.error("Application could not start. Please check the logs or ensure data files are available.")

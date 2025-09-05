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
        # Recreate the indices Series inside the app
        indices = pd.Series(qualified_phones.index, index=qualified_phones['model'])
        return qualified_phones, cosine_sim, indices
    except FileNotFoundError:
        st.error("Model files not found. Please make sure 'qualified_phones.joblib' and 'cosine_sim.joblib' are in your GitHub repository.")
        return None, None, None

qualified_phones, cosine_sim, indices = load_data()

# --- RECOMMENDATION FUNCTION ---
# This is the same function from your notebook, adapted for Streamlit
def get_content_recommendations(model_name, n=10):
    """
    Generates content-based recommendations for a selected phone model.
    """
    if model_name not in indices:
        return f"Model '{model_name}' not found."

    # Get the index of the phone that matches the model name
    idx = indices[model_name]

    # Get the pairwise similarity scores of all phones with that phone
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar phones (excluding the phone itself at index 0)
    sim_scores = sim_scores[1:n+1]

    # Get the phone indices
    phone_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar phones as a DataFrame
    return qualified_phones[['brand', 'model', 'price', 'performance']].iloc[phone_indices]

# --- STREAMLIT UI ---

st.set_page_config(page_title="Cellphone Recommender", layout="wide")
st.title('📱 Cellphone Recommender System')
st.markdown("Choose a phone you like, and we will recommend 10 others with similar features!")

# Check if data loaded correctly before creating the UI
if qualified_phones is not None and cosine_sim is not None:
    # Create a dropdown menu for the user to select a phone
    phone_list = sorted(qualified_phones['model'].tolist())
    selected_phone = st.selectbox(
        'Select a Phone Model:',
        phone_list
    )

    # Create a button that triggers the recommendation process
    if st.button('Find Similar Phones'):
        if selected_phone:
            # Show a loading spinner while processing
            with st.spinner(f'Finding recommendations for {selected_phone}...'):
                recommendations = get_content_recommendations(selected_phone, n=10)
                
                st.success("Here are your top 10 recommendations:")
                # Display the results in a clean table
                st.table(recommendations.reset_index(drop=True))
        else:
            st.warning("Please select a phone from the list.")

else:
    st.error("Application could not start. Please check the logs.")
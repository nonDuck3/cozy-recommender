from vector_encoder import compute_cosine_similarity
import streamlit as st

file_name = "./data/embeddings/all_embeddings.pkl"
siglip_model_path = "google/siglip-base-patch16-224"

st.set_page_config(
    page_title="COZY · Personal Winter Apparel Recommender",
    page_icon="👚"
)

st.markdown("""
    <style>
    .stForm {
        background-color: #FBE9F9;  
    }

    .stNumberInput button:nth-of-type(1) {
        background-color: #FFFBE5; 
    }

    .stNumberInput button:nth-of-type(2) {
        background-color: #FFFBE5;
    }
            
    .stFormSubmitButton > button {
        background-color: #9566E6;
        color: white;
    }
            
    .stFormSubmitButton > button:hover {
        background-color: #7F4FD6;
    }
    </style>
""", unsafe_allow_html=True)

with st.form("recommendation_form", clear_on_submit=True):
    st.markdown(
        "<h3 style='font-weight: 300; letter-spacing: 2px; color: #5C4033;'>"
        "C O Z Y<span style='color: #D3D3D3; font-weight: 100;'> | </span> "
        "<span style='font-size: 18px; font-style: italic; font-weight: 300; color: #A98467;'>"
        "Comfy Outfits Zoned for You ✨</span></h3>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<h5 style='font-weight: 180;'>Get personalized outfit recommendations based on your style and budget today!</h5>", 
        unsafe_allow_html=True
    )

    user_choice = st.text_input("What are you looking for?", max_chars=50, placeholder="e.g. black cashmere sweater with yellow collar")
    number = st.number_input("Enter budget", min_value=20, max_value=100000, value="min", step=10)
        
    submitted = st.form_submit_button("Search 🔍")
    if submitted:
        if user_choice.strip():
            display_df = compute_cosine_similarity(file_path=siglip_model_path, user_pref=user_choice, file=file_name, max_price=number)
            styled_df = display_df.style.set_properties(**{'background-color': 'white'})
            st.write(f"Showing results for: _{user_choice}_")
            st.dataframe(styled_df,     
                column_config={
                    "Price": st.column_config.NumberColumn(
                        "Price (in AUD)",
                        format="%.2f"
                    )
                }, 
                hide_index=True
            )
        else:
            st.warning("Text fields cannot be empty!")


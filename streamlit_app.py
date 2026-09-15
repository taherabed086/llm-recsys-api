import streamlit as st
import requests

st.title("LLM-Based Recommendation System — Live Demo")

API_URL = "http://127.0.0.1:8000"

dataset = st.selectbox("Choose dataset", ["beauty", "video_games", "yelp"])
user_id = st.text_input("Enter User ID")

if st.button("Get Recommendations"):
    if not user_id:
        st.warning("Please enter a user ID")
    else:
        with st.spinner("Calling the LLM recommender..."):
            response = requests.get(f"{API_URL}/recommend/{dataset}/{user_id}")

        if response.status_code == 200:
            data = response.json()
            st.success("Recommendations:")
            for i, item in enumerate(data["recommendations"], 1):
                st.write(f"**{i}.** {item['item_text']}")
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text or f"HTTP {response.status_code}"
            st.error(f"Error: {error_detail}")
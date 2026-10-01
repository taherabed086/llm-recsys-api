# LLM-Based Recommendation System

A dataset-agnostic recommendation pipeline evaluated across 3 real-world
datasets (Amazon Beauty, Amazon Video Games, Yelp), comparing Item-based
Collaborative Filtering against an LLM-based recommender (Claude API).

## Features
- Canonical schema + adapter pattern for dataset-agnostic processing
- Leave-one-out evaluation (Recall@k, nDCG@k)
- FastAPI backend + Streamlit demo interface

## Running locally
1. `pip install -r requirements.txt`
2. Set `ANTHROPIC_API_KEY` environment variable
3. `uvicorn main:app --reload`
4. `streamlit run streamlit_app.py` (in a separate terminal)
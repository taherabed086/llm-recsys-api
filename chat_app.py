"""
chat_app.py

This is the chat interface for the recommendation system. Instead of
filling out a form (pick a dataset, type a user ID), the user just
describes what they want in plain language, and the app finds the
closest matching products using semantic search.

How it works, step by step:
1. The user's message (plus the rest of the conversation) gets turned
   into a short English search phrase by Claude. This matters because
   the product catalog is in English, so we need a clean query to
   search against.
2. That phrase gets compared against every product description in the
   catalog using sentence embeddings. This is semantic search, so it
   catches things like "makes my skin softer" matching "moisturizer"
   even though the words are completely different.
3. The closest matching products are returned directly, each with its
   own similarity score. Early on, this app used to look up a real
   user who had bought the closest item and return that user's whole
   history. That caused weird results (asking for horror games
   returned sports titles, because one matched user also happened to
   own those). Returning the matched items themselves fixed it.

Confidence is shown explicitly as High, Medium, or Low, based on the
actual similarity score, instead of pretending every answer is equally
certain.
"""

import os
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import anthropic

load_dotenv()

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-5"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_MATCHES = 5
MAX_ITEMS_PER_DATASET = 15000

# These thresholds decide which confidence bucket a similarity score
# falls into. They were picked by trying a few values and seeing what
# felt right, not from a formal evaluation, so they're worth revisiting
# once there's real usage data to look at.
CONFIDENCE_THRESHOLDS = {"high": 0.55, "medium": 0.35, "low": 0.0}

DATASET_FILES = {
    "beauty": "data/canonical_beauty.csv",
    "video_games": "data/canonical_vg.csv",
    "yelp": "data/canonical_yelp.csv",
}

DATASET_LABELS = {
    "beauty": "Beauty and Personal Care",
    "video_games": "Video Games",
    "yelp": "Restaurants and Local Businesses",
}

CONFIDENCE_LABELS = {
    "high": "Strong match",
    "medium": "Moderate match",
    "low": "Weak match",
}

SUGGESTED_PROMPTS = [
    "Something for dry skin",
    "A relaxing building game",
    "A cozy Italian restaurant",
]

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# -----------------------------------------------------------------------
# Page setup and styling
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Recommendation Assistant",
    page_icon="🛍️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #6C5CE7 0%, #341f97 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .confidence-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .confidence-high   { background: #d4f7dc; color: #1e7e34; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low    { background: #ffe1d6; color: #a94442; }
    .confidence-none   { background: #f1f1f1; color: #555; }

    .item-score {
        color: #888;
        font-size: 0.82rem;
    }
    </style>

    <div class="app-header">
        <h1>🛍️ Smart Recommendation Assistant</h1>
        <p>Describe what you are interested in, in your own words. The system finds the closest matching catalog items using semantic search, not exact keyword matching.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# Loading data and the embedding model. Both are cached so this only
# runs once per server session, not on every message the user sends.
# -----------------------------------------------------------------------
@st.cache_data
def load_datasets() -> dict[str, pd.DataFrame]:
    """Reads the three cleaned CSV files used by this app."""
    return {name: pd.read_csv(path) for name, path in DATASET_FILES.items()}


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Loads a small sentence embedding model for semantic search. This
    runs locally once downloaded, so it does not use the Claude API and
    does not cost anything per query.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data
def build_item_embeddings(_model: SentenceTransformer, datasets: dict,
                           max_items: int = MAX_ITEMS_PER_DATASET) -> dict:
    """
    Computes an embedding for a sample of products in each dataset. The
    full catalogs are large, so each dataset is capped at max_items
    products to keep the first run fast. A random sample still covers
    the dataset reasonably well for a demo like this one.
    """
    index = {}
    for name, df in datasets.items():
        unique_items = (
            df.drop_duplicates(subset="item_id")[["item_id", "item_text"]]
            .dropna()
            .reset_index(drop=True)
        )

        if len(unique_items) > max_items:
            unique_items = unique_items.sample(max_items, random_state=42).reset_index(drop=True)

        vectors = _model.encode(
            unique_items["item_text"].tolist(),
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32,
        )
        index[name] = {
            "item_ids": unique_items["item_id"].tolist(),
            "item_texts": unique_items["item_text"].tolist(),
            "vectors": vectors,
        }
    return index


datasets = load_datasets()
embedding_model = load_embedding_model()
item_index = build_item_embeddings(embedding_model, datasets)


# -----------------------------------------------------------------------
# Step 1: turn the conversation into one short English search phrase
# -----------------------------------------------------------------------
def extract_search_phrase(conversation: list[dict]) -> str:
    """
    Reads the whole conversation so far and asks Claude to summarize
    what the user currently wants, as a short phrase we can search
    with. Using the full conversation (not just the last message) is
    what lets follow up questions like "what else" still make sense.
    """
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation
    )
    prompt = f"""Below is a conversation between a user and a product
recommendation assistant. Based on the entire conversation, including
any earlier turns, write one short phrase of 5 to 10 words describing
what the user is currently interested in, suitable for semantic search
against a product catalog.

Conversation:
{transcript}

Respond with only the phrase, no explanation."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=40,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# -----------------------------------------------------------------------
# Step 2: find the closest matching catalog items directly
# -----------------------------------------------------------------------
def find_best_match(search_phrase: str) -> tuple[Optional[str], list[dict], str, float]:
    """
    Compares the search phrase against every indexed product using
    cosine similarity, and returns the closest matches from whichever
    dataset has the single strongest match overall.
    """
    query_vector = embedding_model.encode(search_phrase, convert_to_tensor=True)

    best_dataset = None
    best_items: list[dict] = []
    best_score = -1.0

    for name, df in datasets.items():
        entry = item_index[name]
        similarities = util.cos_sim(query_vector, entry["vectors"])[0]

        k = min(TOP_K_MATCHES, len(similarities))
        top_result = similarities.topk(k)
        top_indices = top_result.indices.tolist()
        top_scores = top_result.values.tolist()

        top_score = top_scores[0]
        if top_score > best_score:
            best_score = top_score
            best_dataset = name
            best_items = [
                {"item_text": entry["item_texts"][i], "score": round(s, 3)}
                for i, s in zip(top_indices, top_scores)
            ]

    if best_score >= CONFIDENCE_THRESHOLDS["high"]:
        confidence = "high"
    elif best_score >= CONFIDENCE_THRESHOLDS["medium"]:
        confidence = "medium"
    elif best_score > CONFIDENCE_THRESHOLDS["low"]:
        confidence = "low"
    else:
        confidence = "none"

    return best_dataset, best_items, confidence, round(max(best_score, 0.0), 3)


# -----------------------------------------------------------------------
# Step 3: format the final answer shown to the user
# -----------------------------------------------------------------------
def build_response(search_phrase: str, dataset_name: str, items: list[dict],
                    confidence: str, score: float) -> str:
    items_list = "\n".join(
        f'- {item["item_text"]}  <span class="item-score">(similarity {item["score"]})</span>'
        for item in items
    )

    badge_class = f"confidence-{confidence}"
    badge_label = CONFIDENCE_LABELS[confidence]
    category_label = DATASET_LABELS.get(dataset_name, dataset_name)

    badge_html = (
        f'<span class="confidence-badge {badge_class}">'
        f"{badge_label}, top score {score}, {category_label}"
        f"</span>"
    )

    return f'{badge_html}\n\nInterpreted request: "{search_phrase}"\n\n{items_list}'


# -----------------------------------------------------------------------
# Chat state
# -----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested prompts are only shown before the first message, so they do
# not clutter the screen once a real conversation has started.
clicked_prompt = None
if not st.session_state.messages:
    st.markdown("**Not sure where to start? Try one of these:**")
    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt_text in zip(cols, SUGGESTED_PROMPTS):
        with col:
            if st.button(prompt_text, use_container_width=True):
                clicked_prompt = prompt_text

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Main chat loop
# -----------------------------------------------------------------------
user_input = st.chat_input("e.g. something that makes my skin softer") or clicked_prompt

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Reading your request..."):
            search_phrase = extract_search_phrase(st.session_state.messages)
            dataset_name, items, confidence, score = find_best_match(search_phrase)

        if confidence == "none":
            answer = (
                '<span class="confidence-badge confidence-none">No match</span>\n\n'
                f'No close match found for "{search_phrase}" in the available data '
                "(Beauty, Video Games, Restaurants). Try describing what you want "
                "a bit differently."
            )
        else:
            answer = build_response(search_phrase, dataset_name, items, confidence, score)

        st.markdown(answer, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# -----------------------------------------------------------------------
# Sidebar, shows what the system covers and what the confidence levels
# mean, so the person using it understands the system's limits up front
# -----------------------------------------------------------------------
with st.sidebar:
    st.subheader("About this system")
    st.caption("Uses semantic search, not literal keyword matching")

    for name, df in datasets.items():
        st.metric(
            label=DATASET_LABELS.get(name, name),
            value=f"{df['item_id'].nunique():,} items",
        )

    st.divider()
    st.caption("Confidence levels")
    st.markdown(
        '<span class="confidence-badge confidence-high">Strong match</span> '
        '<span class="confidence-badge confidence-medium">Moderate match</span> '
        '<span class="confidence-badge confidence-low">Weak match</span>',
        unsafe_allow_html=True,
    )
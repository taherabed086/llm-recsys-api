from fastapi import FastAPI, HTTPException
import pandas as pd
import llm_client

app = FastAPI(title="LLM RecSys API")

CANONICAL_DATA = {
    "beauty": pd.read_csv("data/canonical_beauty.csv"),
    "video_games": pd.read_csv("data/canonical_vg.csv"),
    "yelp": pd.read_csv("data/canonical_yelp.csv"),
}


@app.get("/health")
def health_check():
    return {"status": "ok", "datasets_loaded": list(CANONICAL_DATA.keys())}


@app.get("/recommend/{dataset}/{user_id}")
def recommend(dataset: str, user_id: str, k: int = 5):
    df = CANONICAL_DATA.get(dataset)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    user_rows = df[df["user_id"] == user_id]
    if user_rows.empty:
        raise HTTPException(status_code=404, detail="User not found")

    history_texts = user_rows["item_text"].dropna().tolist()
    all_items = df.drop_duplicates(subset="item_id")[["item_id", "item_text"]]
    candidates_df = all_items.sample(min(20, len(all_items)), random_state=42)

    prompt = llm_client.build_prompt(history_texts, candidates_df["item_text"].tolist(), n_recommend=k)
    response_text = llm_client.call_llm(prompt)
    recs = llm_client.parse_llm_response(response_text, candidates_df["item_id"].tolist())

    recs_with_text = [
        {"item_id": r, "item_text": all_items[all_items["item_id"] == r]["item_text"].values[0]}
        for r in recs
    ]
    return {"user_id": user_id, "dataset": dataset, "recommendations": recs_with_text}
import anthropic
import hashlib
import json
import os
import time

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL_NAME = "claude-sonnet-4-5"
CACHE_DIR = "./llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(prompt: str) -> str:
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def call_llm(prompt: str, max_retries: int = 5, wait_seconds: int = 25) -> str:
    cache_file = _cache_path(prompt)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)["response_text"]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"}
                ]
            )
            response_text = "{" + response.content[0].text
            with open(cache_file, "w") as f:
                json.dump({"prompt": prompt, "response_text": response_text}, f)
            return response_text
        except Exception as e:
            error_msg = str(e).lower()
            if any(t in error_msg for t in ["rate_limit", "overloaded", "529", "429"]):
                print(f"Retryable error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(wait_seconds)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def build_prompt(history_texts, candidate_texts, n_recommend=5):
    history_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(history_texts)) or "(no prior history)"
    candidates_str = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_texts))
    return f"""You are a product recommendation assistant.
This user has previously interacted with:
{history_str}

Here is a list of candidate items:
{candidates_str}

Select the top {n_recommend} candidates (by number) most likely wanted next.
Respond with ONLY the JSON object below, no explanation, no other text:
{{"recommendations": [numbers]}}"""


def parse_llm_response(response_text: str, candidates: list) -> list:
    import re
    try:
        cleaned = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r'\{[^{}]*"recommendations"[^{}]*\}', response_text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []
    numbers = parsed.get("recommendations", [])
    return [candidates[n - 1] for n in numbers if 1 <= n <= len(candidates)]
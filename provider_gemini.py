# provider_gemini.py
#
# Google Gemini provider (google-generativeai SDK).
# A mandatory 2-second delay is applied before every API call to stay
# within Gemini's free-tier rate limits (60 RPM). Raise CALL_DELAY_SECS
# to 0 on a paid tier if throughput matters.
#
# Note on embedding dimensions:
#   text-embedding-004 produces 768-dimensional vectors by default.
#   If your Qdrant index was built with a different embedding model or
#   provider, re-run ingest.py with this provider before searching.
#
# Required:
#   pip install google-generativeai

import time
from typing import List

import google.generativeai as genai

# Delay applied before every Gemini API call (LLM and embedding).
# 2 seconds → ~30 RPM, safely under the 60 RPM free-tier limit.
CALL_DELAY_SECS: float = 2.0

# Supported LLM model IDs (pass as llm_model at construction time)
GEMINI_LLM_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-lite",
]

# Embedding model — text-embedding-004 is the current recommended choice.
# Output dimension: 768 (default). Adjustable via output_dimensionality param.
GEMINI_EMBED_MODEL = "models/text-embedding-004"


class GeminiProvider:
    """LLM + embedding calls routed through the Google Gemini API."""

    name = "Gemini"

    def __init__(
        self,
        api_key    : str,
        llm_model  : str = "gemini-2.0-flash",
        embed_model: str = GEMINI_EMBED_MODEL,
    ):
        genai.configure(api_key=api_key)
        self.llm_model   = llm_model
        self.embed_model = embed_model

    def llm_call(self, system: str, user: str) -> str:
        """
        Single generation call with a system instruction.
        Waits CALL_DELAY_SECS before the request.

        Supported models for system instructions: gemini-1.5-* and gemini-2.0-*.
        """
        time.sleep(CALL_DELAY_SECS)
        model = genai.GenerativeModel(
            model_name=self.llm_model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(temperature=0.0),
        )
        response = model.generate_content(user)
        return response.text

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Returns the embedding vector for `text`.
        Waits CALL_DELAY_SECS before the request.

        task_type controls how the model optimises the embedding:
          "retrieval_document" — use when embedding corpus phrases (ingest)
          "retrieval_query"    — use when embedding a search query
          "semantic_similarity", "classification", "clustering" also accepted.
        """
        time.sleep(CALL_DELAY_SECS)
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type=task_type,
        )
        return result["embedding"]

# provider_databricks.py
#
# Databricks-hosted model provider (OpenAI-compatible API).
# Works with any model served via Databricks Model Serving endpoints.
#
# Required:
#   pip install openai

from typing import List
from openai import OpenAI


class DatabricksProvider:
    """LLM + embedding calls routed through a Databricks Model Serving endpoint."""

    name = "Databricks"

    def __init__(
        self,
        base_url   : str,
        token      : str,
        llm_model  : str,
        embed_model: str,
    ):
        self._client     = OpenAI(api_key=token, base_url=base_url)
        self.llm_model   = llm_model
        self.embed_model = embed_model

    def llm_call(self, system: str, user: str) -> str:
        """Single chat-completion call. Returns the assistant message as a string."""
        resp = self._client.chat.completions.create(
            model=self.llm_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Returns the embedding vector for `text`.
        `task_type` is accepted for interface compatibility with GeminiProvider
        but is not forwarded to the OpenAI-compatible API (it has no effect here).
        """
        resp = self._client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        v = resp.data[0].embedding
        return v if isinstance(v, list) else list(v)

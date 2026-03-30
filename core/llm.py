from __future__ import annotations

import json
from typing import Any

import requests
from openai import OpenAI

from core.config import settings


class AnswerEngine:
    def __init__(self) -> None:
        self.backend = settings.llm_backend
        self._openai = OpenAI(api_key=settings.openai_api_key) if self.backend == "openai" and settings.openai_api_key else None

    def generate_answer(self, question: str, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "結論からお伝えすると、まだ確度の高い根拠が見つかっていません。PDFの取り込み後に、もう一度ご質問いただければ丁寧に確認します。"

        evidence = [
            {
                "page_no": hit.get("page_no"),
                "major_tag": hit.get("major_tag"),
                "medium_tag": hit.get("medium_tag"),
                "minor_tag": hit.get("minor_tag"),
                "shorthand_json": hit.get("shorthand_json"),
                "verified": hit.get("verified"),
                "verification_summary": hit.get("verification_summary"),
                "row_label": hit.get("row_label"),
                "col_label": hit.get("col_label"),
                "value": hit.get("value"),
            }
            for hit in hits[:4]
        ]

        if self.backend == "ollama":
            return self._ollama_answer(question, evidence)
        if self.backend == "openai" and self._openai:
            return self._openai_answer(question, evidence)
        return self._fallback_answer(evidence)

    def _fallback_answer(self, evidence: list[dict[str, Any]]) -> str:
        top = evidence[0]
        conclusion = top.get("value") or json.dumps(top.get("shorthand_json", {}), ensure_ascii=False)
        page_no = top.get("page_no")
        verification = "検証済み" if top.get("verified") else "要再確認"
        row_col = " / ".join([v for v in [top.get("row_label"), top.get("col_label")] if v])
        if row_col:
            row_col = f"（{row_col}）"
        return (
            f"結論からお伝えすると、最有力候補は {conclusion} です。"
            f"根拠は P.{page_no} {row_col} にあり、照合状態は {verification} です。"
            f"必要でしたら、このまま前後の候補や根拠JSONまでご案内します。"
        )

    def _ollama_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        prompt = self._prompt(question, evidence)
        response = requests.post(
            f"{settings.ollama_host}/api/chat",
            json={
                "model": settings.ollama_chat_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "あなたは技術資料エージェントです。結論を先に、日本語で温かく丁寧に、根拠ページと照合状態を明記して回答してください。推測は禁止です。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["message"]["content"]

    def _openai_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        prompt = self._prompt(question, evidence)
        result = self._openai.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは技術資料エージェントです。結論を先に、日本語で温かく丁寧に、根拠ページと照合状態を必ず明記してください。推測は禁止です。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return result.choices[0].message.content or self._fallback_answer(evidence)

    @staticmethod
    def _prompt(question: str, evidence: list[dict[str, Any]]) -> str:
        return (
            "ユーザー質問:\n"
            f"{question}\n\n"
            "根拠候補(JSON):\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "要件:\n"
            "1. 結論を最初の一文で述べる\n"
            "2. 推測を避ける\n"
            "3. ページ番号、行列ラベル、検証状況を含める\n"
            "4. 温かみがあり丁寧な文体にする\n"
            "5. 120文字前後で簡潔に\n"
        )

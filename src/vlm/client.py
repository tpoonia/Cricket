"""#7: one VLM interface for all backends.

Backends: "mock" (offline, default; see mock.py), "gemini" (Flash-Lite, main),
"claude" (Haiku, second family), "ollama" (Qwen3-VL, local). Strict JSON schema, modes
"label"/"rules", k runs (default 5, temperature>0), disk cache keyed by hash(model, prompt,
image/scenario). Budget guard refuses paid calls that would exceed $10 (Hard rule #5).

Real backends are code-complete but only run when the key/service is present; otherwise they
raise a clear error pointing at the mock. No key is present in this environment -> mock is used.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from src.rules.schema import validate_rules
from src.vlm import mock
from src.vlm.prompts import PITCH_LABELS, SKY_LABELS, build_prompt

CACHE_ROOT = Path("cache/vlm")
COST_LEDGER = Path("cache/cost_ledger.json")
BUDGET_USD = 10.0

# Rough per-call cost estimates (USD) for the budget guard; refined by the 10-call pilot.
EST_COST = {"gemini": 0.0008, "claude": 0.004, "ollama": 0.0, "mock": 0.0}


class BudgetExceeded(RuntimeError):
    pass


def _cost_so_far() -> float:
    if COST_LEDGER.exists():
        return json.loads(COST_LEDGER.read_text()).get("total_usd", 0.0)
    return 0.0


def _add_cost(usd: float) -> None:
    COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    total = _cost_so_far() + usd
    COST_LEDGER.write_text(json.dumps({"total_usd": total}))


class VLMClient:
    def __init__(self, backend: str = "mock", model: str | None = None, temperature: float = 0.7):
        self.backend = backend
        self.model = model or {
            "mock": "mock-v1",
            "gemini": "gemini-2.0-flash-lite",
            "claude": "claude-haiku-4-5",
            "ollama": "qwen3-vl:4b",
        }.get(backend, backend)
        self.temperature = temperature

    # ---- cache key ----
    def _key(self, mode: str, prompt: str, image_path: str | None, scenario: dict) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode())
        h.update(mode.encode())
        h.update(prompt.encode())
        if image_path and Path(image_path).exists():
            h.update(Path(image_path).read_bytes())
        else:
            h.update(json.dumps(scenario, sort_keys=True, default=str).encode())
        return h.hexdigest()[:20]

    def _cache_path(self, key: str) -> Path:
        return CACHE_ROOT / self.backend / f"{key}.json"

    # ---- public ----
    def run(self, scenario: dict, mode: str, k: int = 5) -> list[dict]:
        """Return k parsed+validated responses for a scenario (cached)."""
        image_path = scenario.get("image_path")
        prompt = build_prompt(
            mode, has_image=bool(image_path) or scenario.get("has_image", False),
            weather=scenario.get("weather"), host=scenario.get("host"), month=scenario.get("month"),
        )
        key = self._key(mode, prompt, image_path, scenario)
        cache = self._cache_path(key)
        if cache.exists():
            return json.loads(cache.read_text())["responses"]

        responses = [self._one(scenario, mode, prompt, image_path, draw) for draw in range(k)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"model": self.model, "backend": self.backend, "mode": mode, "responses": responses}
        ))
        return responses

    def _one(self, scenario, mode, prompt, image_path, draw) -> dict:
        if self.backend == "mock":
            raw = mock.mock_label(scenario, draw) if mode == "label" else mock.mock_rules(scenario, draw)
        else:
            self._budget_check()
            raw = self._call_real(mode, prompt, image_path)
            _add_cost(EST_COST.get(self.backend, 0.0))
        return self._validate(raw, mode)

    def _budget_check(self) -> None:
        if _cost_so_far() + EST_COST.get(self.backend, 0.0) > BUDGET_USD:
            raise BudgetExceeded(f"would exceed ${BUDGET_USD}; stop paid calls, continue on mock/local")

    # ---- schema validation (never trust raw model output) ----
    def _validate(self, raw: dict, mode: str) -> dict:
        out = {"backend": raw.get("backend", self.backend), "schema_ok": True}
        try:
            if mode == "label":
                pitch = raw.get("pitch")
                sky = raw.get("sky")
                if pitch not in PITCH_LABELS or sky not in SKY_LABELS:
                    out["schema_ok"] = False
                out.update({"pitch": pitch, "sky": sky, "confidence": raw.get("confidence")})
            else:
                out["rules"] = validate_rules(raw.get("rules", []))
        except Exception as e:  # noqa: BLE001
            out["schema_ok"] = False
            out["error"] = str(e)
            out.setdefault("rules", [])
        return out

    # ---- real backends (gated) ----
    def _call_real(self, mode: str, prompt: str, image_path: str | None) -> dict:
        if self.backend == "gemini":
            return self._call_gemini(prompt, image_path)
        if self.backend == "claude":
            return self._call_claude(prompt, image_path)
        if self.backend == "ollama":
            return self._call_ollama(prompt, image_path)
        raise ValueError(f"unknown backend {self.backend!r}")

    @staticmethod
    def _parse_json(text: str) -> dict:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"no JSON object in response: {text[:120]!r}")

    def _call_gemini(self, prompt: str, image_path: str | None) -> dict:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing — use backend='mock' or set the key")
        from google import genai  # type: ignore

        client = genai.Client(api_key=key)
        parts = [prompt]
        if image_path:
            parts.append(genai.types.Part.from_bytes(
                data=Path(image_path).read_bytes(), mime_type="image/png"))
        resp = client.models.generate_content(
            model=self.model, contents=parts,
            config={"temperature": self.temperature},
        )
        return self._parse_json(resp.text)

    def _call_claude(self, prompt: str, image_path: str | None) -> dict:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY missing — use backend='mock' or set the key")
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=key)
        content: list = [{"type": "text", "text": prompt}]
        if image_path:
            b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode()
            content.insert(0, {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}})
        msg = client.messages.create(
            model=self.model, max_tokens=400, temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return self._parse_json(msg.content[0].text)

    def _call_ollama(self, prompt: str, image_path: str | None) -> dict:
        import urllib.request

        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": self.temperature}}
        if image_path:
            payload["images"] = [base64.standard_b64encode(Path(image_path).read_bytes()).decode()]
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 (localhost)
            data = json.load(r)
        return self._parse_json(data["response"])

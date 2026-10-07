"""Provider-agnostic LLM client (Gemini / Anthropic / OpenAI) over plain HTTPS.

* API keys come ONLY from environment variables (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY),
  optionally loaded from a git-ignored .env file; they are never logged or cached.
* temperature 0; responses cached on disk keyed by sha256(provider, model, system, user, json flag),
  so re-running notebooks and evaluations is free and reproducible.
* retries with exponential backoff; token usage appended to data/cache/llm/usage.jsonl.
* If no key is configured, `available()` is False and callers fall back to rule-based code paths.

    python -m scholar_ir.llm_client --ping            # one tiny request to check the key/model
    python -m scholar_ir.llm_client --list-models     # Gemini models available to your key
    SCHOLAR_IR_LLM_MODEL=<model> python -m ...        # override the model for one run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests

from .config import load_config, load_dotenv

log = logging.getLogger("llm")

ENV_KEYS = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class LLMError(RuntimeError):
    pass


class LLMFatal(LLMError):
    """Non-retriable error (bad model name, bad key, bad request)."""


class LLMClient:
    last_good_global: str | None = None  # shared across instances in one process
    def __init__(self, cfg: dict | None = None, provider: str | None = None, model: str | None = None):
        load_dotenv()
        self.cfg = cfg or load_config()
        lc = self.cfg["llm"]
        self.provider = (provider or os.environ.get("SCHOLAR_IR_LLM") or lc["provider"]).lower()
        chosen = (model or os.environ.get("SCHOLAR_IR_LLM_MODEL")
                  or (lc["model"].get(self.provider) if self.provider in lc["model"] else None))
        # a list = fallback chain: if one model stays overloaded (HTTP 503), the next one is tried
        self.models = [chosen] if isinstance(chosen, str) else list(chosen or [])
        self.model = self.models[0] if self.models else None
        self.key = os.environ.get(ENV_KEYS.get(self.provider, ""), "")
        self.cache_dir = Path(lc["cache_dir"])
        if not self.cache_dir.is_absolute():
            self.cache_dir = Path(self.cfg["_root"]) / self.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = lc["temperature"]
        self.retries = lc["max_retries"]
        self.timeout = lc["timeout_seconds"]
        self.cache_hits = 0
        self.calls = 0
        self.passes = lc.get("max_passes", 3)
        self.last_good = LLMClient.last_good_global
        self.on_status = None  # optional callback(str) for progress messages (used by the GUI)

    def available(self) -> bool:
        return self.provider in ENV_KEYS and bool(self.key) and bool(self.model)

    def _cache_file(self, model: str, system: str, user: str, as_json: bool) -> Path:
        h = hashlib.sha256(json.dumps([self.provider, model, system, user, as_json]).encode()).hexdigest()
        return self.cache_dir / f"{h[:32]}.json"

    def complete(self, system: str, user: str, as_json: bool = False, max_tokens: int = 1024) -> str:
        """One completion at temperature 0 (cached). Tries the configured models in order."""
        for m in self.models:
            cf = self._cache_file(m, system, user, as_json)
            if cf.exists():
                self.cache_hits += 1
                return json.loads(cf.read_text())["text"]
        if not self.available():
            raise LLMError(f"no API key for provider {self.provider!r} (set {ENV_KEYS.get(self.provider)})")
        last = None
        # Fail over fast: on an overload (503/5xx/timeout) move straight to the next model; only after a
        # whole pass over the chain fails do we wait (2, 4, 8 s ...). The model that last answered is
        # tried first next time ("sticky"), so a busy model costs one probe per call, not several.
        order = ([self.last_good] if self.last_good in self.models else []) + \
                [m for m in self.models if m != self.last_good]
        dead: set[str] = set()
        for rnd in range(self.passes):
            for m in order:
                if m in dead:
                    continue
                self.model = m
                self._status(f"asking {m}" + (f" (round {rnd + 1})" if rnd else ""))
                try:
                    text, usage = getattr(self, f"_{self.provider}")(system, user, as_json, max_tokens)
                except LLMFatal as exc:
                    last = exc
                    dead.add(m)
                    log.info("model %s unusable (%s); skipping it", m, str(exc)[:120])
                    continue
                except (requests.RequestException, LLMError, KeyError, IndexError) as exc:
                    last = exc
                    log.info("%s busy/failed (%s); trying the next model", m, str(exc)[:90].replace("\n", " "))
                    continue
                self._cache_file(m, system, user, as_json).write_text(json.dumps(
                    {"provider": self.provider, "model": m, "text": text, "usage": usage, "time": time.time()}))
                with open(self.cache_dir / "usage.jsonl", "a") as fh:
                    fh.write(json.dumps({"model": m, **(usage or {})}) + "\n")
                self.calls += 1
                self.last_good = m
                LLMClient.last_good_global = m
                return text
            if len(dead) == len(order):
                break
            wait = min(30, 2 * 2 ** rnd)
            self._status(f"all models busy, waiting {wait}s")
            log.warning("all %d models busy (last: %s); waiting %ss", len(order), str(last)[:80].replace("\n", " "), wait)
            time.sleep(wait)
        raise LLMError(f"LLM unavailable: every model in {self.models} failed ({str(last)[:200]})")

    def _status(self, msg: str) -> None:
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:  # a UI callback must never break a call
                pass

    # ---- providers ------------------------------------------------------------------------------
    def _gemini(self, system, user, as_json, max_tokens):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        # thinking models count reasoning tokens against maxOutputTokens, so leave generous headroom
        gen = {"temperature": self.temperature, "maxOutputTokens": max(max_tokens, 4096)}
        if as_json:
            gen["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
        r = requests.post(url, json=body, headers={"x-goog-api-key": self.key}, timeout=self.timeout)
        if r.status_code != 200:
            err = LLMFatal if r.status_code in (400, 401, 403, 404) else LLMError
            raise err(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        parts = data["candidates"][0].get("content", {}).get("parts", [])
        text = "".join(pt.get("text", "") for pt in parts if not pt.get("thought"))
        if not text.strip():
            raise LLMError(f"Gemini returned no text (finishReason={data['candidates'][0].get('finishReason')})")
        u = data.get("usageMetadata", {})
        return text, {"in": u.get("promptTokenCount"), "out": u.get("candidatesTokenCount")}

    def _anthropic(self, system, user, as_json, max_tokens):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature, "system": system,
                "messages": [{"role": "user", "content": user + ("\nReturn only JSON." if as_json else "")}]}
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                          headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            err = LLMFatal if r.status_code in (400, 401, 403, 404) else LLMError
            raise err(f"Anthropic HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        u = data.get("usage", {})
        return text, {"in": u.get("input_tokens"), "out": u.get("output_tokens")}

    def _openai(self, system, user, as_json, max_tokens):
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if as_json:
            body["response_format"] = {"type": "json_object"}
        r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"})
        if r.status_code != 200:
            err = LLMFatal if r.status_code in (400, 401, 403, 404) else LLMError
            raise err(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--list-models", action="store_true", help="Gemini: list models this key can use")
    args = ap.parse_args(argv)
    c = LLMClient()
    print(f"provider={c.provider} model={c.model} key_present={bool(c.key)}")
    if args.list_models and c.provider == "gemini":
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", headers={"x-goog-api-key": c.key},
                         params={"pageSize": 1000}, timeout=60)
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print("  ", m["name"].replace("models/", ""))
    if args.ping:
        print(c.complete("You are terse.", 'Reply with the JSON {"ok": true}.', as_json=True, max_tokens=50))


if __name__ == "__main__":
    main()

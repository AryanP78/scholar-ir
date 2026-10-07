"""Natural-language request -> structured query (the LLM only TRANSLATES; it never ranks or sees the corpus).

    "recent papers on dense retrieval for QA since 2021 in IR"
      -> {"terms": [...], "phrases": ["dense retrieval"], "zone": "all", "year_min": 2021,
          "year_max": null, "categories": ["cs.IR"], "intent": "recent"}
      -> 'question answering "dense retrieval" year:>=2021 cat:cs.IR'   (our query language)

The JSON is validated with pydantic. Invalid JSON -> one retry -> fall back to the rule-based NL
parser below, which is also the baseline in the parser evaluation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .llm_client import LLMClient, LLMError

log = logging.getLogger("llm_parser")

CATEGORY_NAMES = {
    "cs.IR": "information retrieval, search, recommender systems",
    "cs.CL": "computation and language, NLP, language models",
    "cs.LG": "machine learning",
    "cs.AI": "artificial intelligence, agents, reasoning, planning",
    "cs.CV": "computer vision, images, video",
    "cs.DB": "databases, query processing, SQL",
    "cs.SI": "social and information networks, graphs, social media",
}


class StructuredQuery(BaseModel):
    terms: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)
    zone: Literal["all", "title", "abstract"] = "all"
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    categories: list[str] = Field(default_factory=list)
    intent: Literal["foundational", "recent", "specific", "exploratory"] = "specific"

    @field_validator("categories")
    @classmethod
    def _known_categories(cls, v):
        return [c for c in v if c in CATEGORY_NAMES]

    @field_validator("year_min", "year_max")
    @classmethod
    def _sane_year(cls, v):
        return v if v is None or 1990 <= v <= 2100 else None

    def compile(self) -> str:
        """Compile to the structured query language of query_parser.py."""
        zone = "" if self.zone == "all" else f"{self.zone}:"
        parts = [f"{zone}{t}" for t in self.terms if t.strip()]
        parts += [f'{zone}"{ph}"' for ph in self.phrases if ph.strip()]
        if self.year_min and self.year_max:
            parts.append(f"year:{self.year_min}..{self.year_max}")
        elif self.year_min:
            parts.append(f"year:>={self.year_min}")
        elif self.year_max:
            parts.append(f"year:<={self.year_max}")
        parts += [f"cat:{c}" for c in self.categories]
        return " ".join(parts)


SYSTEM_PROMPT = """You translate a researcher's plain-English request for computer-science papers into a JSON search query.
Return ONLY a JSON object with these keys:
  "terms": list of single content words to search for (no stop words, no filler such as "papers", "find", "work on"),
  "phrases": list of multi-word technical expressions that must appear exactly (e.g. "dense retrieval"); do not repeat their words in terms,
  "zone": "title" if the user wants words in the title, "abstract" if in the abstract, else "all",
  "year_min": integer or null (e.g. "since 2021" -> 2021, "after 2019" -> 2020, "from the last N years" -> CURRENT_YEAR-N+1),
  "year_max": integer or null (e.g. "before 2015" -> 2014, "up to 2018" -> 2018),
  "categories": list from [cs.IR, cs.CL, cs.LG, cs.AI, cs.CV, cs.DB, cs.SI] ONLY if the user names a field explicitly, else [],
  "intent": "foundational" (seminal/classic/survey/well-established work), "recent" (latest/new/emerging work),
            "specific" (a precise technical question), or "exploratory" (a broad look around a topic).
Categories: """ + "; ".join(f"{k} = {v}" for k, v in CATEGORY_NAMES.items()) + """
CURRENT_YEAR = {year}. Do not add search terms the user did not imply."""


# ---- rule-based NL parser (fallback + evaluation baseline) ---------------------------------------

_FILLER = {"paper", "papers", "find", "show", "me", "work", "works", "research", "article", "articles", "about",
           "on", "for", "the", "a", "an", "of", "in", "with", "and", "to", "what", "are", "is", "some", "any",
           "looking", "i", "want", "need", "please", "related", "regarding", "using", "from", "that", "which",
           "publications", "studies", "study", "approaches", "methods", "techniques", "since", "after", "before",
           "between", "year", "years", "last", "past", "recent", "latest", "new", "seminal", "classic", "survey",
           "surveys", "overview", "foundational", "early", "published", "field", "area", "category", "title",
           "abstract", "words", "word", "mention", "mentions", "mentioning", "only", "can", "you", "get", "list",
           "give", "by", "or", "up", "until", "older", "than", "newer", "influential", "key", "important",
           "landmark", "good", "best", "top", "highly", "cited", "state", "art", "sota", "emerging", "current",
           "how", "does", "do", "did", "why", "when", "where", "who", "whom", "such", "as", "it", "its", "they",
           "their", "there", "this", "these", "those", "into", "be", "been", "used", "use", "main", "common"}
_INTENT_CUES = {"foundational": ["seminal", "classic", "foundational", "survey", "overview", "landmark",
                                 "influential", "highly cited", "well-known", "introduction", "tutorial", "history"],
                "recent": ["recent", "latest", "new", "emerging", "state-of-the-art", "state of the art", "current",
                           "this year", "last year"]}
_CAT_CUES = {"cs.IR": ["information retrieval", " ir ", "cs.ir"], "cs.CL": ["nlp", "computational linguistics", "cs.cl"],
             "cs.CV": ["computer vision", "cs.cv"], "cs.DB": ["database", "databases", "cs.db"],
             "cs.LG": ["machine learning category", "cs.lg"], "cs.AI": ["cs.ai"], "cs.SI": ["social networks category", "cs.si"]}


def rule_parse(text: str, current_year: int) -> StructuredQuery:
    t = " " + text.lower() + " "
    sq = StructuredQuery()
    m = re.search(r"between (\d{4}) and (\d{4})", t) or re.search(r"from (\d{4}) to (\d{4})", t)
    if m:
        sq.year_min, sq.year_max = int(m.group(1)), int(m.group(2))
    else:
        if (m := re.search(r"(?:since|from|in or after) (\d{4})", t)):
            sq.year_min = int(m.group(1))
        if (m := re.search(r"after (\d{4})", t)):
            sq.year_min = int(m.group(1)) + 1
        if (m := re.search(r"before (\d{4})", t)):
            sq.year_max = int(m.group(1)) - 1
        if (m := re.search(r"(?:up to|until|no later than) (\d{4})", t)):
            sq.year_max = int(m.group(1))
        if (m := re.search(r"(?:last|past) (\d+) years", t)):
            sq.year_min = current_year - int(m.group(1)) + 1
        if (m := re.search(r"\bin (\d{4})\b", t)) and sq.year_min is None and sq.year_max is None:
            sq.year_min = sq.year_max = int(m.group(1))
    for cat, cues in _CAT_CUES.items():
        if any(c in t for c in cues):
            sq.categories.append(cat)
    if " in the title" in t or " title " in t:
        sq.zone = "title"
    elif " in the abstract" in t:
        sq.zone = "abstract"
    sq.phrases = re.findall(r'"([^"]+)"', text)
    found = any(c in t for c in _INTENT_CUES["foundational"])
    recent = any(c in t for c in _INTENT_CUES["recent"]) or (sq.year_min is not None and sq.year_min >= current_year - 3)
    sq.intent = "foundational" if found and not recent else "recent" if recent and not found else "specific"
    rest = re.sub(r'"[^"]+"', " ", t)
    rest = re.sub(r"\b\d{4}\b|cs\.[a-z]{2}", " ", rest)
    for cues in _CAT_CUES.values():
        for c in cues:
            rest = rest.replace(c, " ")
    words = re.findall(r"[a-z0-9][a-z0-9\-]*", rest)
    sq.terms = [w for w in words if w not in _FILLER and not w.isdigit()]
    return sq


# ---- LLM parser ---------------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in LLM output")
    return json.loads(m.group(0))


def llm_parse(text: str, client: LLMClient, current_year: int) -> tuple[StructuredQuery, str]:
    """Returns (structured query, source) where source is 'llm', 'llm-retry' or 'rule-fallback'."""
    if not client.available():
        return rule_parse(text, current_year), "rule-fallback"
    system = SYSTEM_PROMPT.replace("{year}", str(current_year))
    for attempt, user in enumerate((f"Request: {text}",
                                    f"Request: {text}\nYour previous answer was not valid JSON for the schema. "
                                    "Return ONLY the JSON object.")):
        try:
            raw = client.complete(system, user, as_json=True, max_tokens=400)
            return StructuredQuery(**_extract_json(raw)), "llm" if attempt == 0 else "llm-retry"
        except (ValueError, ValidationError, LLMError, TypeError) as exc:
            log.warning("LLM parse attempt %d failed: %s", attempt + 1, str(exc)[:200])
    return rule_parse(text, current_year), "rule-fallback"


INTENT_PROMPT = """Classify what kind of papers a researcher wants for this search query.
Answer with JSON {"intent": X} where X is one of:
"foundational" (seminal, classic, highly-regarded or survey work), "recent" (newest work, current trends),
"specific" (a precise technical question), "exploratory" (broad look around a topic)."""


def llm_intent(text: str, client: LLMClient) -> str | None:
    """LLM intent label used as an optional feature of the adaptive lambda."""
    if not client.available():
        return None
    try:
        raw = client.complete(INTENT_PROMPT, f"Query: {text}", as_json=True, max_tokens=50)
        val = _extract_json(raw).get("intent")
        return val if val in ("foundational", "recent", "specific", "exploratory") else None
    except (ValueError, LLMError):
        return None

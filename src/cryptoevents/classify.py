"""Keyword rules that decide whether a text is about crypto.

Two rules on purpose. `naive_match` is the obvious substring search, kept as a
baseline. `rule_match` strips links and @handles first and only accepts whole
words, which is what the event list uses. Both are scored against hand labels.
"""
from __future__ import annotations

import html
import re

# Whole-word terms. Each entry is a regex fragment matched case-insensitively.
TERMS = {
    "bitcoin": r"bitcoins?",
    "crypto": r"crypto(currenc(y|ies))?|crypto[- ]assets?",
    "blockchain": r"blockchains?",
    "digital asset": r"digital[- ]assets?",
    "stablecoin": r"stablecoins?",
    "cbdc": r"cbdcs?|central bank digital currenc(y|ies)",
    "virtual currency": r"virtual currenc(y|ies)",
    "libra": r"libra",
    "ethereum": r"ethereum|ether",
    "nft": r"nfts?|non[- ]fungible tokens?",
    "memecoin": r"meme ?coins?",
}
NAIVE = re.compile(r"bitcoin|crypto|blockchain|libra|digital asset|stablecoin|nft|coin", re.I)
_RULE = re.compile(r"(?<![\w@])(" + "|".join(TERMS.values()) + r")(?!\w)", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_HANDLE = re.compile(r"@\w+")
_TAG = re.compile(r"<[^>]+>")
# Job titles are not crypto news: "Special Advisor for AI and Crypto" shows up
# in many unrelated executive orders.
_TITLES = re.compile(r"(special )?advisor for ai (and|&) crypto", re.I)


def clean_text(text: str) -> str:
    """Drop links, @handles, HTML and job titles, collapse whitespace."""
    text = html.unescape(_TAG.sub(" ", text or ""))
    text = re.sub(r"\s+", " ", text)
    text = _TITLES.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def naive_match(text: str) -> bool:
    return bool(NAIVE.search(text or ""))


def matched_terms(text: str) -> list[str]:
    found = []
    for m in _RULE.finditer(clean_text(text)):
        word = m.group(0).lower()
        for name, pattern in TERMS.items():
            if re.fullmatch(pattern, word, re.I) and name not in found:
                found.append(name)
    return found


def rule_match(text: str) -> bool:
    return bool(matched_terms(text))


def count_hits(text: str) -> int:
    return len(_RULE.findall(clean_text(text)))


def document_rule(title: str, excerpts: str, min_terms: int = 2) -> bool:
    """A presidential document counts if its title names a crypto term, or the
    search excerpts from its full text contain at least `min_terms` different
    crypto terms. One passing mention (a scam warning, a list of examples) is
    not enough."""
    return rule_match(title) or len(matched_terms(excerpts)) >= min_terms


def precision(predicted: list[bool], truth: list[bool]) -> tuple[float, int, int]:
    """Share of predicted positives that the hand labels confirm."""
    tp = sum(1 for p, t in zip(predicted, truth) if p and t)
    pos = sum(1 for p in predicted if p)
    return (tp / pos if pos else float("nan"), tp, pos)

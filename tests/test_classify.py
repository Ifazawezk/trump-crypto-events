import pytest

from cryptoevents.classify import (clean_text, document_rule, matched_terms, naive_match,
                                   precision, rule_match)


@pytest.mark.parametrize("text", [
    "I am not a fan of Bitcoin and other Cryptocurrencies",
    "Facebook Libra's virtual currency will have little standing",
    "Unregulated Crypto Assets can facilitate unlawful behavior",
    "Big day for #Bitcoin",
    "a new framework for stablecoins",
])
def test_rule_accepts_crypto_text(text):
    assert rule_match(text)


@pytest.mark.parametrize("text", [
    "What a coincidence!",
    "the Ronald Reagan Library",
    "Kirsten Gillibrand dropped out",
    "@CryptOhio: Melania would be the best",
    "Love it! http://t.co/NFTwpTlrPm",
    "signs the Commemorative Coin Act",
    "the Special Advisor for AI and Crypto shall review",
    "Advisor for AI &amp; <span class=\"match\">Crypto</span>, the Assistant",
])
def test_rule_rejects_false_positives(text):
    assert not rule_match(text)


def test_naive_baseline_is_fooled():
    assert naive_match("What a coincidence") and naive_match("the library")


def test_clean_text_strips_links_handles_tags():
    assert clean_text("<b>Hi</b> @someone https://t.co/x  there") == "Hi there"


def test_matched_terms_are_named_once():
    assert matched_terms("Bitcoin, bitcoin and blockchain") == ["bitcoin", "blockchain"]


def test_document_rule_needs_title_or_two_terms():
    assert document_rule("Establishment of the Strategic Bitcoin Reserve", "")
    assert document_rule("Some Order", "digital assets ... blockchain-based services")
    assert not document_rule("Consumer Week", "growth of cryptocurrency")


def test_precision():
    assert precision([True, True, False], [True, False, True]) == (0.5, 1, 2)

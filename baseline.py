"""
Rule-based keyword-matching baseline.

Two jobs:

1. `extract()` — the comparison baseline for the project. Finds skill mentions
   by dictionary lookup over the taxonomy's surface forms. This is what the
   fine-tuned BERT model has to beat on Precision / Recall / F1.

2. `bio_tags()` — a *weak labeler*. The Kaggle datasets are not hand-annotated
   for skill spans, so we bootstrap NER training labels by running this matcher
   over tokenized text and emitting BIO tags. The BERT model then learns to
   generalize beyond the dictionary (catching variants the matcher misses).

Matching is longest-form-first and whole-token (word-boundary) to avoid
partial-word hits — e.g. "java" must not fire inside "javascript".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import Category, Skill, surface_to_skill


@dataclass(frozen=True)
class Match:
    """A located skill mention."""
    skill: Skill
    surface: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive

    @property
    def category(self) -> Category:
        return self.skill.category


class KeywordMatcher:
    """Compiles the taxonomy into one regex per surface form."""

    # characters allowed to sit on either side of a match without breaking the
    # word boundary (handles C++, Security+, etc. which contain symbols)
    _BOUNDARY = r"(?<![A-Za-z0-9+#])(?:{})(?![A-Za-z0-9+#])"

    def __init__(self) -> None:
        self._lookup = surface_to_skill()
        # sort longest-first so "machine learning" wins over "learning"
        forms = sorted(self._lookup.keys(), key=len, reverse=True)
        # one big alternation, escaped; longest-first preserves precedence
        alternation = "|".join(re.escape(f) for f in forms)
        self._pattern = re.compile(
            self._BOUNDARY.format(alternation), flags=re.IGNORECASE
        )

    def extract(self, text: str) -> list[Match]:
        """Return non-overlapping skill matches, left-to-right, longest-first."""
        matches: list[Match] = []
        seen_spans: list[tuple[int, int]] = []
        for m in self._pattern.finditer(text):
            start, end = m.start(), m.end()
            # skip if this span overlaps one we already took (longest wins
            # because finditer + longest-first alternation hits longer first)
            if any(s < end and start < e for s, e in seen_spans):
                continue
            surface = m.group(0).lower()
            skill = self._lookup.get(surface)
            if skill is None:
                # case-folding edge: re-resolve via lowercase
                skill = self._lookup.get(surface.lower())
            if skill is None:
                continue
            matches.append(Match(skill, m.group(0), start, end))
            seen_spans.append((start, end))
        matches.sort(key=lambda mt: mt.start)
        return matches

    def bio_tags(self, tokens: list[str], offsets: list[tuple[int, int]],
                 text: str) -> list[str]:
        """
        Emit a BIO tag per token by aligning char-level matches to token
        offsets. `offsets` are (start, end) char spans per token, as produced
        by a fast tokenizer's offset mapping.
        """
        tags = ["O"] * len(tokens)
        matches = self.extract(text)
        for match in matches:
            cat = match.category.value
            first = True
            for i, (tok_s, tok_e) in enumerate(offsets):
                if tok_s == tok_e:  # special tokens have empty spans
                    continue
                # token overlaps the matched span
                if tok_s < match.end and match.start < tok_e:
                    tags[i] = (f"B-{cat}" if first else f"I-{cat}")
                    first = False
        return tags


def summarize(matches: list[Match]) -> dict[str, int]:
    """Count matched skills per category — handy for quick EDA."""
    counts: dict[str, int] = {c.value: 0 for c in Category}
    for m in matches:
        counts[m.category.value] += 1
    return counts


if __name__ == "__main__":
    sample = (
        "We seek a mechanical engineer proficient in SolidWorks and FEA, with "
        "strong knowledge of GD&T and heat transfer. PE license preferred. "
        "Must have excellent communication skills and experience with Python "
        "and MATLAB. Familiarity with C++ is a plus."
    )
    matcher = KeywordMatcher()
    found = matcher.extract(sample)
    print(f"Found {len(found)} skills:\n")
    for m in found:
        print(f"  [{m.category.value:9s}] {m.surface:20s} "
              f"-> {m.skill.canonical}  ({m.start}:{m.end})")
    print("\nBy category:", summarize(found))

"""Tests for deterministic YouTube formatting guideline checks."""

from __future__ import annotations

import datetime

import srt

from subtitle_studio.detect.guidelines import check_guidelines, check_near_duplicate_boundary


def _sub(index: int, start_s: float, end_s: float, content: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=datetime.timedelta(seconds=start_s),
        end=datetime.timedelta(seconds=end_s),
        content=content,
    )


class TestCheckNearDuplicateBoundary:
    def test_flags_real_duplicate_take(self) -> None:
        """Reproduces the veille-tech.mp4 case: a CTA line re-recorded and left
        uncut, transcribed correctly as two adjacent segments with different
        wording but a long shared clause."""
        subs = [
            _sub(
                12,
                49.81,
                54.66,
                "écris skills en commentaire et je te l'envoie un\nDM. Et si ce type de contenu t'intéresse,",
            ),
            _sub(
                13,
                54.68,
                58.60,
                "si ce type de contenu sur l'IA vous intéresse,\ndites le moi en commentaire et j'essaierai",
            ),
        ]
        violations = check_near_duplicate_boundary(subs)
        assert len(violations) == 1
        assert violations[0].segment == 13
        assert violations[0].rule == "near_duplicate"
        assert violations[0].severity == "warning"
        assert "si ce type de contenu" in violations[0].description

    def test_flags_fully_identical_adjacent_segments(self) -> None:
        """The simplest re-recorded-take case: the exact same line twice."""
        content = "Et si ce type de contenu vous intéresse vraiment, dites-le moi."
        subs = [
            _sub(1, 0, 3, content),
            _sub(2, 3, 6, content),
        ]
        violations = check_near_duplicate_boundary(subs)
        assert len(violations) == 1
        assert violations[0].segment == 2
        assert " ".join(content.lower().split()) in violations[0].description

    def test_exact_threshold_boundary_is_inclusive(self) -> None:
        """match.size >= min_overlap_chars: exactly at the threshold must still flag."""
        a = "le pipeline de déploiement est prêt pour la prod"
        b = "le pipeline de déploiement est prêt à partir"
        # The longest common substring between a and b is exactly 36 chars
        # ("le pipeline de déploiement est prêt "), verified independently.
        subs = [_sub(1, 0, 2, a), _sub(2, 2, 4, b)]
        assert len(check_near_duplicate_boundary(subs, min_overlap_chars=36)) == 1
        assert check_near_duplicate_boundary(subs, min_overlap_chars=37) == []

    def test_no_violation_below_threshold(self) -> None:
        """Short, generic overlaps (connectors, articles) must not trigger."""
        subs = [
            _sub(1, 0, 2, "C'est un peu comme une tâche."),
            _sub(2, 2, 4, "C'est complètement différent du reste."),
        ]
        assert check_near_duplicate_boundary(subs) == []

    def test_unrelated_adjacent_segments_no_violation(self) -> None:
        subs = [
            _sub(1, 0, 2, "Bonjour le monde."),
            _sub(2, 2, 4, "Comment ça va aujourd'hui ?"),
        ]
        assert check_near_duplicate_boundary(subs) == []

    def test_custom_threshold_catches_shorter_overlap(self) -> None:
        subs = [
            _sub(1, 0, 2, "je pense que le déploiement est prêt"),
            _sub(2, 2, 4, "le déploiement est prêt pour la prod"),
        ]
        assert check_near_duplicate_boundary(subs, min_overlap_chars=30) == []
        violations = check_near_duplicate_boundary(subs, min_overlap_chars=20)
        assert len(violations) == 1
        assert violations[0].segment == 2

    def test_match_is_case_insensitive(self) -> None:
        subs = [
            _sub(1, 0, 2, "Si ce type de contenu vous intéresse,"),
            _sub(2, 2, 4, "SI CE TYPE DE CONTENU vous a plu,"),
        ]
        violations = check_near_duplicate_boundary(subs)
        assert len(violations) == 1

    def test_skips_pairs_sharing_the_same_index(self) -> None:
        """CPS auto-split halves keep the original index until reindexing at
        write time (R7). If the split sentence contained a real repeated
        clause (a stutter/retake Whisper folded into one dense segment), the
        two halves must never be reported as duplicating each other under a
        self-referential 'segment N shares with segment N' message. Without
        the same-index guard, this pair shares a 49-char substring — well
        above the default threshold — so this test would fail if the guard
        were removed."""
        subs = [
            _sub(7, 0, 3.5, "et si ce type de contenu vous interesse vraiment,"),
            _sub(7, 3.5, 7, "et si ce type de contenu vous interesse vraiment, dites le moi"),
        ]
        assert check_near_duplicate_boundary(subs) == []

    def test_sorts_by_start_before_comparing(self) -> None:
        """Input order should not matter — comparison follows timeline order."""
        subs = [
            _sub(
                13,
                54.68,
                58.60,
                "si ce type de contenu sur l'IA vous intéresse,",
            ),
            _sub(
                12,
                49.81,
                54.66,
                "Et si ce type de contenu t'intéresse,",
            ),
        ]
        violations = check_near_duplicate_boundary(subs)
        assert len(violations) == 1
        assert violations[0].segment == 13


class TestCheckGuidelinesIncludesNearDuplicate:
    def test_aggregates_near_duplicate_violations(self) -> None:
        subs = [
            _sub(1, 0, 5, "Et si ce type de contenu t'intéresse,"),
            _sub(2, 5, 10, "si ce type de contenu sur l'IA vous intéresse,"),
        ]
        violations = check_guidelines(subs)
        near_dup = [v for v in violations if v.rule == "near_duplicate"]
        assert len(near_dup) == 1
        assert near_dup[0].segment == 2

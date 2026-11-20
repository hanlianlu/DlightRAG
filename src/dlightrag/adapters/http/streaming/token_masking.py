# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe-prefix release of a streamed answer: what a reader may see before the rest arrives.

A ``token`` event carries one piece of a growing answer, and a protected Skill document can
cross any piece boundary. A stateless content gate cannot find a passage split across two
frames, so this module carries a fingerprint window of the stream's tail and scans it
together with each new piece, releasing only text no protected run can still claim. Both SSE
transports — the browser's rendered frames and REST's canonical ones — read through the same
component, because the property is a property of the event stream, not of either projection.

Two thresholds, deliberately different:

* :data:`~dlightrag.engine.agent.skill_protection.FINGERPRINT_CHARS` (40) is *detection*.
  A verified run that short may still grow in the next piece, so nothing it covers is
  released until its end is known. Detecting only at the finished answer's threshold would
  release the leading characters of a run that a later piece makes long enough to replace.
* :data:`~dlightrag.engine.agent.skill_protection.MIN_MASK_CHARS` (80) is *replacement*,
  the threshold the finished answer uses. A run that ends below it is shown verbatim in the
  finished answer, so the draft must show it too; a run that ends at or above it becomes
  ``[...]``. While the verdict is undecided the run is held rather than guessed at, because a
  placeholder emitted for what turns out to be a short quotation could not be retracted. The
  index itself answers this question, at the threshold and in the coordinates it masks in, so
  the draft and the settled answer cannot disagree about a borderline run.

The holdback is measured in *visible* characters: the index normalizes whitespace runs to
one character and drops invisible characters, so a holdback counted in raw characters can
normalize to less than one fingerprint window and let a run slip out a piece early.

State belongs to one subscription. A reconnect starts from the cursor's text, so a fresh
instance is what a new follower builds; two readers never share a carry.
"""

from dlightrag.engine.agent.skill_protection import (
    FINGERPRINT_CHARS,
    MASK_PLACEHOLDER,
    MIN_MASK_CHARS,
    ProtectedTextIndex,
)

#: Visible characters one token-frame renderer withholds from the end of the stream.
#:
#: The two extra characters are the margin a raw slice boundary can lose to whitespace
#: collapsing: the index matches in normalized coordinates, so the carry has to survive
#: normalization with a whole fingerprint window still in it.
STREAM_HOLD_VISIBLE_CHARS = FINGERPRINT_CHARS + 2

#: Events after which no text this subscription holds can still be shown.
#:
#: ``reset`` replaces the draft the subscriber reads; ``done`` and ``error`` end the stream.
#: A carried tail cannot ride on any of those frames, and the final answer a reader keeps is
#: the terminal presentation, so the state is dropped rather than flushed into a frame that
#: does not own it.
FORGET_EVENT_TYPES = frozenset({"reset", "done", "error"})


def _proves_run_end(buffer: str, run_end: int) -> bool:
    """Whether a verified run that stops before the buffer's end has shown a real mismatch.

    The index normalizes a whitespace run to one character and maps it back to the run's first
    raw character, so a buffer ending in (or just after) whitespace reports a run ``text_end``
    before the buffer's last raw character even when the passage continues in text not yet
    read. A run ending only where the normalization has nothing left to compare is therefore
    not proven to end: the next raw character decides, and until one arrives the safe reading
    is that the run may continue.
    """
    return any(
        character.isprintable() and not character.isspace() for character in buffer[run_end:]
    )


def _carries_a_normalized_character(character: str) -> bool:
    """Whether this raw character contributes at least one normalized character.

    A footprint of the normalization without importing it: whitespace runs collapse (so a
    whitespace character may contribute nothing), control and format characters are dropped,
    and every printable character contributes at least itself. Counting only the characters
    that certainly survive keeps the holdback long enough rather than too short for the text a
    reader is actually shown; a deliberately entity-encoded passage is a stated limit of the
    streaming projection, and the settled answer's gate reads it as its characters.
    """
    return character.isprintable() and not character.isspace()


def _carry_start(text: str) -> int:
    """The raw index whose suffix holds the last :data:`STREAM_HOLD_VISIBLE_CHARS` visible characters.

    Returns 0 when the text has fewer, which holds the whole text: a frame can always wait
    for the next batch, and waiting is the safe direction.
    """
    visible = 0
    for offset in range(len(text) - 1, -1, -1):
        if _carries_a_normalized_character(text[offset]):
            visible += 1
            if visible >= STREAM_HOLD_VISIBLE_CHARS:
                return offset
    return 0


class TokenFrameMasker:
    """Release a token stream in order, never handing out a passage that may be protected.

    Each call joins the carried tail to the new piece, verifies the protected runs the index
    finds in it, and returns the prefix that is safe to show now. Three states describe what
    the carry is:

    ``plain``
        The carry is the stream's last few visible characters. A completed run is replaced
        or released as its verified length decides.
    ``pending``
        The carry begins at a detected run's first character and the run has not ended. Its
        reader-visible length is undecided, so it is held whole — replacing it now could hide
        a short quotation the finished answer will show, and releasing it could expose the
        first characters of a passage that grows past the replacement threshold.
    ``inside``
        A run long enough to be replaced has been replaced once, and the carry follows its
        tail until a later piece verifies where it ends. This is the only state in which a
        ``[...]`` has already been released for a run whose end is still unseen, and it is
        safe because a run this long is replaced on every reader surface.
    """

    def __init__(self, index: ProtectedTextIndex) -> None:
        self._index = index
        self._carry = ""
        self._pending_run = False
        self._inside_run = False

    def reset(self) -> None:
        """Forget everything the stream has carried; the next piece starts a fresh draft."""
        self._carry = ""
        self._pending_run = False
        self._inside_run = False

    def release(self, text: str) -> str:
        """The part of ``text`` this reader may be shown now, given what the carry holds."""
        if not text:
            return ""
        buffer = self._carry + text
        if self._pending_run:
            return self._release_pending_run(buffer)
        if self._inside_run:
            return self._release_from_run(buffer)
        return self._release_plain(buffer)

    def _replaces_run(self, run: str) -> bool:
        """Whether the settled answer replaces this verified run rather than showing it.

        The question is put to the index instead of measured here: the same scan, threshold
        and normalization decide the finished answer, so a run the reader ends up seeing
        verbatim is never hidden in the draft, and one that ends up replaced is never leaked
        while its end is still in flight.
        """
        return bool(self._index.scan(run, minimum=MIN_MASK_CHARS))

    def _release_plain(self, buffer: str) -> str:
        """Release a buffer's safe head, carrying the tail the index could still find a run in.

        Every verified run that starts before the release point is found in this buffer, so a
        raw character can only be released when no protected run starts in front of it.
        """
        release_limit = _carry_start(buffer)
        if release_limit <= 0:
            self._carry = buffer
            return ""
        pieces: list[str] = []
        cursor = 0
        for finding in self._index.scan(buffer, minimum=FINGERPRINT_CHARS):
            if finding.text_start >= release_limit:
                break
            pieces.append(buffer[cursor : finding.text_start])
            if finding.text_end >= len(buffer) or not _proves_run_end(buffer, finding.text_end):
                # The run reaches—or only whitespace separates it from—the end of what this
                # subscription has read, so whether the reader will see it replaced or
                # verbatim is not yet known.
                if self._replaces_run(buffer[finding.text_start :]):
                    # Already long enough that every reader surface replaces it; the
                    # placeholder is final wherever the passage ends, and following only
                    # its tail keeps a document-long run out of memory.
                    pieces.append(MASK_PLACEHOLDER)
                    self._inside_run = True
                    self._carry = buffer[_carry_start(buffer) :]
                else:
                    # A run this short may still end below the replacement threshold,
                    # where the finished answer shows it verbatim. A placeholder now
                    # could only be retracted, so the run is held whole instead.
                    self._pending_run = True
                    self._carry = buffer[finding.text_start :]
                return "".join(pieces)
            if self._replaces_run(buffer[finding.text_start : finding.text_end]):
                pieces.append(MASK_PLACEHOLDER)
            else:
                # A verified run below the threshold reads verbatim in the finished
                # answer, so the draft shows it too; withholding it would delay text the
                # reader is owed and change what the answer looks like mid-stream.
                pieces.append(buffer[finding.text_start : finding.text_end])
            cursor = finding.text_end
        end = max(cursor, release_limit)
        pieces.append(buffer[cursor:end])
        self._carry = buffer[end:]
        return "".join(pieces)

    def _release_pending_run(self, buffer: str) -> str:
        """Measure the run the carry begins with, and release it only once its end is known.

        The carried run starts at the buffer's first character. Until a later piece shows
        where it ends, releasing any of it could expose the head of a long passage; masking
        it could hide a short quotation the finished answer shows. The buffer is held whole
        on either uncertainty.
        """
        findings = self._index.scan(buffer, minimum=FINGERPRINT_CHARS)
        first = findings[0] if findings else None
        if first is None or first.text_start > 0:
            # Nothing verifies the carried run's own start, so its end cannot be placed
            # either: hold everything rather than release a span whose extent is unknown.
            self._carry = buffer
            return ""
        if first.text_end >= len(buffer) or not _proves_run_end(buffer, first.text_end):
            if not self._replaces_run(buffer):
                self._carry = buffer
                return ""
            # It has grown past the replacement threshold while still continuing, so its
            # masked form is final; emit it once and follow the tail until it ends.
            self._pending_run = False
            self._inside_run = True
            self._carry = buffer[_carry_start(buffer) :]
            return MASK_PLACEHOLDER
        # The run ended inside this buffer, and its verified length now decides what the
        # finished answer would show: the placeholder, or the quotation itself.
        self._pending_run = False
        self._carry = ""
        run = buffer[: first.text_end]
        replaced = MASK_PLACEHOLDER if self._replaces_run(run) else run
        return replaced + self._release_plain(buffer[first.text_end :])

    def _release_from_run(self, buffer: str) -> str:
        """Follow a passage already replaced until this buffer shows where it ends.

        The carry begins inside the run, so it holds no text the reader is owed. Release
        resumes on the text after a verified end that lies at or before the carry; anything
        else is held whole. Holding more than the tail is what a buffer whose run cannot be
        re-identified gets: the pass may only release text it has positively proven, and a
        run whose reported span shifts between batches is not proof of where it ended.
        """
        carried = len(self._carry)
        findings = self._index.scan(buffer, minimum=FINGERPRINT_CHARS)
        first = findings[0] if findings else None
        if (
            first is not None
            and first.text_start <= carried
            and first.text_end < len(buffer)
            and _proves_run_end(buffer, first.text_end)
        ):
            self._inside_run = False
            self._carry = ""
            return self._release_plain(buffer[first.text_end :])
        if first is not None and first.text_start <= carried:
            # The run still reaches the end of this buffer: nothing is verified past it.
            self._carry = buffer[_carry_start(buffer) :]
            return ""
        # No finding starts inside the carried text: hold the whole buffer until one does.
        self._carry = buffer
        return ""


__all__ = [
    "FORGET_EVENT_TYPES",
    "STREAM_HOLD_VISIBLE_CHARS",
    "TokenFrameMasker",
]

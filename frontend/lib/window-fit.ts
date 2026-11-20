// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * How much room a page has left below something that stands in it.
 *
 * A window inside a page must not be taller than the page's remaining space: capped by the viewport
 * instead, a fold that opens below the top of its page hangs past the end of it and the page — not
 * the window — becomes the thing that scrolls. The room cannot come from CSS (a `details` lays its
 * content out in a slot, so no flex chain reaches the window), so it is measured here.
 *
 * Both edges are taken against the scroller's own top, so the answer does not move when the page is
 * scrolled: adding the scroll position back instead — which is what this did first — makes the
 * window grow as a reader reads, and a height that depends on the scroll position never settles.
 */
interface Box {
  /** The scroller's visible height. */
  height: number;
  /** What the thing standing in the page has for a top, in viewport coordinates. */
  top: number;
  /** The scroller's own top, in the same coordinates. */
  scrollerTop: number;
  /** The scroller's bottom padding, kept clear of the window. */
  paddingBottom: number;
}

export function roomBelow({height, top, scrollerTop, paddingBottom}: Box): number {
  return height - (top - scrollerTop) - paddingBottom;
}

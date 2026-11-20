# Panel widths open at a share of their split

A docked panel's default width is a proportion of the split it lives in, not a fixed pixel token.

## Status

Accepted and implemented.

## Context

Every docked panel opened at a pixel default: the conversation list at `260px`, the Inspector and the
Artifact Canvas at `420px`, the last two read from `--panel-width` and `--artifact-canvas-width` in
`geometry.css`. Those numbers were chosen for a laptop-width window. On a 1800px window a 420px
Inspector is a phone column beside a 1120px answer, and the traces dock — which draws a Run's
timeline beside an agent page — could never use the width it needs; on a small window the fixed
sidebar crowds the conversation. The panel already clamped its **maximum** to the room the answer
keeps, but its **default** ignored the window it was opening into.

A width the reader drags is a different thing from a width a panel opens at: the first is a stated
preference and must be kept in pixels, the second should follow the space available. Mixing them is
what made the token feel rigid.

## Decision

A panel's default width is `clamp(min, round(fraction × axis), max)`, where `axis` is the rendered
width of the split the panel belongs to — not the viewport — and the shares are:

| Panel | fraction | min | max |
| --- | --- | --- | --- |
| Conversation list | 18% of its split | 240 | 360 |
| Inspector (Files / Sources / live) | 38% of the panel split | 320 | 560 |
| Inspector (Run traces) | 50% of the panel split | 420 | 960 |
| Artifact Canvas (side by side) | 42% of its split | 320 | 720 |

- The computed default is held once more to the room the split leaves: the chat's 520px reserve, a
  panel docked opposite, and the dividers between them. A share can therefore never take the
  answer's space.
- The shares live in one module, `frontend/lib/panel-share.ts`; the split adapter in
  `ui/split-panel.ts` owns only pixel state (drag, persistence, clamping) and asks that module what
  width a grain opens at. The stylesheet's `--panel-width` / `--artifact-canvas-width` are
  proportional stand-ins for the first paint, not the intended size.
- The Shell tells the splits the Inspector's read grain with the rest of its state
  (`syncPanelSplitState`); the splits do not read it back off `<body>` classes, which are for the
  stylesheets.
- Defaults apply only when the reader has not stated a width: a first open, or a change of grain
  while no width is stated, or a double-click on the sash. A drag and that double-click store pixels;
  a window resize only clamps what is drawn and never re-derives a share.
- A panel nobody has resized writes no width to storage; the stylesheet tokens stay as the fallback
  for a split with no rendered axis to measure, and for when scripting arranges the Shell.
- Compact drawers and the Canvas overlay/wide modes keep their existing behavior; this decision is
  about the docked, resizable column.

Because the shares are of each split's own axis, nesting composes: the conversation split's axis is
the window, the panel split's is the column beside it, and the Canvas split's is what the panel split
leaves. A panel widened by its share narrows the axis the next panel's share is measured against,
which is what keeps the composition from oversubscribing the window.

## Considered options

- **Keep the fixed tokens.** Rejected: a fixed default is what looks like a phone column on a wide
  window and crowds the answer on a small one.
- **A viewport percentage for every panel.** Rejected: panels nest, so a viewport share for the
  Inspector ignores the conversation list that already took part of the window; the shares would
  oversubscribe on a wide window and collide on a narrow one.
- **Pure CSS (`width: min(860px, 43vw)`).** Rejected for the dock: the split drags and persists in
  pixels, so a CSS-only default would be overridden by the Shell's rendered width on the second
  layout; the computed default has to be a number the split can own.
- **Always proportional, including after a drag.** Rejected: a drag is a stated preference; rescaling
  it on the next resize would silently undo the reader's choice.

## Consequences

- Wide windows get panels that scale with the work area; narrow ones keep the answer's reserve
  regardless of the share.
- The traces dock reaches its two-column layout only when the panel split's 50% passes 736px —
  roughly an 1800px window with the conversation list open (1800: sidebar 324, panel axis 1475,
  traces 738). Below that the dock falls back to its narrow layout rather than forcing a two-column
  squeeze.
- Tests that want a width must state the room (the split's axis), not a window size: the same share
  means different pixels at different axes.
- The stylesheet tokens are now fallbacks, not intended sizes; changing them no longer changes the
  size a reader sees on a normal desktop.

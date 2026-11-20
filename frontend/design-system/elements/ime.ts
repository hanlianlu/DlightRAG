// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Whether a key press belongs to an IME composition and not to the page.

 *  The key that confirms or cancels a composition (Enter, Escape, an arrow or Tab that picks a
 *  candidate) is the composition's. A handler that treats it as a command sends a half-typed
 *  message, cancels a Run, or closes a panel on a reader who only meant to finish a word.
 *
 *  Chromium and Firefox mark such a keydown `isComposing`. WebKit ends the composition first:
 *  the IME's keydown arrives after `compositionend`, where `isComposing` is already false, and
 *  WebKit reports `keyCode` 229 ("processed by an IME") instead. A handler checks both, here,
 *  once, and returns before it acts on a key for which this is true.
 */
export function isImeKey(event: KeyboardEvent): boolean {
  // `keyCode` is deprecated, but 229 is the only signal WebKit leaves on that late keydown.
  return event.isComposing || event.keyCode === 229;
}

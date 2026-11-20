// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Whole counts as a reader skims them: a token total is compacted to the reader's own unit —
 * 12344750 reads 12.3M — so a row stays short where the exact digits would run long. */

/** Format one whole count for display. Anything that is not a finite number reads as nothing. */
export function formatCompactCount(value: number, locale: string): string {
  if (!Number.isFinite(value)) return '';
  return new Intl.NumberFormat(locale, {notation: 'compact', maximumFractionDigits: 1}).format(value);
}

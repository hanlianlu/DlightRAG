// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Parsing and ghost preview for the `/skill:<name>` and `/name` web directives. */

export interface SkillDirective {
  readonly skill: string;
  readonly query: string;
}

export interface SkillDirectiveState {
  /** 'canonical' = `/skill:<name>` path (including partial 'skill' keyword); 'shorthand' = `/name`. */
  readonly kind: 'canonical' | 'shorthand';
  /** Name prefix typed so far (after the colon, or after the slash for shorthand). */
  readonly prefix: string;
  /** For canonical: the part of the 'skill' keyword already typed ('', 's', 'sk', ...). */
  readonly typedKeyword: string;
}

const CANONICAL_PARTIAL = /^\/(s(?:k(?:i(?:l(?:l)?)?)?)?)?(?::([a-z0-9-]*))?$/;
const SHORTHAND_PARTIAL = /^\/([a-z0-9-]*)$/;
const SKILL_KEYWORD_PREFIX = /^s(?:k(?:i(?:l)?)?)?$/;

export function parseSkillDirective(text: string): SkillDirective | null {
  const trimmed = text.trim();
  const canonical = /^\/skill:([a-z0-9-]+)\s*([\s\S]*)$/.exec(trimmed);
  if (canonical !== null) {
    return {skill: canonical[1]!, query: canonical[2]!.trim()};
  }
  const shorthand = /^\/([a-z0-9-]+)\s*([\s\S]*)$/.exec(trimmed);
  if (shorthand !== null && !trimmed.startsWith('/skill:')) {
    return {skill: shorthand[1]!, query: shorthand[2]!.trim()};
  }
  return null;
}

/**
 * The directive the draft is currently on, or null when the text is not on a
 * directive path. `/`, `/s`, `/skill`, `/skill:`, `/skill:r` are canonical;
 * `/td`, `/code-r` are shorthand; text with a trailing question is null.
 */
export function skillDirectiveState(text: string): SkillDirectiveState | null {
  const trimmed = text.trim();
  const canonical = CANONICAL_PARTIAL.exec(trimmed);
  if (canonical !== null) {
    return {kind: 'canonical', typedKeyword: canonical[1] ?? '', prefix: canonical[2] ?? ''};
  }
  const shorthand = SHORTHAND_PARTIAL.exec(trimmed);
  if (shorthand !== null && !SKILL_KEYWORD_PREFIX.test(shorthand[1]!)) {
    return {kind: 'shorthand', typedKeyword: '', prefix: shorthand[1]!};
  }
  return null;
}

/**
 * The ghost suffix to preview in the input for one skill: exactly the part of
 * the full directive Tab will install that the user has not typed yet. During
 * the canonical keyword phase the ghost teaches the full `/skill:` form.
 */
export function skillGhostSuffix(text: string, skillName: string): string {
  const state = skillDirectiveState(text);
  if (state === null) return '';
  const typed = text.trim();
  if (state.kind === 'canonical') {
    return `/skill:${skillName}`.slice(typed.length);
  }
  return skillName.slice(state.prefix.length);
}

/** The full directive form (with trailing space) Tab/Enter/click will install. */
export function committedSkillDirective(text: string, skillName: string): string {
  const state = skillDirectiveState(text);
  if (state !== null && state.kind === 'shorthand') {
    return `/${skillName} `;
  }
  return `/skill:${skillName} `;
}

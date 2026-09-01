// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Parsing for the `/skill:<name> [question]` web directive. */

export interface SkillDirective {
  readonly skill: string;
  readonly query: string;
}

const DIRECTIVE_PATTERN = /^\/skill:([a-z0-9-]+)\s*([\s\S]*)$/;
const DIRECTIVE_PREFIX_PATTERN = /^\/(?:s(?:k(?:i(?:l(?:l(?::([a-z0-9-]*))?)?)?)?)?)?$/;

export function parseSkillDirective(text: string): SkillDirective | null {
  const match = DIRECTIVE_PATTERN.exec(text.trim());
  if (match === null) return null;
  return {skill: match[1]!, query: match[2]!.trim()};
}

/**
 * The name prefix of a partially typed `/skill:` directive, or null when the
 * draft is not on the directive path. Bare `/` through `/skill` yield '';
 * `/skill:r` yields 'r'; anything with trailing text yields null.
 */
export function skillDirectivePrefix(text: string): string | null {
  const match = DIRECTIVE_PREFIX_PATTERN.exec(text.trim());
  return match === null ? null : (match[1] ?? '');
}

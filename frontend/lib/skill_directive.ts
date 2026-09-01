// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Parsing for the `/skill:<name> [question]` web directive. */

export interface SkillDirective {
  readonly skill: string;
  readonly query: string;
}

const DIRECTIVE_PATTERN = /^\/skill:([a-z0-9-]+)\s*([\s\S]*)$/;

export function parseSkillDirective(text: string): SkillDirective | null {
  const match = DIRECTIVE_PATTERN.exec(text.trim());
  if (match === null) return null;
  return {skill: match[1]!, query: match[2]!.trim()};
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for the discovered global Agent Skill catalog. */

export interface SkillSummary {
  readonly name: string;
  readonly description: string;
  readonly source: 'global' | 'workspace';
}

let catalogRequest: Promise<readonly SkillSummary[]> | null = null;

export function listSkills(): Promise<readonly SkillSummary[]> {
  if (catalogRequest === null) {
    catalogRequest = fetch('/web/api/skills')
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load Agent Skills (${response.status})`);
        }
        const body = (await response.json()) as {skills?: unknown};
        if (!Array.isArray(body.skills)) throw new Error('Malformed Agent Skills response');
        return body.skills.filter((item): item is SkillSummary =>
          item !== null && typeof item === 'object'
          && typeof (item as Record<string, unknown>).name === 'string'
          && typeof (item as Record<string, unknown>).description === 'string');
      })
      .catch((error: unknown) => {
        catalogRequest = null;
        throw error;
      });
  }
  return catalogRequest;
}

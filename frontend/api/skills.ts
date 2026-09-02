// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for the discovered global Agent Skill catalog. */

import * as v from 'valibot';

const skillSummary = v.object({
  name: v.string(),
  description: v.string(),
  source: v.picklist(['global', 'owner']),
});
export type SkillSummary = v.InferOutput<typeof skillSummary>;

let catalogRequest: Promise<readonly SkillSummary[]> | null = null;

export function listSkills(): Promise<readonly SkillSummary[]> {
  if (catalogRequest === null) {
    catalogRequest = fetch('/web/api/skills')
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load Agent Skills (${response.status})`);
        }
        const body = v.parse(v.object({skills: v.array(v.unknown())}), await response.json());
        // One malformed entry must not reject the whole catalog; skip it.
        return body.skills.filter((item): item is SkillSummary => v.is(skillSummary, item));
      })
      .catch((error: unknown) => {
        catalogRequest = null;
        throw error;
      });
  }
  return catalogRequest;
}

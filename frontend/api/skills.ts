// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for the merged Agent Skill catalog. */

import * as v from 'valibot';
import {parseWire} from './wire.ts';

const skillSummary = v.object({
  name: v.string(),
  description: v.string(),
  source: v.picklist(['builtin', 'global', 'owner']),
});
export type SkillSummary = v.InferOutput<typeof skillSummary>;

/** The catalog is the server's live view, so every call asks again. */
export async function listSkills(): Promise<readonly SkillSummary[]> {
  const response = await fetch('/web/api/skills');
  const body = await parseWire(response, v.object({skills: v.array(v.unknown())}));
  // One malformed entry must not reject the whole catalog; skip it.
  return body.skills.filter((item): item is SkillSummary => v.is(skillSummary, item));
}

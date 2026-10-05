// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for Agent Skills: the merged catalog the composer completes from, and the
 *  owner's own Skills that Settings lists, switches, reads, and deletes. */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {apiError, parseWire} from './wire.ts';

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

const ownerSkill = v.object({name: v.string(), description: v.string(), enabled: v.boolean()});
/** One of the owner's own Skills; a disabled one is kept but the agent does not load it. */
export type OwnerSkill = v.InferOutput<typeof ownerSkill>;

const ownerSkills = v.object({skills: v.array(ownerSkill), limit: v.number()});
/** The owner's own Skills by name, and how many the owner may keep. */
export type OwnerSkills = v.InferOutput<typeof ownerSkills>;

const OWNER = '/web/api/skills/mine';

export async function listOwnerSkills(signal?: AbortSignal): Promise<OwnerSkills> {
  return parseWire(await fetch(OWNER, {signal}), ownerSkills);
}

/** Turn one Skill on or off; the reply is the Skill as it now stands. */
export async function setOwnerSkillEnabled(
  name: string,
  enabled: boolean,
  signal?: AbortSignal,
): Promise<OwnerSkill> {
  return parseWire(await fetch(`${OWNER}/${encodeURIComponent(name)}/enabled`, {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({enabled}),
    signal,
  }), ownerSkill);
}

/** Delete one Skill for good; a 404 means it is already gone. */
export async function deleteOwnerSkill(name: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${OWNER}/${encodeURIComponent(name)}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

/** The text of a Skill's SKILL.md, exactly as stored. */
export async function getOwnerSkillDocument(name: string, signal?: AbortSignal): Promise<string> {
  const response = await fetch(`${OWNER}/${encodeURIComponent(name)}/document`, {signal});
  if (!response.ok) throw await apiError(response);
  return await response.text();
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The agent efforts the server accepts on a browser submission and names in its bootstrap.
 *
 * One list serves the bootstrap schema, the submission request, and the composer's own checks,
 * so a level can neither be offered that a submission would reject nor be accepted that the
 * composer cannot name. The composer's labels and the stored choice are `lib/agent-effort.ts`.
 */

import * as v from 'valibot';

export const AGENT_EFFORT_LEVELS = ['low', 'high', 'max'] as const;

export type AgentEffort = (typeof AGENT_EFFORT_LEVELS)[number];

/** One level, as the wire spells it. */
export const agentEffortLevel = v.picklist(AGENT_EFFORT_LEVELS);

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * Who an agent is.
 *
 * A Run has one agent of its own and any number of children, and the interfaces used to say so in
 * three spellings at once: the wire sends `null` for the Run's agent and `"main"` for the same
 * agent when a search found it, the roster names it `"main"`, and a page that shows nothing used
 * the empty string. Comparing those spellings is how the Run's own matches went missing from a
 * Chinese search. They are one value here, decided once, where the wire is read.
 */

/** The Run's own agent, or one of its children. */
export type AgentId = {kind: 'run'} | {kind: 'child'; id: string};

/** What names the Run's own agent wherever a child's id would go: the roster's own spelling. */
export const MAIN_AGENT = 'main';

export const RUN_AGENT: AgentId = {kind: 'run'};

/** The agent a wire value names. Anything that is not a child's id is the Run's own agent. */
export function agentIdOf(wire: string | null | undefined): AgentId {
  return wire === null || wire === undefined || wire === '' || wire === MAIN_AGENT
    ? RUN_AGENT
    : {kind: 'child', id: wire};
}

/** The id the roster and the wire use for this agent, so a lookup can be one map. */
export function agentKey(agent: AgentId): string {
  return agent.kind === 'run' ? MAIN_AGENT : agent.id;
}

/** The agent a roster row stands for: any row that names its child, whatever else it carries. */
export function agentOfRow(row: {readonly childSessionId?: string | null | undefined}): AgentId {
  return agentIdOf(row.childSessionId);
}

/**
 * A page of wire messages with their agents decided. Every parser that reads messages calls this,
 * so the spellings stop at the boundary and nothing below it compares `null` against `"main"`.
 */
export function readAgents<T extends {agent?: string | null | undefined}>(
  messages: readonly T[],
): (T & {agent: AgentId})[] {
  return messages.map((message) => ({...message, agent: agentIdOf(message.agent)}));
}

export function sameAgent(one: AgentId | null, other: AgentId | null): boolean {
  if (one === null || other === null) return one === other;
  return agentKey(one) === agentKey(other);
}

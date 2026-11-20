// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import type {AgentId} from './agent-id.ts';

/** What the product calls an agent that has no name of its own: the Run's own agent, or a child the
 *  roster gives neither an objective nor a label. Call it while rendering, so a language change is read. */
export function agentName(agent: AgentId): string {
  return agent.kind === 'run'
    ? msg('Main agent', {id: 'agentName.main'})
    : msg('Child agent', {id: 'agentName.child'});
}

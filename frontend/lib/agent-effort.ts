// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The agent effort a browser submission may choose.
 *
 * Levels arrive from the authenticated bootstrap, so the composer only ever
 * offers what this deployment accepts and never invents a level the server
 * would reject. The stored choice is the caller's own override; a submission
 * with no stored choice omits the field and the deployment default applies.
 */

import {readStored, writeStored} from './storage.ts';

export type AgentEffort = 'low' | 'high' | 'max';

export interface AgentEffortOffer {
    readonly levels: readonly AgentEffort[];
    readonly default: AgentEffort | null;
}

/** English source labels; ids are `chatComposer.effort.<level>`. */
export const EFFORT_LABELS: Record<AgentEffort, string> = {
    low: 'Low',
    high: 'High',
    max: 'Max',
};

export const AGENT_EFFORT_STORAGE_KEY = 'dlightrag.composer.effort';

const ALL_LEVELS: readonly AgentEffort[] = ['low', 'high', 'max'];

export const EMPTY_AGENT_EFFORT_OFFER: AgentEffortOffer = {levels: [], default: null};

export function isAgentEffort(value: unknown): value is AgentEffort {
    return ALL_LEVELS.includes(value as AgentEffort);
}

/** Keep only offered levels, in the deployment's own order. */
export function offeredLevels(offer: AgentEffortOffer): AgentEffort[] {
    return [...offer.levels].filter(isAgentEffort);
}

/**
 * The stored override, accepted only while this deployment still offers it.
 *
 * A level the deployment dropped, a value written by another surface, or a
 * blocked storage all read as "no override" so a submission stays valid.
 */
export function storedAgentEffort(offer: AgentEffortOffer): AgentEffort | null {
    const value = readStored(AGENT_EFFORT_STORAGE_KEY);
    return offeredLevels(offer).includes(value as AgentEffort) ? value as AgentEffort : null;
}

export function storeAgentEffort(effort: AgentEffort | null): void {
    writeStored(AGENT_EFFORT_STORAGE_KEY, effort);
}

/** The level the composer shows: the stored override, else the deployment default. */
export function displayedAgentEffort(offer: AgentEffortOffer): AgentEffort | null {
    return storedAgentEffort(offer) ?? offer.default;
}

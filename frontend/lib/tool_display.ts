// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Human-readable verbs for the fixed answer tools; unknown tools fall back to a
 *  prettified raw name so new tools render sensibly with zero maintenance. */

import {msg} from '@lit/localize';

const TOOL_VERBS: Record<string, string> = {
  search_knowledge_base: 'Searching the knowledge base',
  search_web: 'Searching the web',
  read: 'Reading a document',
  write: 'Writing a file',
  edit: 'Editing a file',
  grep: 'Searching files',
  bash: 'Running a command',
  find: 'Finding files',
  ls: 'Listing files',
  load_skill: 'Loading skill',
  publish_skill: 'Publishing skill',
  delete_skill: 'Removing skill',
  remember: 'Saving a memory',
  forget: 'Removing a memory',
  recall_memory: 'Recalling memories',
  spawn_agent: 'Dispatching a subagent',
};

export interface ToolDisplay {
  readonly verb: string;
  readonly verbId: string | null;
  readonly known: boolean;
}

export function toolDisplay(name: string): ToolDisplay {
  const verb = TOOL_VERBS[name];
  if (verb !== undefined) {
    return {verb, verbId: `chatFeature.tool.${name}`, known: true};
  }
  return {verb: prettyToolName(name), verbId: null, known: false};
}

/** The localized verb for one tool row; unknown tools render their prettified
 *  raw name directly, so new tools need no catalog entry. */
export function toolVerbText(verb: string, verbId: string | null): string {
  return verbId === null ? verb : msg(verb, {id: verbId});
}

export function prettyToolName(name: string): string {
  const spaced = name.replace(/[:_-]+/g, ' ').trim();
  if (!spaced) return name;
  return spaced.replace(/\b[a-z]/g, (char) => char.toUpperCase());
}

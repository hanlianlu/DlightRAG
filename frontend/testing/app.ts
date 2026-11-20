// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the tests that mount the whole application shell share: the bootstrap the server sends, and a fetch
 *  that answers it. */

import {DEFAULT_CHANGES} from './workspaces.ts';

/** The bootstrap payload as the wire carries it. */
export const bootstrap = {
  contract_version: 6,
  workspaces: [
    {workspace: 'default', display_name: 'Default', embedding_model: 'embed-test', changes: [...DEFAULT_CHANGES]},
  ],
  workspaces_next_cursor: null,
  primary_workspace: 'default',
  active_workspaces: ['default'],
  known_workspaces: ['default'],
  default_workspace: 'default',
  answer_attachments: {
    count_limit: 6,
    image_max_bytes: 1024,
    document_max_bytes: 2048,
    extensions: ['md', 'pdf'],
    image_capability: 'supported',
    image_limit: 3,
    accept: 'image/*,.md,.pdf',
  },
  active_html_preview_enabled: true,
  agent_effort: {levels: ['low', 'high', 'max'], default: 'high'},
  answer_feedback_enabled: true,
} as const;

/** The page's reads at start: the bootstrap, and a conversation list holding `conversations`. */
export function bootstrapResponse(input: RequestInfo | URL, conversations: readonly unknown[] = []): Response {
  return Response.json(String(input) === '/web/api/conversations'
    ? {items: conversations, next_cursor: null}
    : bootstrap);
}

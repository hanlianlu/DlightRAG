// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {workspacePageItem} from './workspaces.ts';

import {parseWire} from './wire.ts';
import type {AgentEffort} from '../lib/agent-effort.ts';

export type ImageCapabilityStatus = 'supported' | 'unsupported' | 'unknown';

const answerAttachmentBootstrap = v.pipe(
  v.object({
    count_limit: v.number(),
    image_max_bytes: v.number(),
    document_max_bytes: v.number(),
    extensions: v.array(v.string()),
    image_capability: v.picklist(['supported', 'unsupported', 'unknown']),
    image_limit: v.number(),
    accept: v.string(),
  }),
  v.transform((w) => ({
    countLimit: w.count_limit,
    imageMaxBytes: w.image_max_bytes,
    documentMaxBytes: w.document_max_bytes,
    extensions: w.extensions,
    imageCapability: w.image_capability,
    imageLimit: w.image_limit,
    accept: w.accept,
  })),
);
export type AnswerAttachmentBootstrap = v.InferOutput<typeof answerAttachmentBootstrap>;

const agentEffortBootstrap = v.pipe(
  v.object({
    levels: v.array(v.picklist(['low', 'high', 'max'])),
    default: v.optional(v.nullable(v.picklist(['low', 'high', 'max']))),
  }),
  v.transform((w) => ({
    levels: w.levels as readonly AgentEffort[],
    default: (w.default ?? null) as AgentEffort | null,
  })),
);
export type AgentEffortBootstrap = v.InferOutput<typeof agentEffortBootstrap>;

const webBootstrap = v.pipe(
  v.object({
    contract_version: v.literal(3),
    personal_mcp_connections: v.boolean(),
    workspaces: v.array(workspacePageItem),
    workspaces_next_cursor: v.optional(v.nullable(v.string())),
    primary_workspace: v.string(),
    active_workspaces: v.array(v.string()),
    known_workspaces: v.optional(v.nullable(v.array(v.string()))),
    answer_attachments: answerAttachmentBootstrap,
    active_html_preview_enabled: v.boolean(),
    agent_effort: agentEffortBootstrap,
  }),
  v.transform((w) => ({
    contractVersion: w.contract_version,
    personalMcpConnections: w.personal_mcp_connections,
    workspaces: w.workspaces,
    workspacesNextCursor: w.workspaces_next_cursor ?? null,
    primaryWorkspace: w.primary_workspace,
    activeWorkspaces: w.active_workspaces,
    knownWorkspaces: w.known_workspaces ?? null,
    answerAttachments: w.answer_attachments,
    activeHtmlPreviewEnabled: w.active_html_preview_enabled,
    agentEffort: w.agent_effort,
  })),
);
export type WebBootstrap = v.InferOutput<typeof webBootstrap>;

export class BootstrapApiError extends Error {
  readonly status: number;

  constructor(status: number) {
    super('Failed to load the Web application');
    this.name = 'BootstrapApiError';
    this.status = status;
  }
}

export async function getWebBootstrap(signal?: AbortSignal): Promise<WebBootstrap> {
  const response = await fetch('/web/api/bootstrap', {signal});
  return parseWire(
    response,
    webBootstrap,
    (status) => new BootstrapApiError(status),
    'Failed to load the Web application',
  );
}

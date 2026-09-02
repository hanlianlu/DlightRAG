// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {workspacePageItem} from './workspaces.ts';

import {parseWire} from './wire.ts';

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

const webBootstrap = v.pipe(
  v.object({
    contract_version: v.literal(1),
    workspaces: v.array(workspacePageItem),
    workspaces_next_cursor: v.optional(v.nullable(v.string())),
    primary_workspace: v.string(),
    active_workspaces: v.array(v.string()),
    known_workspaces: v.optional(v.nullable(v.array(v.string()))),
    answer_attachments: answerAttachmentBootstrap,
    active_html_preview_enabled: v.boolean(),
  }),
  v.transform((w) => ({
    contractVersion: w.contract_version,
    workspaces: w.workspaces,
    workspacesNextCursor: w.workspaces_next_cursor ?? null,
    primaryWorkspace: w.primary_workspace,
    activeWorkspaces: w.active_workspaces,
    knownWorkspaces: w.known_workspaces ?? null,
    answerAttachments: w.answer_attachments,
    activeHtmlPreviewEnabled: w.active_html_preview_enabled,
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

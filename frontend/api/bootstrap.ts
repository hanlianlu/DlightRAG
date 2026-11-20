// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {agentEffortLevel} from './agent-effort.ts';
import {workspacePageItem} from './workspaces.ts';
import {parseWire} from './wire.ts';

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

const agentEffortBootstrap = v.pipe(
  v.object({
    levels: v.array(agentEffortLevel),
    default: v.optional(v.nullable(agentEffortLevel)),
  }),
  v.transform((w) => ({levels: w.levels, default: w.default ?? null})),
);

const observabilityBootstrap = v.pipe(
  v.object({
    url: v.string(),
    apiKey: v.string(),
    service: v.string(),
    // A server that predates the knob sends none, and every session is recorded.
    replay_sample_rate: v.optional(v.number(), 1),
  }),
  v.transform((w) => ({
    url: w.url,
    apiKey: w.apiKey,
    service: w.service,
    replaySampleRate: w.replay_sample_rate,
  })),
);

const webBootstrap = v.pipe(
  v.object({
    contract_version: v.literal(6),
    workspaces: v.array(workspacePageItem),
    workspaces_next_cursor: v.optional(v.nullable(v.string())),
    primary_workspace: v.string(),
    active_workspaces: v.array(v.string()),
    known_workspaces: v.optional(v.nullable(v.array(v.string()))),
    default_workspace: v.optional(v.string()),
    answer_attachments: answerAttachmentBootstrap,
    active_html_preview_enabled: v.boolean(),
    agent_effort: agentEffortBootstrap,
    answer_feedback_enabled: v.boolean(),
    observability: v.optional(v.nullable(observabilityBootstrap)),
  }),
  v.transform((w) => ({
    contractVersion: w.contract_version,
    workspaces: w.workspaces,
    workspacesNextCursor: w.workspaces_next_cursor ?? null,
    primaryWorkspace: w.primary_workspace,
    activeWorkspaces: w.active_workspaces,
    knownWorkspaces: w.known_workspaces ?? null,
    defaultWorkspace: w.default_workspace ?? '',
    answerAttachments: w.answer_attachments,
    activeHtmlPreviewEnabled: w.active_html_preview_enabled,
    agentEffort: w.agent_effort,
    /** Whether the reader is offered the thumbs under an answer: the `answer_feedback` Runtime Policy as it
     *  resolves for them. A rating the server then refuses (403) turns it off for the rest of the page. */
    answerFeedbackEnabled: w.answer_feedback_enabled,
    /** Where the page sends RUM and session replay, present only where the operator configured both
     *  halves of that channel. Its absence is the off state, and the page starts no SDK. */
    observability: w.observability ?? null,
  })),
);
export type WebBootstrap = v.InferOutput<typeof webBootstrap>;

export async function getWebBootstrap(signal?: AbortSignal): Promise<WebBootstrap> {
  const response = await fetch('/web/api/bootstrap', {signal});
  return parseWire(response, webBootstrap);
}

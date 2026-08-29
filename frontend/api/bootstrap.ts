// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface BootstrapWorkspace {
  workspace: string;
  display_name: string;
  embedding_model: string;
}

export type ImageCapabilityStatus = 'supported' | 'unsupported' | 'unknown';

export interface AnswerAttachmentBootstrap {
  count_limit: number;
  image_max_bytes: number;
  document_max_bytes: number;
  extensions: string[];
  image_capability: ImageCapabilityStatus;
  image_limit: number;
  accept: string;
}

export interface WebBootstrap {
  contract_version: 1;
  workspaces: BootstrapWorkspace[];
  workspaces_next_cursor?: string | null;
  primary_workspace: string;
  active_workspaces: string[];
  known_workspaces?: string[] | null;
  answer_attachments: AnswerAttachmentBootstrap;
  active_html_preview_enabled: boolean;
}

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
  if (!response.ok) throw new BootstrapApiError(response.status);
  try {
    const data = await response.json() as WebBootstrap;
    return {
      ...data,
      workspaces_next_cursor: data.workspaces_next_cursor ?? null,
      known_workspaces: data.known_workspaces ?? null,
    };
  } catch {
    throw new BootstrapApiError(response.status);
  }
}

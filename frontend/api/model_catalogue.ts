// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Web API client for runtime model catalogue administration. */

import {csrfHeaders} from './csrf.ts';

export type ReasoningLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface ReasoningProfilePayload {
  format: string;
  levels: Record<ReasoningLevel, string | null>;
}

export interface ModelProfilePayload {
  context_window_tokens: number;
  max_input_tokens: number | null;
  max_output_tokens: number | null;
  supports_images: boolean;
  reasoning: ReasoningProfilePayload | null;
}

export interface ModelCatalogueEntry {
  provider: string;
  model: string;
  base_url: string | null;
  profile: ModelProfilePayload;
  source: 'builtin' | 'overlay' | 'fallback';
}

export interface ModelCatalogue {
  revision: string;
  models: ModelCatalogueEntry[];
}

export class ModelCatalogueRequestError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

async function readResponse(response: Response): Promise<ModelCatalogue> {
  if (!response.ok) {
    let detail = `Model catalogue request failed (${response.status})`;
    try {
      const body = await response.json() as {detail?: string};
      if (body.detail) detail = body.detail;
    } catch {
      // Preserve the bounded status message when the response is not JSON.
    }
    throw new ModelCatalogueRequestError(response.status, detail);
  }
  return await response.json() as ModelCatalogue;
}

export async function getModelCatalogue(signal?: AbortSignal): Promise<ModelCatalogue> {
  return await readResponse(await fetch('/web/api/models/catalogue', {signal}));
}

export async function putModelCatalogueEntry(
  entry: Omit<ModelCatalogueEntry, 'source'>,
  revision: string,
  signal?: AbortSignal,
): Promise<ModelCatalogue> {
  return await readResponse(await fetch('/web/api/models/catalogue', {
    method: 'PUT',
    headers: {...csrfHeaders('application/json'), 'If-Match': revision},
    body: JSON.stringify(entry),
    signal,
  }));
}

export async function removeModelCatalogueEntry(
  entry: Pick<ModelCatalogueEntry, 'provider' | 'model' | 'base_url'>,
  revision: string,
  signal?: AbortSignal,
): Promise<ModelCatalogue> {
  const params = new URLSearchParams({provider: entry.provider, model: entry.model});
  if (entry.base_url) params.set('base_url', entry.base_url);
  return await readResponse(await fetch(`/web/api/models/catalogue?${params}`, {
    method: 'DELETE',
    headers: {...csrfHeaders(), 'If-Match': revision},
    signal,
  }));
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Connections page's fixtures, shared by its behaviour tests and its layout tests. */

import '../ui/settings-connections.ts';
import {waitFor} from './dom.ts';

/** A draft as the projection really reports it: never probed, so `disabled` with no catalogue. */
export const draft = {
  connection_id: 'fixture',
  label: 'External',
  endpoint: 'https://fixture.example/mcp',
  enabled: false,
  activation_epoch: 1,
  generation: 1,
  authentication: 'none',
  endpoint_headers: [],
  credential_header: null,
  credential_prefix: null,
  status: 'disabled',
  authorization_status: null,
  // No catalogue has been published yet, so the card has neither a size nor a cost to state.
  tool_count: null,
  declaration_tokens: null,
};

/** A probed Connection with a published catalogue: 12 tools at an estimated 12.3K tokens each turn. */
export const published = {
  ...draft,
  enabled: true,
  status: 'ready',
  tool_count: 12,
  declaration_tokens: 12_300,
};

export type Feature = HTMLElementTagNameMap['dl-settings-connections'];
export type Wire = {url: string; method?: string; body: Record<string, unknown>};

/** A reply that answers every method with the same owner projection. */
export function replies(connections: unknown[], recorded?: Wire[], presets: unknown[] = []): typeof window.fetch {
  return async (url, init) => {
    if (init?.method) {
      recorded?.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
    }
    return Response.json({revision: '1', connections, presets});
  };
}

export function mount(): Feature {
  const feature = document.createElement('dl-settings-connections');
  document.body.append(feature);
  return feature;
}

/** The page shows its cards as soon as it has read them; a card body needs its own expansion. */
export async function loaded(feature: Feature): Promise<void> {
  await waitFor(() => feature.view !== null);
  await feature.updateComplete;
}

export async function openCard(feature: Feature, connectionId: string): Promise<void> {
  await loaded(feature);
  feature.querySelector<HTMLButtonElement>(`[data-card="${connectionId}"]`)!.click();
  await feature.updateComplete;
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlModelCatalogue} from './model_catalogue.ts';
import './model_catalogue.ts';

const originalFetch = window.fetch;
const revision = `sha256:${'1'.repeat(64)}`;
const nextRevision = `sha256:${'2'.repeat(64)}`;

function response(currentRevision: string, context = 100000): Response {
  return new Response(JSON.stringify({
    revision: currentRevision,
    models: [{
      provider: 'openai',
      model: 'test-model',
      base_url: null,
      profile: {
        context_window_tokens: context,
        max_input_tokens: null,
        max_output_tokens: 10000,
        supports_images: false,
        reasoning: null,
      },
      source: 'overlay',
    }],
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('loads and revision-guards one complete catalogue entry edit', async () => {
  const calls: Array<{method: string; headers: Headers; body: string}> = [];
  window.fetch = async (_input, init) => {
    const method = init?.method || 'GET';
    calls.push({
      method,
      headers: new Headers(init?.headers),
      body: String(init?.body || ''),
    });
    return method === 'PUT' ? response(nextRevision, 200000) : response(revision);
  };
  const editor = document.createElement('dl-model-catalogue') as DlModelCatalogue;
  document.body.append(editor);
  await editor.updateComplete;

  Array.from(editor.querySelectorAll('button'))
    .find((button) => button.textContent?.trim() === 'Load catalogue')?.click();
  await waitFor(() => editor.textContent?.includes('openai/test-model') ?? false);
  Array.from(editor.querySelectorAll('button'))
    .find((button) => button.textContent?.trim() === 'Edit')?.click();
  await editor.updateComplete;
  const textarea = editor.querySelector<HTMLTextAreaElement>('textarea')!;
  const payload = JSON.parse(textarea.value);
  payload.profile.context_window_tokens = 200000;
  textarea.value = JSON.stringify(payload);
  textarea.dispatchEvent(new Event('input'));
  Array.from(editor.querySelectorAll('button'))
    .find((button) => button.textContent?.trim() === 'Save model')?.click();
  await waitFor(() => editor.textContent?.includes(nextRevision) ?? false);

  expect(calls.map((call) => call.method)).to.deep.equal(['GET', 'PUT']);
  expect(calls[1].headers.get('If-Match')).to.equal(revision);
  expect(JSON.parse(calls[1].body).profile.context_window_tokens).to.equal(200000);
});

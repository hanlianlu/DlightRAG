// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {expect} from '@esm-bundle/chai';
import './settings-connections.ts';

const originalFetch = window.fetch;
const item = {connection_id: 'fixture', label: 'External', endpoint: 'https://fixture.example/mcp',
  enabled: false, activation_epoch: 1, generation: 1, authentication: 'none', status: 'ready', authorization_status: null,
  last_error_kind: null, catalogue_created_at: '2026-09-01T00:00:00Z',
  tools: [{remote_name: 'write', local_name: 'mcp_fixture_write', description: 'External write', input_schema: {type: 'object'}}]};
async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}
afterEach(() => {window.fetch = originalFetch; document.body.replaceChildren();});
it('requires whole-Connection warning before enable, with no tool checkbox', async () => {
  const mutations: Record<string, unknown>[] = [];
  window.fetch = async (_url, init) => {
    if (init?.method) mutations.push(JSON.parse(String(init.body)));
    return Response.json({revision: '1', single_user: true, connections: [{...item, enabled: mutations.length > 0}]});
  };
  const feature = document.createElement('dl-settings-connections');
  document.body.append(feature);
  await waitFor(() => Boolean(feature.querySelector('[data-enable]')));
  expect(feature.textContent).to.contain('single-user');
  feature.querySelector<HTMLButtonElement>('[data-enable]')!.click();
  await feature.updateComplete;
  expect(mutations).to.have.length(0);
  expect(feature.textContent).to.contain('modify, send, or delete');
  expect(feature.querySelector('input[type=checkbox]')).to.equal(null);
  feature.querySelector<HTMLButtonElement>('[data-consent]')!.click();
  await waitFor(() => mutations.length === 1);
  expect(mutations[0]).to.include({kind: 'enable', consent_version: 1});
});
it('begins explicit candidate OAuth and offers only the admitted provider redirect', async () => {
  const mutations: {url: string; body: Record<string, unknown>}[] = [];
  window.fetch = async (url, init) => {
    if (init?.method) {
      mutations.push({url: String(url), body: JSON.parse(String(init.body))});
      return Response.json({authorization_url: 'https://as.example/authorize?state=fixture'});
    }
    return Response.json({revision: 'current', single_user: false, connections: [{...item, authentication: 'bearer', enabled: true}]});
  };
  const feature = document.createElement('dl-settings-connections');
  document.body.append(feature);
  await waitFor(() => Boolean(feature.querySelector('[data-oauth]')));
  feature.querySelector<HTMLInputElement>('[data-endpoint]')!.value = 'https://candidate.example/mcp';
  feature.querySelector<HTMLButtonElement>('[data-oauth]')!.click();
  await waitFor(() => Boolean(feature.querySelector('[data-oauth-continue]')));
  expect(mutations).to.have.length(1);
  expect(mutations[0]!.url).to.equal('/web/api/connections/mcp/fixture/oauth');
  expect(mutations[0]!.body).to.deep.equal({expected_revision: 'current', endpoint: 'https://candidate.example/mcp'});
  const link = feature.querySelector<HTMLAnchorElement>('[data-oauth-continue]')!;
  expect(link.href).to.equal('https://as.example/authorize?state=fixture');
  expect(link.rel).to.contain('noreferrer');
  expect(feature.querySelector('input[type=checkbox]')).to.equal(null);
});

it('localizes refresh and needs-auth status, labels credentials, and restores focus after consent', async () => {
  const {setLanguagePreference} = await import('../i18n/locale.ts');
  try {
    await setLanguagePreference('zh');
    window.fetch = async () => Response.json({revision: 'current', single_user: false, connections: [{...item, status: 'needs-auth', authentication: 'oauth'}]});
    const feature = document.createElement('dl-settings-connections');
    document.body.append(feature);
    await waitFor(() => Boolean(feature.querySelector('[data-enable]')));
    expect(feature.textContent).to.contain('需要授权');
    expect(feature.textContent).to.contain('已同意的权限范围内自动刷新');
    expect(feature.textContent).not.to.contain('needs-auth');
    for (const input of feature.querySelectorAll<HTMLInputElement>('input')) expect(input.labels!.length).to.equal(1);
    const enable = feature.querySelector<HTMLButtonElement>('[data-enable]')!;
    enable.focus(); enable.click(); await feature.updateComplete;
    const consent = feature.querySelector<HTMLButtonElement>('[data-consent]')!;
    expect(consent.textContent).to.contain('同意并启用所有工具');
    consent.click();
    await waitFor(() => feature.consent === null);
    await feature.updateComplete;
    expect(document.activeElement).to.equal(feature.querySelector('[data-connections-root]'));
    expect(feature.querySelector('input[type=checkbox]')).to.equal(null);
  } finally {await setLanguagePreference('en');}
});

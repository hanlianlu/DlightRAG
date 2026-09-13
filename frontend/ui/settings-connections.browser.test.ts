// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {expect} from '@esm-bundle/chai';
import './settings-connections.ts';

const originalFetch = window.fetch;

/** A draft as the projection really reports it: never probed, so `disabled` with no catalogue. */
const draft = {
  connection_id: 'fixture',
  label: 'External',
  endpoint: 'https://fixture.example/mcp',
  enabled: false,
  activation_epoch: 1,
  generation: 1,
  authentication: 'none',
  status: 'disabled',
  authorization_status: null,
};

type Feature = HTMLElementTagNameMap['dl-settings-connections'];
type Wire = {url: string; method?: string; body: Record<string, unknown>};

/** A reply that answers every method with the same owner projection. */
function replies(connections: unknown[], recorded?: Wire[]): typeof window.fetch {
  return async (url, init) => {
    if (init?.method) {
      recorded?.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
    }
    return Response.json({revision: '1', connections});
  };
}

/** A successful probe publishes a working observation; a mock that never says so lies. */
function probeReplies(connections: () => Record<string, unknown>[], recorded?: Wire[]): typeof window.fetch {
  let probed = false;
  return async (url, init) => {
    if (init?.method) {
      recorded?.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
      if (String(url).endsWith('/probe')) probed = true;
    }
    return Response.json({
      revision: '1',
      connections: connections().map((connection) => (probed ? {...connection, status: 'ready'} : connection)),
    });
  };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 100; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error('condition did not become true');
}

function mount(): Feature {
  const feature = document.createElement('dl-settings-connections');
  document.body.append(feature);
  return feature;
}

/** The group is collapsed by default; a card body needs its own expansion on top of that. */
async function openGroup(feature: Feature): Promise<void> {
  await waitFor(() => feature.view !== null);
  feature.querySelector<HTMLButtonElement>('[data-connections-root]')!.click();
  await feature.updateComplete;
}

async function openCard(feature: Feature, connectionId: string): Promise<void> {
  await openGroup(feature);
  feature.querySelector<HTMLButtonElement>(`[data-card="${connectionId}"]`)!.click();
  await feature.updateComplete;
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('stays collapsed, reports the inventory, and never projects the catalogue', async () => {
  window.fetch = replies([draft, {...draft, connection_id: 'second', enabled: true, status: 'ready'}]);
  const feature = mount();
  await waitFor(() => feature.textContent.includes('1 of 2 enabled'));
  expect(feature.textContent).not.to.contain('tool');
  expect(feature.querySelector('[data-switch]')).to.equal(null);
  await openGroup(feature);
  expect(feature.querySelectorAll('[data-switch]')).to.have.length(2);
});

it('lets the switch own discovery, and asks the owner once before the first enable', async () => {
  const commands: Wire[] = [];
  window.fetch = probeReplies(() => [draft], commands);
  const feature = mount();
  await openGroup(feature);

  feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!.click();
  const consent = feature.querySelector<HTMLDialogElement>('#connections-consent')!;
  await waitFor(() => consent.open);
  // Nothing reaches the server before the owner answers the standing-authorization gate.
  expect(commands).to.deep.equal([]);
  consent.querySelector<HTMLButtonElement>('button[value=enable]')!.click();

  await waitFor(() => commands.length === 2);
  // Discovery is the probe route, not a body field; enabling carries the recorded consent.
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/probe');
  expect(commands[0]!.method).to.equal('POST');
  expect(commands[1]!.method).to.equal('PATCH');
  expect(commands[1]!.body).to.deep.equal({
    expected_revision: '1',
    kind: 'enable',
    consent_version: 1,
  });
});

it('checks a Connection whose last observation failed, and never enables a dead one', async () => {
  const commands: Wire[] = [];
  const unconfirmed = {...draft, authentication: 'oauth', status: 'needs-auth'};
  window.fetch = replies([unconfirmed], commands);
  const feature = mount();
  await openGroup(feature);
  feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!.click();
  const consent = feature.querySelector<HTMLDialogElement>('#connections-consent')!;
  await waitFor(() => consent.open);
  consent.querySelector<HTMLButtonElement>('button[value=enable]')!.click();

  await waitFor(() => commands.length === 1);
  await feature.updateComplete;
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/probe');
  expect(feature.querySelector('[data-switch="fixture"]')!.getAttribute('aria-checked')).to.equal('false');
});

it('does not ask an owner who already runs an enabled Connection', async () => {
  const commands: Wire[] = [];
  const running = {...draft, connection_id: 'running', enabled: true, status: 'ready'};
  window.fetch = probeReplies(() => [running, draft], commands);
  const feature = mount();
  await openGroup(feature);
  feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!.click();

  await waitFor(() => commands.length === 2);
  const consent = feature.querySelector<HTMLDialogElement>('#connections-consent')!;
  expect(consent.open).to.equal(false);
  expect(commands.map((command) => command.url)).to.deep.equal([
    '/web/api/connections/mcp/fixture/probe',
    '/web/api/connections/mcp/fixture',
  ]);
});

it('distinguishes revoked and refreshing from a plain disabled Connection', async () => {
  window.fetch = replies([
    {...draft, connection_id: 'revoked', status: 'revoked', authentication: 'oauth'},
    {...draft, connection_id: 'busy', enabled: true, status: 'refreshing'},
  ]);
  const feature = mount();
  await openGroup(feature);
  const metas = [...feature.querySelectorAll('[class*=rowMeta]')].map((node) => node.textContent!.trim());
  expect(metas).to.deep.equal(['OAuth · Revoked', 'None · Refreshing']);
});

it('asks before deleting, names the Connection, and offers no revoke control', async () => {
  const commands: Wire[] = [];
  window.fetch = replies([{...draft, enabled: true, status: 'ready', authentication: 'bearer'}], commands);
  const feature = mount();
  await openCard(feature, 'fixture');

  // Deleting already destroys the stored credential, so a second destroy action must not exist.
  expect(feature.textContent).to.not.contain('Revoke credentials');
  expect(feature.querySelector('[data-revoke]')).to.equal(null);

  feature.querySelector<HTMLButtonElement>('[data-delete="fixture"]')!.click();
  const dialog = feature.querySelector<HTMLDialogElement>('#connections-delete')!;
  await waitFor(() => dialog.open);
  expect(dialog.textContent).to.contain('Delete External?');
  expect(dialog.textContent).to.contain('https://fixture.example/mcp');
  expect(dialog.textContent).to.contain('switch it off instead');
  dialog.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
  await waitFor(() => !dialog.open);
  expect(commands).to.deep.equal([]);
});

it('begins authorization against the stored endpoint and labels every credential field', async () => {
  const commands: Wire[] = [];
  const granted = {...draft, authentication: 'oauth', enabled: true, status: 'ready'};
  window.fetch = async (url, init) => {
    if (init?.method) {
      commands.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
      return Response.json({authorization_url: 'https://as.example/authorize?state=fixture'});
    }
    return Response.json({revision: 'current', connections: [granted]});
  };
  const feature = mount();
  await openCard(feature, 'fixture');
  for (const input of feature.querySelectorAll<HTMLInputElement>('input')) {
    expect(input.labels?.length).to.equal(1);
  }
  feature.querySelector<HTMLButtonElement>('[data-oauth="fixture"]')!.click();
  await waitFor(() => feature.querySelector('[data-oauth-continue]') !== null);
  expect(commands).to.have.length(1);
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/oauth');
  expect(commands[0]!.body).to.deep.equal({
    expected_revision: 'current',
    endpoint: 'https://fixture.example/mcp',
  });
  const link = feature.querySelector<HTMLAnchorElement>('[data-oauth-continue]')!;
  expect(link.href).to.equal('https://as.example/authorize?state=fixture');
  expect(link.rel).to.contain('noreferrer');
});

it('clears a typed credential when the drawer tears the Feature down', async () => {
  window.fetch = replies([{...draft, authentication: 'bearer', enabled: true, status: 'ready'}]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const input = feature.querySelector<HTMLInputElement>('[data-bearer="fixture"]')!;
  input.value = 'lin_api_secret';
  feature.remove();
  expect(input.value).to.equal('');
});

it('localizes the group summary and the gate copy, and restores focus on cancel', async () => {
  const {setLanguagePreference} = await import('../i18n/locale.ts');
  try {
    await setLanguagePreference('zh');
    window.fetch = replies([{...draft, authentication: 'oauth'}]);
    const feature = mount();
    await openGroup(feature);
    expect(feature.textContent).to.contain('0 / 1 已启用');
    expect(feature.textContent).not.to.contain('needs-auth');
    const toggle = feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!;
    toggle.focus();
    toggle.click();
    const consent = feature.querySelector<HTMLDialogElement>('#connections-consent')!;
    await waitFor(() => consent.open);
    expect(consent.textContent).to.contain('我已了解');
    consent.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
    await waitFor(() => !consent.open);
    expect(document.activeElement).to.equal(toggle);
  } finally {
    await setLanguagePreference('en');
  }
});

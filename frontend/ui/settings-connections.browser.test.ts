// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {expect} from '@esm-bundle/chai';
import {setViewport} from '@web/test-runner-commands';
import './settings-connections.ts';
import {buttonNamed, fieldNamed, linkNamed, linkStyles, waitFor} from '../testing/dom.ts';
import type {SettingsSummary} from './settings-summary.ts';

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
function replies(connections: unknown[], recorded?: Wire[], presets: unknown[] = []): typeof window.fetch {
  return async (url, init) => {
    if (init?.method) {
      recorded?.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
    }
    return Response.json({revision: '1', connections, presets});
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
      presets: [],
    });
  };
}

function mount(): Feature {
  const feature = document.createElement('dl-settings-connections');
  document.body.append(feature);
  return feature;
}

/** The page shows its cards as soon as it has read them; a card body needs its own expansion. */
async function loaded(feature: Feature): Promise<void> {
  await waitFor(() => feature.view !== null);
  await feature.updateComplete;
}

async function openCard(feature: Feature, connectionId: string): Promise<void> {
  await loaded(feature);
  feature.querySelector<HTMLButtonElement>(`[data-card="${connectionId}"]`)!.click();
  await feature.updateComplete;
}

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
});

it('shows every card at once, reports the inventory, and never projects the catalogue', async () => {
  window.fetch = replies([draft, {...draft, connection_id: 'second', enabled: true, status: 'ready'}]);
  const feature = mount();
  await waitFor(() => feature.textContent.includes('1 of 2 enabled'));
  expect(feature.textContent).not.to.contain('tool');
  // There is no group to open first: the page is the open state.
  expect(feature.querySelectorAll('[data-switch]')).to.have.length(2);
  expect(feature.querySelector('[aria-expanded]')?.getAttribute('aria-expanded')).to.equal('false');
  expect(feature.textContent).to.contain('MCP');
});

it('tells the dialog how many Connections are on, each time it reads them', async () => {
  window.fetch = replies([draft, {...draft, connection_id: 'second', enabled: true, status: 'ready'}]);
  const heard: SettingsSummary[] = [];
  document.body.addEventListener('dl-settings-summary', (event) => { heard.push(event.detail); });
  mount();
  await waitFor(() => heard.length > 0);

  expect(heard[0]).to.deep.equal({section: 'connections', enabled: 1, total: 2});
});

it('fills the form from a preset and opens the new Connection on the tab its tier implies', async () => {
  const commands: Wire[] = [];
  const presets = [
    {preset_id: 'notion', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp', default_authentication: 'oauth'},
    {preset_id: 'wolfram', label: 'Wolfram', endpoint: 'https://agenttools.wolfram.com/mcp', default_authentication: 'none'},
  ];
  const created = {
    ...draft,
    connection_id: 'new',
    label: 'Notion',
    endpoint: 'https://mcp.notion.com/mcp',
  };
  window.fetch = async (url, init) => {
    if (!init?.method) return Response.json({revision: '1', connections: [], presets});
    commands.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
    return Response.json({revision: '2', connections: [created], presets});
  };
  const feature = mount();
  await loaded(feature);
  const addRow = [...feature.querySelectorAll<HTMLButtonElement>('button')]
    .find((button) => button.textContent?.includes('Add MCP connection'))!;
  addRow.click();
  await feature.updateComplete;

  const chips = [...feature.querySelectorAll<HTMLButtonElement>('[role="group"][aria-label="Presets"] button')];
  expect(chips.map((chip) => chip.textContent)).to.deep.equal(['Notion', 'Wolfram']);
  expect(chips[0]!.getAttribute('aria-label')).to.equal('Use the Notion preset');
  chips[0]!.click();
  await feature.updateComplete;
  expect(feature.querySelector<HTMLInputElement>('[data-new-label]')!.value).to.equal('Notion');
  expect(feature.querySelector<HTMLInputElement>('[data-new-endpoint]')!.value)
    .to.equal('https://mcp.notion.com/mcp');

  const submit = [...feature.querySelectorAll<HTMLButtonElement>('button')]
    .find((button) => button.textContent?.includes('Add connection'))!;
  submit.click();
  await waitFor(() => commands.length === 1);
  // A preset fills the form only: create stays the one command that owns the Connection.
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp');
  expect(commands[0]!.body).to.deep.equal({
    expected_revision: '1', label: 'Notion', endpoint: 'https://mcp.notion.com/mcp',
  });
  await waitFor(() => Boolean(feature.querySelector('[data-card="new"]')));
  // The card opens on the tab the preset implies, so the owner does not have to guess.
  const card = feature.querySelector('[data-card="new"]')!.closest('article')!;
  const selected = [...card.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]')];
  expect(selected.map((button) => button.textContent)).to.deep.equal(['OAuth']);
});

it('lets the switch own discovery, and asks the owner once before the first enable', async () => {
  const commands: Wire[] = [];
  window.fetch = probeReplies(() => [draft], commands);
  const feature = mount();
  await loaded(feature);

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
  await loaded(feature);
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
  await loaded(feature);
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
  await loaded(feature);
  const metas = [...feature.querySelectorAll('[class*=rowMeta]')].map((node) => node.textContent!.trim());
  expect(metas).to.deep.equal(['OAuth · Revoked', 'None · Refreshing']);
});

it('tells a changed Connection from a failed authorization, and both ask to authorize again', async () => {
  const notes: Record<string, string> = {};
  for (const outcome of ['failed', 'changed'] as const) {
    window.fetch = replies([{...draft, authentication: 'oauth', authorization_status: outcome}]);
    const feature = mount();
    await openCard(feature, 'fixture');
    const note = feature.querySelector('[role=alert]')!;
    notes[outcome] = note.textContent!.trim();
    feature.remove();
  }
  expect(notes.failed).to.equal('Authorization failed or expired. Authorize again to use this server.');
  expect(notes.changed).to.equal(
    'This connection changed while you were authorizing, so nothing was saved. Authorize again.',
  );
});

it('asks before deleting, names the Connection, and offers no revoke control', async () => {
  const commands: Wire[] = [];
  window.fetch = replies([{...draft, enabled: true, status: 'ready', authentication: 'bearer'}], commands);
  const feature = mount();
  await openCard(feature, 'fixture');

  // Deleting already destroys the stored credential, so a second destroy action must not exist.
  expect(feature.textContent).to.not.contain('Revoke credentials');
  expect(feature.querySelector('[data-revoke]')).to.equal(null);

  const trigger = feature.querySelector<HTMLButtonElement>('[data-delete="fixture"]')!;
  expect(trigger.textContent!.trim()).to.equal('Delete…');
  expect(trigger.parentElement!.textContent).to.contain(
    'Deleting removes the endpoint, the label and the stored credential; to pause it, switch it off instead.',
  );
  trigger.click();
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
    return Response.json({revision: 'current', connections: [granted], presets: []});
  };
  const feature = mount();
  await openCard(feature, 'fixture');
  for (const input of feature.querySelectorAll<HTMLInputElement>('input')) {
    expect(input.labels?.length).to.equal(1);
  }
  buttonNamed(feature, 'Authorize with OAuth')!.click();
  await waitFor(() => linkNamed(feature, 'Continue to provider authorization') !== null);
  expect(commands).to.have.length(1);
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/oauth');
  expect(commands[0]!.body).to.deep.equal({
    expected_revision: 'current',
    endpoint: 'https://fixture.example/mcp',
  });
  const link = linkNamed(feature, 'Continue to provider authorization')!;
  expect(link.href).to.equal('https://as.example/authorize?state=fixture');
  expect(link.rel).to.contain('noreferrer');
});

it('clears a typed credential when the dialog tears the Feature down', async () => {
  window.fetch = replies([{...draft, authentication: 'bearer', enabled: true, status: 'ready'}]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const input = feature.querySelector<HTMLInputElement>('[data-bearer="fixture"]')!;
  input.value = 'lin_api_secret';
  feature.remove();
  expect(input.value).to.equal('');
});

it('localizes the summary and the gate copy, and restores focus on cancel', async () => {
  const {setLanguagePreference} = await import('../i18n/locale.ts');
  try {
    await setLanguagePreference('zh');
    window.fetch = replies([{...draft, authentication: 'oauth'}]);
    const feature = mount();
    await loaded(feature);
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

describe('laid out as a page', () => {
  const original = {width: window.innerWidth, height: window.innerHeight};
  let unlink: () => void;
  before(async () => {
    unlink = await linkStyles([
      '../design-system/index.css',
      '../styles/layout.css',
      '../styles/settings-page.module.css',
      '../styles/settings-connections.module.css',
    ].map((href) => new URL(href, import.meta.url).href));
  });
  after(async () => {
    unlink();
    await setViewport(original);
  });

  it('puts Label and Endpoint side by side in a wide pane, and the delete hint beside Delete…', async () => {
    await setViewport({width: 1280, height: 800});
    window.fetch = replies([draft]);
    const feature = mount();
    await openCard(feature, 'fixture');

    const label = fieldNamed(feature, 'Label')!.getBoundingClientRect();
    const endpoint = feature.querySelector('[class*=endpointRow]')!.getBoundingClientRect();
    expect(label.right).to.be.at.most(endpoint.left);
    expect(label.top).to.be.closeTo(endpoint.top, 12);

    const deleteButton = feature.querySelector('[data-delete="fixture"]')!;
    const button = deleteButton.getBoundingClientRect();
    const hint = deleteButton.previousElementSibling!.getBoundingClientRect();
    expect(hint.right).to.be.at.most(button.left);
    expect(Math.abs(hint.top + hint.height / 2 - (button.top + button.height / 2))).to.be.below(button.height);
  });

  it('stacks them in one column on a phone', async () => {
    await setViewport({width: 390, height: 844});
    window.fetch = replies([draft]);
    const feature = mount();
    await openCard(feature, 'fixture');

    const label = fieldNamed(feature, 'Label')!.getBoundingClientRect();
    const endpoint = feature.querySelector('[class*=endpointRow]')!.getBoundingClientRect();
    expect(endpoint.top).to.be.at.least(label.bottom);
  });
});

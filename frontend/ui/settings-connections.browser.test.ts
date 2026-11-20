// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {expect} from '@esm-bundle/chai';
import './settings-connections.ts';
import {draft, type Feature, loaded, mount, openCard, published, replies, type Wire} from '../testing/connections.ts';
import {buttonNamed, linkNamed, waitFor} from '../testing/dom.ts';
import {memorySettings, mountSettings, openSettings, settingsClosed, wire} from '../testing/settings.ts';
import type {SettingsSummary} from './settings-summary.ts';

const originalFetch = window.fetch;
const originalSetTimeout = window.setTimeout;
const originalClearTimeout = window.clearTimeout;

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

/** A preset as the wire sends it: copy, a tab, and the non-secret facts that tab needs. */
function presetView(
  presetId: string,
  label: string,
  endpoint: string,
  authentication: string,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    preset_id: presetId, label, endpoint, default_authentication: authentication,
    endpoint_headers: [], credential_header: 'Authorization', credential_prefix: 'Bearer ', ...extra,
  };
}

/** Every delay the page asks its next read for, on a timer that runs that read at once. */
function capturedPollDelays(fires: number): number[] {
  const delays: number[] = [];
  let fired = 0;
  window.setTimeout = ((handler: TimerHandler, delay = 0) => {
    delays.push(delay);
    return fired++ < fires ? originalSetTimeout(handler, 0) : 0;
  }) as unknown as typeof window.setTimeout;
  window.clearTimeout = (() => {}) as typeof window.clearTimeout;
  return delays;
}

afterEach(() => {
  window.fetch = originalFetch;
  window.setTimeout = originalSetTimeout;
  window.clearTimeout = originalClearTimeout;
  document.body.replaceChildren();
});

it('shows every card at once, reports the inventory, and never projects the catalogue', async () => {
  window.fetch = replies([draft, {...draft, connection_id: 'second', enabled: true, status: 'ready'}]);
  const feature = mount();
  await waitFor(() => feature.textContent.includes('1 of 2 enabled'));
  // Neither fixture has a published catalogue: no card states a cost, and no catalogue reaches
  // the page -- only the inventory and, when published, what the declarations would add.
  expect(feature.textContent).not.to.contain('≈');
  // There is no group to open first: the page is the open state.
  expect(feature.querySelectorAll('[data-switch]')).to.have.length(2);
  expect(feature.querySelector('[aria-expanded]')?.getAttribute('aria-expanded')).to.equal('false');
  expect(feature.textContent).to.contain('MCP');
});

it('states what a published catalogue adds to every Research request', async () => {
  const oneTool = {...published, connection_id: 'single', label: 'Single', enabled: false, status: 'disabled',
    tool_count: 1, declaration_tokens: 2_000};
  window.fetch = replies([published, oneTool]);
  const feature = mount();
  await openCard(feature, 'fixture');

  expect(feature.querySelector('[data-card="fixture"]')!.closest('article')!.textContent)
    .to.contain('≈ 12.3K tokens per request · 12 tools');
  // A disabled Connection's cost is hypothetical, and the card still states it.
  await openCard(feature, 'single');
  expect(feature.querySelector('[data-card="single"]')!.closest('article')!.textContent)
    .to.contain('≈ 2K tokens per request · 1 tool');
});

it('states the token cost alone when the projection has no tool count', async () => {
  const tokensOnly = {...draft, enabled: true, status: 'ready', tool_count: null, declaration_tokens: 12_300};
  window.fetch = replies([tokensOnly]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;

  expect(card.textContent).to.contain('≈ 12.3K tokens per request');
  expect(card.textContent).not.to.contain('tools');
});

it('says nothing about declaration cost until a catalogue is published', async () => {
  window.fetch = replies([draft]);
  const feature = mount();
  await openCard(feature, 'fixture');

  expect(feature.textContent).not.to.contain('≈');
});

it('tells the dialog how many Connections are on, each time it reads them', async () => {
  window.fetch = replies([draft, {...draft, connection_id: 'second', enabled: true, status: 'ready'}]);
  const heard: SettingsSummary[] = [];
  document.body.addEventListener('dl-settings-summary', (event) => { heard.push(event.detail); });
  mount();
  await waitFor(() => heard.length > 0);

  expect(heard[0]).to.deep.equal({section: 'connections', enabled: 1, total: 2});
});

it('reads again in a second while an authorization is in flight, within a bounded window', async () => {
  // Thirty quick looks is the product's promise: an authorization settles seconds after the
  // browser comes back, and one abandoned at the provider must not hold the page at that pace.
  const QUICK_LOOKS = 30;
  const delays = capturedPollDelays(QUICK_LOOKS + 4);
  const pending = {...draft, authentication: 'oauth', authorization_status: 'pending'};
  const settled = {...draft, authentication: 'oauth', authorization_status: null};
  let reads = 0;
  window.fetch = async () => {
    reads += 1;
    // The flow is still in flight for the whole quick window, settles, and a later one starts.
    const connection = reads <= QUICK_LOOKS + 1 || reads >= QUICK_LOOKS + 3 ? pending : settled;
    return Response.json({revision: String(reads), connections: [connection], presets: []});
  };
  mount();
  await waitFor(() => delays.length > QUICK_LOOKS + 2);

  expect(delays.slice(0, QUICK_LOOKS)).to.deep.equal(Array(QUICK_LOOKS).fill(1000));
  // The quick looks are spent and the authorization is still pending: the page settles back.
  expect(delays[QUICK_LOOKS]).to.equal(5000);
  // A read that finds nothing in flight is steady…
  expect(delays[QUICK_LOOKS + 1]).to.equal(5000);
  // …and a later authorization spends the quick looks again.
  expect(delays[QUICK_LOOKS + 2]).to.equal(1000);
});

it('fills the form from a preset and opens the new Connection on the tab its tier implies', async () => {
  const commands: Wire[] = [];
  const presets = [
    {...presetView('notion', 'Notion', 'https://mcp.notion.com/mcp', 'oauth')},
    {...presetView('wolfram', 'Wolfram', 'https://agenttools.wolfram.com/mcp', 'none')},
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
    endpoint_headers: [],
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

it('says the owner\'s word again when the revision moved under it', async () => {
  // The background refresh, and the authorization the reader just finished, both move the
  // revision: refusing the switch because the page had not read that yet answers "reload and
  // retry" at somebody who only flipped a switch.
  const commands: Wire[] = [];
  const current = {...draft, authentication: 'oauth', authorization_status: 'succeeded'};
  let moved = false;
  window.fetch = async (url, init) => {
    if (init?.method) {
      commands.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
      // The first probe meets a revision this page has not read yet.
      if (!moved && String(url).endsWith('/probe')) {
        moved = true;
        return Response.json({kind: 'rejected', message: 'Connections revision changed'}, {status: 409});
      }
      return Response.json({revision: '2', connections: [{...current, enabled: true, status: 'ready'}], presets: []});
    }
    return Response.json({revision: '2', connections: [current], presets: []});
  };
  const feature = mount();
  await loaded(feature);

  feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!.click();
  feature.querySelector<HTMLDialogElement>('#connections-consent')!
    .querySelector<HTMLButtonElement>('button[value=enable]')!.click();

  await waitFor(() => feature.view?.connections[0]?.enabled === true);
  expect(feature.textContent).not.to.contain('revision changed');
  // The reader's word, said again against the revision the server holds now.
  expect(commands.map((command) => command.url)).to.deep.equal([
    '/web/api/connections/mcp/fixture/probe',
    '/web/api/connections/mcp/fixture/probe',
    '/web/api/connections/mcp/fixture',
  ]);
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

it('begins authorization against the stored endpoint and sends the browser to the provider', async () => {
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
  const visited: string[] = [];
  // The document itself is what leaves in the product; a test keeps its own page.
  feature.navigateTo = (url) => { visited.push(url); };
  await openCard(feature, 'fixture');
  for (const input of feature.querySelectorAll<HTMLInputElement>('input')) {
    expect(input.labels?.length).to.equal(1);
  }
  buttonNamed(feature, 'Authorize with OAuth')!.click();
  await waitFor(() => visited.length === 1);

  // One click is the whole gesture: the consent page is where the reader is headed.
  expect(visited).to.deep.equal(['https://as.example/authorize?state=fixture']);
  expect(commands).to.have.length(1);
  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/oauth');
  expect(commands[0]!.body).to.deep.equal({
    expected_revision: 'current',
    endpoint: 'https://fixture.example/mcp',
  });
  // One click is the whole gesture: nothing else on the card asks to be clicked, and a reader
  // the browser brings back sets off again from this same control.
  expect(linkNamed(feature, 'Continue to provider authorization')).to.equal(null);
  expect(buttonNamed(feature, 'Authorize with OAuth')).not.to.equal(null);
  expect(feature.textContent).not.to.contain('Continue to provider');
});

it('shows a Connection whose authorization is in flight as the OAuth one it is', async () => {
  // The return from a provider is a fresh page: the tab the reader had chosen is gone and no
  // credential is stored yet, so the card would read "None" at the reader who just authorized it.
  // A flow in flight is the server's own word about what this Connection is.
  window.fetch = replies([{...draft, authentication: 'none', authorization_status: 'pending'}]);
  const feature = mount();
  await loaded(feature);

  // The row is what the reader comes back to; the card is closed until they open it.
  expect(feature.querySelector('[data-card="fixture"]')!.parentElement!.textContent)
    .to.contain('OAuth · Authorization in progress');

  await openCard(feature, 'fixture');
  expect(feature.textContent).to.contain('Authorization pending.');
  expect(buttonNamed(feature, 'Authorize with OAuth')).not.to.equal(null);
  expect(feature.textContent).not.to.contain('No credential');
  const selected = [...feature.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]')];
  expect(selected.map((button) => button.textContent)).to.deep.equal(['OAuth']);
});

it('says a refused authorization on the row, with the light that belongs to a fault', async () => {
  // A refusal is reported inside the card, which is where nobody is looking when the page has
  // just reloaded: the row has to carry it too, or the reader sees a credential-less "None".
  window.fetch = replies([{...draft, authentication: 'none', authorization_status: 'failed'}]);
  const feature = mount();
  await loaded(feature);

  const row = feature.querySelector('[data-card="fixture"]')!.parentElement!;
  expect(row.textContent).to.contain('OAuth · Authorization failed');
  expect(row.querySelector('[class*="dotProblem"]')).not.to.equal(null);

  await openCard(feature, 'fixture');
  expect(feature.textContent).to.contain('Authorization failed or expired. Authorize again');
});

it('clears a typed credential when the dialog tears the Feature down', async () => {
  window.fetch = replies([{...draft, authentication: 'bearer', enabled: true, status: 'ready'}]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const input = feature.querySelector<HTMLInputElement>('[data-credential-token="fixture"]')!;
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

describe('while Settings stays open', () => {
  const enabled = {...draft, enabled: true, status: 'ready'};
  const disabled = {...draft, enabled: false, status: 'disabled'};
  let now = 0;
  let timers: Map<number, {due: number; handler: () => void}>;

  /** The page's timers run on a clock that only `pass` turns, and a timer that is cleared never fires. */
  function startClock(): void {
    now = 0;
    timers = new Map();
    let ids = 0;
    window.setTimeout = ((handler: TimerHandler, delay = 0) => {
      ids += 1;
      timers.set(ids, {due: now + delay, handler: handler as () => void});
      return ids;
    }) as typeof window.setTimeout;
    window.clearTimeout = ((id?: number) => { timers.delete(id!); }) as typeof window.clearTimeout;
  }

  /** Let the answers to what the page asked for land. */
  async function quiet(): Promise<void> {
    for (let turn = 0; turn < 5; turn += 1) await new Promise((resolve) => originalSetTimeout(resolve, 0));
  }

  /** Let `ms` go by: each timer that falls due fires in its turn, and its read answers before the next. */
  async function pass(ms: number): Promise<void> {
    const end = now + ms;
    for (;;) {
      const next = [...timers].filter(([, timer]) => timer.due <= end).sort(([, a], [, b]) => a.due - b.due)[0];
      if (!next) break;
      timers.delete(next[0]);
      now = next[1].due;
      next[1].handler();
      await quiet();
    }
    now = end;
  }

  const toggle = (feature: Feature): HTMLButtonElement => feature.querySelector<HTMLButtonElement>('[data-switch="fixture"]')!;

  /** A page whose second read is held until `release`, and whose command turns the Connection off. */
  async function overtaken(): Promise<{feature: Feature; release: () => void; reads: () => number}> {
    startClock();
    let reads = 0;
    let release = (): void => {};
    window.fetch = async (_url, init) => {
      if (init?.method) return Response.json({revision: '2', connections: [disabled], presets: []});
      reads += 1;
      // Sent before the command, the second read answers after it, with what was true before.
      if (reads === 2) await new Promise<void>((resolve) => { release = resolve; });
      return Response.json({revision: '1', connections: [enabled], presets: []});
    };
    const feature = mount();
    await loaded(feature);
    await pass(5000);
    await waitFor(() => reads === 2);
    toggle(feature).click();
    await waitFor(() => toggle(feature).getAttribute('aria-checked') === 'false' && !feature.busy);
    return {feature, release: () => release(), reads: () => reads};
  }

  it('reads again every five seconds, so a change made elsewhere shows up', async () => {
    startClock();
    let reads = 0;
    window.fetch = async () => Response.json({revision: String(++reads), connections: [draft], presets: []});
    const feature = mount();
    await loaded(feature);
    expect(reads).to.equal(1);

    await pass(4999);
    expect(reads).to.equal(1);
    await pass(1);
    expect(reads).to.equal(2);
    await pass(5000);
    expect(reads).to.equal(3);
  });

  it('does not read while a command runs, and reads again once it has answered', async () => {
    startClock();
    let reads = 0;
    let answer = (): void => {};
    window.fetch = async (_url, init) => {
      if (!init?.method) {
        reads += 1;
        return Response.json({revision: '1', connections: [enabled], presets: []});
      }
      await new Promise<void>((resolve) => { answer = resolve; });
      return Response.json({revision: '2', connections: [disabled], presets: []});
    };
    const feature = mount();
    await loaded(feature);
    toggle(feature).click();
    await waitFor(() => feature.busy);

    await pass(20_000);
    expect(reads).to.equal(1);

    answer();
    await waitFor(() => !feature.busy);
    await pass(5000);
    expect(reads).to.equal(2);
  });

  it('lets no read that a command overtook undo what the command answered', async () => {
    const {feature, release} = await overtaken();

    release();
    await quiet();

    expect(toggle(feature).getAttribute('aria-checked')).to.equal('false');
  });

  it('reads once per interval after a read that a command overtook', async () => {
    const {release, reads} = await overtaken();
    release();
    await quiet();

    const before = reads();
    await pass(30_000);

    expect(reads() - before).to.equal(6);
  });

  it('stops reading when Settings closes', async () => {
    startClock();
    let reads = 0;
    const api = wire({
      'GET /web/api/connections/mcp': () => {
        reads += 1;
        return Response.json({revision: '1', connections: [draft], presets: []});
      },
      'GET /web/api/memory/settings': () => memorySettings(false),
    });
    window.fetch = api.fetch;
    const {settings} = mountSettings();
    const dialog = await openSettings(settings, 'connections');
    await loaded(settings.querySelector('dl-settings-connections')!);
    expect(reads).to.equal(1);
    await pass(5000);
    expect(reads).to.equal(2);

    dialog.close();
    await waitFor(settingsClosed);
    await pass(30_000);

    expect(reads).to.equal(2);
  });
});

it('fills the header a Feishu preset needs and opens the Header tab on the new Connection', async () => {
  const commands: Wire[] = [];
  const presets = [presetView('feishu', 'Feishu/Lark', 'https://mcp.feishu.cn/mcp', 'header', {
    endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']],
    credential_header: 'X-Lark-MCP-TAT',
    credential_prefix: '',
  })];
  const created = {
    ...draft,
    connection_id: 'new',
    label: 'Feishu/Lark',
    endpoint: 'https://mcp.feishu.cn/mcp',
    endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']],
  };
  window.fetch = async (url, init) => {
    if (!init?.method) return Response.json({revision: '1', connections: [], presets});
    commands.push({url: String(url), method: init.method, body: JSON.parse(String(init.body))});
    return Response.json({revision: '2', connections: [created], presets});
  };
  const feature = mount();
  await loaded(feature);
  feature.querySelector<HTMLButtonElement>('[data-add]')!.click();
  await feature.updateComplete;

  // The preset states the allowlist without which the endpoint discovers nothing at all.
  buttonNamed(feature, 'Use the Feishu/Lark preset')!.click();
  await feature.updateComplete;
  const names = feature.querySelectorAll<HTMLInputElement>('[data-header-name="new"]');
  const values = feature.querySelectorAll<HTMLInputElement>('[data-header-value="new"]');
  expect(names).to.have.length(1);
  expect(names[0]!.value).to.equal('X-Lark-MCP-Allowed-Tools');
  expect(values[0]!.value).to.equal('create-doc,fetch-doc');

  // A row the owner added beside it travels with the same create command.
  feature.querySelector<HTMLButtonElement>('[data-new-header-add]')!.click();
  await feature.updateComplete;
  const added = feature.querySelectorAll<HTMLInputElement>('[data-header-name="new"]');
  added[1]!.value = 'X-Api-Version';
  added[1]!.dispatchEvent(new Event('input', {bubbles: true}));
  feature.querySelector<HTMLButtonElement>('button.primary-btn')!.click();
  await waitFor(() => commands.length === 1);
  expect(commands[0]!.body.endpoint_headers).to.deep.equal([
    ['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc'], ['X-Api-Version', ''],
  ]);

  // The card opens on a static token, whose transport the preset already answered: a header of
  // its own, named, with no prefix.
  await waitFor(() => Boolean(feature.querySelector('[data-card="new"]')));
  const card = feature.querySelector('[data-card="new"]')!.closest('article')!;
  const selected = [...card.querySelectorAll<HTMLButtonElement>('button[aria-pressed="true"]')];
  expect(selected.map((button) => button.textContent)).to.deep.equal(['Static token']);
  expect(card.querySelector<HTMLInputElement>('[data-transport="custom"]')!.checked).to.equal(true);
  expect(card.querySelector<HTMLInputElement>('[data-credential-header="new"]')!.value)
    .to.equal('X-Lark-MCP-TAT');
  expect(card.querySelector<HTMLInputElement>('[data-credential-prefix="new"]')!.value).to.equal('');
});

it('saves a credential in the header it names, and reads the name back after a reload', async () => {
  const commands: Wire[] = [];
  const stated = {...draft, authentication: 'header',
    endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc']],
    credential_header: 'X-Lark-MCP-TAT', credential_prefix: ''};
  window.fetch = replies([stated], commands);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;

  // What the server read back is what the form shows: the row above already says which header
  // this Connection's credential travels in, and the form agrees with it.
  expect(card.querySelector<HTMLInputElement>('[data-credential-header="fixture"]')!.value)
    .to.equal('X-Lark-MCP-TAT');
  expect(feature.querySelector('[data-card="fixture"]')!.textContent).to.contain('Credential in X-Lark-MCP-TAT');

  const token = card.querySelector<HTMLInputElement>('[data-credential-token="fixture"]')!;
  token.value = 't-write-only-token';
  buttonNamed(card, 'Save credential')!.click();
  await waitFor(() => commands.length === 1);

  expect(commands[0]!.url).to.equal('/web/api/connections/mcp/fixture/header');
  expect(commands[0]!.method).to.equal('PUT');
  expect(commands[0]!.body).to.deep.equal({
    expected_revision: '1', header: 'X-Lark-MCP-TAT', prefix: '', token: 't-write-only-token',
    endpoint: 'https://fixture.example/mcp',
  });
  // The token is never read back; the header the credential travels in is.
  await waitFor(() => feature.querySelector<HTMLInputElement>('[data-credential-token="fixture"]')!.value === '');
  expect(feature.querySelector<HTMLInputElement>('[data-credential-header="fixture"]')!.value)
    .to.equal('X-Lark-MCP-TAT');
});

it('edits the headers a Connection states, dropping a row that names nothing', async () => {
  const commands: Wire[] = [];
  const stated = {...draft, endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc']]};
  window.fetch = replies([stated], commands);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;
  // What the Connection already states is what the form shows.
  expect(card.querySelector<HTMLInputElement>('[data-header-name="fixture"]')!.value)
    .to.equal('X-Lark-MCP-Allowed-Tools');

  buttonNamed(card, 'Add header')!.click();
  await feature.updateComplete;
  const names = card.querySelectorAll<HTMLInputElement>('[data-header-name="fixture"]');
  const values = card.querySelectorAll<HTMLInputElement>('[data-header-value="fixture"]');
  expect(names).to.have.length(2);
  names[1]!.value = 'X-Api-Version';
  names[1]!.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.updateComplete;
  values[1]!.value = '2026-01-01';
  values[1]!.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.updateComplete;
  buttonNamed(card, 'Save headers')!.click();
  await waitFor(() => commands.length === 1);

  expect(commands[0]!.method).to.equal('PATCH');
  expect(commands[0]!.body.endpoint_headers).to.deep.equal([
    ['X-Lark-MCP-Allowed-Tools', 'create-doc'], ['X-Api-Version', '2026-01-01'],
  ]);

  // Removing a row takes it out of what the form states, and an empty row states nothing.
  await waitFor(() => !feature.busy);
  buttonNamed(card, 'Remove header')!.click();
  await feature.updateComplete;
  buttonNamed(card, 'Add header')!.click();
  await feature.updateComplete;
  buttonNamed(card, 'Remove header')!.click();
  await feature.updateComplete;
  buttonNamed(card, 'Save headers')!.click();
  await waitFor(() => commands.length === 2);
  expect(commands[1]!.body.endpoint_headers).to.deep.equal([]);
});

it('keeps the headers an owner typed when the server refuses them', async () => {
  const stated = {...draft, endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc']]};
  window.fetch = async (_url, init) => {
    if (init?.method) {
      // The module's own refusal envelope, as the route sends it.
      return Response.json({kind: 'rejected', message: 'An endpoint header may not carry a credential'}, {status: 422});
    }
    return Response.json({revision: '1', connections: [stated], presets: []});
  };
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;
  const name = card.querySelector<HTMLInputElement>('[data-header-name="fixture"]')!;
  name.value = 'Authorization';
  name.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.updateComplete;

  buttonNamed(card, 'Save headers')!.click();
  await waitFor(() => feature.error);
  await waitFor(() => !feature.busy);

  // The module's own words are what the reader reads, and the row the owner typed is still there.
  expect(feature.textContent).to.contain('An endpoint header may not carry a credential');
  expect(feature.textContent).not.to.contain('revision changed');
  expect(card.querySelector<HTMLInputElement>('[data-header-name="fixture"]')!.value).to.equal('Authorization');
  expect(card.querySelector<HTMLInputElement>('[data-header-value="fixture"]')!.value).to.equal('create-doc');
});

it('refuses a stated header that states a value under no name', async () => {
  const stated = {...draft, endpoint_headers: []};
  window.fetch = replies([stated]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;
  buttonNamed(card, 'Add header')!.click();
  await feature.updateComplete;
  const value = card.querySelector<HTMLInputElement>('[data-header-value="fixture"]')!;
  value.value = 'create-doc';
  value.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.updateComplete;

  buttonNamed(card, 'Save headers')!.click();
  await feature.updateComplete;

  // Nothing is sent, and the form says which row is incomplete instead of dropping it silently.
  expect(feature.textContent).to.contain('A stated header needs a name');
  expect(card.querySelector<HTMLInputElement>('[data-header-value="fixture"]')!.value).to.equal('create-doc');
});

it('says a server published no tools, instead of calling it ready', async () => {
  const empty = {...draft, enabled: true, status: 'no-tools'};
  window.fetch = replies([empty]);
  const feature = mount();
  await loaded(feature);

  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;
  // The row says what the observation was, and the note says what to look at -- the request
  // headers this Connection states -- because that is what a tool allowlist turns on.
  expect(card.querySelector('[data-card="fixture"]')!.textContent).to.contain('No tools');
  feature.querySelector<HTMLButtonElement>('[data-card="fixture"]')!.click();
  await feature.updateComplete;
  expect(card.textContent).to.contain('no tools under the request headers');
  expect(feature.querySelector('.dl-sr-only')?.textContent).to.contain('needs attention');
});

it('never shows a guessed header name for a credential the deployment cannot read', async () => {
  // The preset endpoint, a stored header credential, and a deployment that cannot open the
  // envelope: the field stays empty and the note says why. Offering the preset's name here
  // would read as a read-back beside the sentence saying it cannot be read.
  const stored = {...draft, label: 'Feishu/Lark', endpoint: 'https://mcp.feishu.cn/mcp',
    authentication: 'header', credential_header: null, credential_prefix: null};
  window.fetch = replies([stored], undefined, [
    presetView('feishu', 'Feishu/Lark', 'https://mcp.feishu.cn/mcp', 'header', {
      endpoint_headers: [['X-Lark-MCP-Allowed-Tools', 'create-doc,fetch-doc']],
      credential_header: 'X-Lark-MCP-TAT',
      credential_prefix: '',
    }),
  ]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;

  expect(card.querySelector<HTMLInputElement>('[data-credential-header="fixture"]')!.value).to.equal('');
  expect(card.textContent).to.contain('cannot read its key ring');
  expect(feature.querySelector('[data-card="fixture"]')!.textContent).not.to.contain('Credential in');
});

it('names the header a bearer travels in, and offers no link to a transport it has not got', async () => {
  const bearer = {...draft, authentication: 'bearer', enabled: true, status: 'ready'};
  window.fetch = replies([bearer]);
  const feature = mount();
  await openCard(feature, 'fixture');
  const card = feature.querySelector('[data-card="fixture"]')!.closest('article')!;

  // `bearer` is the default instance: it names itself instead of claiming to be unreadable.
  expect(card.textContent).to.contain('Authorization:');
  expect(card.textContent).not.to.contain('cannot read');
  expect(card.querySelector('[data-transport]')).not.to.equal(null);
});

it('says what to try next when a credential is refused, by what the Connection holds', async () => {
  const refused = {authentication: 'bearer', enabled: false, status: 'needs-auth'} as const;
  for (const [authentication, expected, forbidden] of [
    ['bearer', 'move the credential to a header of its own', 'Reauthorize'],
    ['header', 'a user token and an app token are not interchangeable', 'Reauthorize'],
    ['oauth', 'Reauthorize', 'header of its own'],
    ['none', 'Add a static token', 'Reauthorize'],
  ] as const) {
    window.fetch = replies([{...draft, ...refused, authentication}]);
    const feature = mount();
    await openCard(feature, 'fixture');

    expect(feature.textContent).to.contain(expected);
    expect(feature.textContent).not.to.contain(forbidden);
    if (authentication === 'header' || authentication === 'bearer') {
      // Both causes are named for a static token: a lost key ring needs no header change, and
      // the card cannot tell the two apart, so it must not assert a refusal that never reached
      // the server. OAuth has one cause and one move, so its sentence stays as it was.
      expect(feature.textContent).to.contain('or this deployment cannot open it');
    }
    document.body.replaceChildren();
  }
});

it('tells a reader configuring an unknown server how the card will guide them', async () => {
  window.fetch = replies([]);
  const feature = mount();
  await loaded(feature);
  feature.querySelector<HTMLButtonElement>('[data-add]')!.click();
  await feature.updateComplete;

  expect(feature.textContent).to.contain('Create it with the endpoint alone');
});

it('answers a token Connection by its own kind, not by an authorization it no longer runs', async () => {
  // A Connection keeps its latest OAuth flow row for as long as the row lives, whatever
  // credential it holds now. Reading that row first told a refused bearer to "authorize again",
  // which is not a move it has.
  const token = {...draft, authentication: 'bearer', status: 'needs-auth', authorization_status: 'failed'};
  window.fetch = replies([token]);
  const feature = mount();
  await openCard(feature, 'fixture');

  expect(feature.textContent).to.contain('move the credential to a header of its own');
  expect(feature.textContent).not.to.contain('Authorization failed or expired');
});

it('still tells a reader authorizing that the flow failed', async () => {
  const authorizing = {...draft, authentication: 'none', status: 'degraded', authorization_status: 'failed'};
  window.fetch = replies([authorizing]);
  const feature = mount();
  await openCard(feature, 'fixture');

  expect(feature.textContent).to.contain('Authorization failed or expired');
});

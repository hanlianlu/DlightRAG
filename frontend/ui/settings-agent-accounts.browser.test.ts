// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {waitFor} from '../testing/dom.ts';
import {
  type Handler,
  type WireAccount,
  type Wire,
  agentAccountsView,
  daysAgo,
  mountSettings,
  openSettings,
  wire,
  wireAccount,
} from '../testing/settings.ts';
import './settings-agent-accounts.ts';
import type {SettingsSummary} from './settings-summary.ts';

const originalFetch = window.fetch;

/** A page wide enough for the table, and one too narrow for it (the line is 36rem, 576px). */
const WIDE = 700;
const NARROW = 400;

type Feature = HTMLElementTagNameMap['dl-settings-agent-accounts'];

/** Four accounts that between them cover every case a row can be in. */
const ACCOUNTS: WireAccount[] = [
  wireAccount('discourse.org', {
    email: 'agent@discourse.org', username: 'dlr-discourse',
    created_at: '2025-03-04T12:00:00Z', last_used_at: daysAgo(0),
  }),
  wireAccount('huggingface.co', {
    email: 'agent@huggingface.co', username: 'dlight-agent',
    created_at: '2025-03-05T12:00:00Z', last_used_at: daysAgo(3),
  }),
  wireAccount('ycombinator.com', {
    email: null, username: 'dlight_reader', created_at: '2025-03-06T12:00:00Z', last_used_at: null,
  }),
  wireAccount('example.net', {
    email: null, username: null, created_at: '2025-03-07T12:00:00Z', last_used_at: daysAgo(400),
  }),
];

/** The double-submit cookie a mutation's CSRF header is read from. */
function csrfCookie(token: string | null): void {
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = token === null ? 'dlightrag_web_csrf=; max-age=0' : `dlightrag_web_csrf=${token}`;
}

afterEach(async () => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  document.body.className = '';
  csrfCookie(null);
});

/** The page as the dialog gives it: a block as wide as the pane it is in. */
function mount(width: number): Feature {
  const pane = document.createElement('div');
  pane.style.width = `${width}px`;
  const feature = document.createElement('dl-settings-agent-accounts');
  feature.style.display = 'block';
  pane.append(feature);
  document.body.append(pane);
  return feature;
}

/** Mount the page in a pane of one width and wait for it to read. */
async function shown(routes: Record<string, Handler> = {}, width = WIDE): Promise<{feature: Feature; api: Wire}> {
  const api = wire(routes);
  window.fetch = api.fetch;
  const feature = mount(width);
  await waitFor(() => feature.view !== null || feature.error);
  await feature.updateComplete;
  return {feature, api};
}

function listing(accounts: WireAccount[] = ACCOUNTS): Record<string, Handler> {
  return {'GET /web/api/agent-accounts': () => Response.json(agentAccountsView(accounts))};
}

function cells(row: Element): string[] {
  return [...row.children].map((cell) => cell.textContent!.replace(/\s+/g, ' ').trim());
}

function removeButton(feature: Feature, site: string): HTMLElement {
  return feature.querySelector<HTMLElement>(`[data-remove="${site}"]`)!;
}

function registration(feature: Feature): HTMLButtonElement {
  return feature.querySelector<HTMLButtonElement>('#agent-accounts-registration')!;
}

function caption(feature: Feature): string {
  return document.getElementById(registration(feature).getAttribute('aria-describedby')!)!.textContent!.trim();
}

/** The notices the page asked the shell to show. */
function notices(): string[] {
  const heard: string[] = [];
  document.body.addEventListener('dl-toast-request', (event) => { heard.push(event.detail.message); });
  return heard;
}

it('lists each website with how the agent signs in there, when it registered, and when it last did', async () => {
  const {feature, api} = await shown(listing());

  const table = feature.querySelector('table')!;
  expect([...table.querySelectorAll('thead th')].map((cell) => cell.textContent!.trim()))
    .to.deep.equal(['Website', 'Sign-in', 'Registered', 'Last sign-in', 'Remove']);
  expect([...table.querySelectorAll('thead th')].every((cell) => cell.getAttribute('scope') === 'col')).to.equal(true);
  const rows = [...table.querySelectorAll('tbody tr')];
  expect(rows.map((row) => row.querySelector('th')!.getAttribute('scope'))).to.deep.equal(['row', 'row', 'row', 'row']);
  expect(rows.map(cells)).to.deep.equal([
    ['D discourse.org', 'agent@discourse.org dlr-discourse', 'Mar 4, 2025', 'Today', ''],
    ['H huggingface.co', 'agent@huggingface.co dlight-agent', 'Mar 5, 2025', '3 days ago', ''],
    ['Y ycombinator.com', 'dlight_reader', 'Mar 6, 2025', '—Never', ''],
    ['E example.net', '—Not recorded', 'Mar 7, 2025', rows[3]!.children[3]!.textContent!.trim(), ''],
  ]);
  // Older than a week, a sign-in reads as a date, and the year shows when it is not this year's.
  expect(rows[3]!.children[3]!.textContent).to.match(/^\w{3} \d{1,2}, \d{4}$/);
  // The monogram is the site's first letter and decorative, and no row asks for an image.
  expect(rows[0]!.querySelector('[class*=tile]')!.getAttribute('aria-hidden')).to.equal('true');
  expect(feature.querySelector('img')).to.equal(null);
  expect(api.requests.map((request) => `${request.method} ${request.path}`)).to.deep.equal(['GET /web/api/agent-accounts']);
  expect(api.unexpected).to.deep.equal([]);
  expect([...feature.querySelectorAll('[data-remove]')].map((button) => button.getAttribute('aria-label')))
    .to.deep.equal(ACCOUNTS.map((account) => `Remove the account for ${account.site}`));
});

it('writes each account as three lines where the page is too narrow for a table: the website, how it signs in, and when it last did', async () => {
  const {feature} = await shown(listing(), NARROW);

  expect(feature.querySelector('table')).to.equal(null);
  const rows = [...feature.querySelectorAll('li')].map((row) => [...row.querySelectorAll('[class*=itemText] > span')]
    .map((line) => line.textContent!.trim()));
  // An account with no identity on file has no identity line to show.
  expect(rows).to.deep.equal([
    ['discourse.org', 'agent@discourse.org', 'Signed in today'],
    ['huggingface.co', 'agent@huggingface.co', 'Signed in 3 days ago'],
    ['ycombinator.com', 'dlight_reader', 'Registered Mar 6, 2025, not signed in since'],
    ['example.net', rows[3]![1]!],
  ]);
  expect(rows[3]![1]).to.match(/^Signed in \w{3} \d{1,2}, \d{4}$/);
});

it('turns from a table to three-line rows as its pane narrows past 36rem, and back, without losing an account', async () => {
  const {feature} = await shown(listing(), WIDE);
  const pane = feature.parentElement!;
  const layout = (): string => (feature.querySelector('table') ? 'table' : feature.querySelector('ul') ? 'rows' : 'none');
  expect(layout()).to.equal('table');

  // Just under and just over the line, the way a pane meets it: by being resized while open.
  pane.style.width = '560px';
  await waitFor(() => layout() === 'rows');
  expect(feature.querySelectorAll('li')).to.have.length(ACCOUNTS.length);
  pane.style.width = '600px';
  await waitFor(() => layout() === 'table');
  expect(feature.querySelectorAll('tbody tr')).to.have.length(ACCOUNTS.length);
});

it('shows when an account registered even where it never signed in and there is no room for a table', async () => {
  const {feature} = await shown(listing(), NARROW);

  const never = [...feature.querySelectorAll('li')]
    .find((row) => row.querySelector('[title]')?.getAttribute('title') === 'ycombinator.com')!;
  expect(never.textContent).to.contain('Registered Mar 6, 2025, not signed in since');
});

it('invites the first account when there is none, and keeps the switch', async () => {
  const {feature} = await shown(listing([]));

  expect(feature.querySelector('table')).to.equal(null);
  expect(feature.textContent).to.contain('No accounts yet');
  expect(feature.textContent).to.contain(
    'When Research meets a website that needs a free account, the agent signs up under its own identity and the account appears here.',
  );
  expect(registration(feature)).not.to.equal(null);
});

it('says it could not load, offers Retry, and shows the accounts once a retry reads them', async () => {
  let reads = 0;
  const {feature, api} = await shown({
    'GET /web/api/agent-accounts': () => {
      reads += 1;
      return reads === 1 ? new Response('unavailable', {status: 503}) : Response.json(agentAccountsView(ACCOUNTS));
    },
  });

  const alert = feature.querySelector('[role="alert"]')!;
  expect(alert.textContent).to.equal('Could not load agent accounts.');
  expect(registration(feature)).to.equal(null);
  const retry = [...feature.querySelectorAll('button')].find((button) => button.textContent!.trim() === 'Retry')!;
  retry.click();

  await waitFor(() => feature.querySelectorAll('tbody tr').length === 4);
  expect(feature.querySelector('[role="alert"]')).to.equal(null);
  expect(reads).to.equal(2);
  expect(api.unexpected).to.deep.equal([]);
});

describe('the sign-up switch', () => {
  it('is on, with what that lets the agent do', async () => {
    const {feature} = await shown(listing());

    expect(registration(feature).getAttribute('role')).to.equal('switch');
    expect(registration(feature).getAttribute('aria-checked')).to.equal('true');
    expect(registration(feature).disabled).to.equal(false);
    expect(document.getElementById(registration(feature).getAttribute('aria-labelledby')!)!.textContent)
      .to.equal('Allow new sign-ups');
    expect(caption(feature)).to.equal('The agent may register on a website when it needs to');
  });

  it('is off when the owner turned it off, and says the agent only signs in', async () => {
    const {feature} = await shown({
      'GET /web/api/agent-accounts': () => Response.json(agentAccountsView(ACCOUNTS, {allowed: true, enabled: false})),
    });

    expect(registration(feature).getAttribute('aria-checked')).to.equal('false');
    expect(registration(feature).disabled).to.equal(false);
    expect(caption(feature)).to.equal('Off: the agent only signs in with the accounts below');
  });

  it('is off and out of reach when the deployment does not allow sign-ups', async () => {
    const {feature} = await shown({
      // The owner's own switch is on, but the deployment's ceiling is what the agent meets.
      'GET /web/api/agent-accounts': () => Response.json(agentAccountsView(ACCOUNTS, {allowed: false, enabled: true})),
    });

    expect(registration(feature).getAttribute('aria-checked')).to.equal('false');
    expect(registration(feature).disabled).to.equal(true);
    expect(caption(feature)).to.equal(
      'This deployment does not allow sign-ups, so the agent only signs in with the accounts below.',
    );
    expect(feature.querySelectorAll('tbody tr')).to.have.length(4);
  });

  it('is out of reach when the deployment has not enabled Agent Accounts, and stored accounts can still go', async () => {
    const {feature} = await shown({
      'GET /web/api/agent-accounts': () => Response.json(agentAccountsView(ACCOUNTS, {allowed: false, enabled: true}, false)),
    });

    expect(registration(feature).disabled).to.equal(true);
    expect(caption(feature)).to.equal(
      'This deployment has not enabled Agent Accounts. Stored accounts can still be removed here.',
    );
    expect(removeButton(feature, 'discourse.org').hasAttribute('disabled')).to.equal(false);
  });

  it('turns with one PUT carrying the CSRF token, shows the view it answers, and also turns from its card', async () => {
    csrfCookie('csrf-token');
    let enabled = true;
    const {feature, api} = await shown({
      ...listing(),
      'PUT /web/api/agent-accounts/settings': (request) => {
        enabled = (request.body as {registration_enabled: boolean}).registration_enabled;
        return Response.json(agentAccountsView(ACCOUNTS, {allowed: true, enabled}));
      },
    });

    registration(feature).click();
    await waitFor(() => registration(feature).getAttribute('aria-checked') === 'false' && !registration(feature).disabled);
    const puts = api.requests.filter((request) => request.method === 'PUT');
    expect(puts.map((request) => request.body)).to.deep.equal([{registration_enabled: false}]);
    expect(puts[0]!.headers.get('X-CSRF-Token')).to.equal('csrf-token');
    expect(caption(feature)).to.equal('Off: the agent only signs in with the accounts below');

    // The card is the switch's label, so a tap on its words turns it back on.
    feature.querySelector<HTMLElement>('#agent-accounts-registration-label')!.click();
    await waitFor(() => registration(feature).getAttribute('aria-checked') === 'true' && !registration(feature).disabled);
    expect(api.requests.filter((request) => request.method === 'PUT').map((request) => request.body))
      .to.deep.equal([{registration_enabled: false}, {registration_enabled: true}]);
  });

  it('keeps its state after a refused PUT, and says so', async () => {
    const heard = notices();
    const {feature} = await shown({
      ...listing(),
      'PUT /web/api/agent-accounts/settings': () => new Response('unavailable', {status: 503}),
    });

    registration(feature).click();
    await waitFor(() => heard.length === 1);
    await waitFor(() => !registration(feature).disabled);

    expect(heard).to.deep.equal(['Could not save the sign-up setting.']);
    expect(registration(feature).getAttribute('aria-checked')).to.equal('true');
  });
});

describe('removing an account', () => {
  const removal = (sites: string[]): Handler => () => Response.json(
    agentAccountsView(ACCOUNTS.filter((account) => !sites.includes(account.site))),
  );

  it('asks first, names the website, and sends nothing until the owner agrees', async () => {
    const {feature, api} = await shown(listing());
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;
    const trigger = removeButton(feature, 'huggingface.co');

    trigger.click();
    await waitFor(() => dialog.open);
    expect(dialog.getAttribute('aria-labelledby')).to.equal('agent-accounts-remove-title');
    expect(document.getElementById('agent-accounts-remove-title')!.textContent)
      .to.equal('Remove the account for huggingface.co?');
    expect(dialog.textContent).to.contain(
      'DlightRAG deletes the saved sign-in email and sealed password, and the agent can no longer sign in there. The account itself stays on the website.',
    );
    expect([...dialog.querySelectorAll('button')].map((button) => button.textContent!.trim()))
      .to.deep.equal(['Cancel', 'Remove account']);

    dialog.querySelector<HTMLButtonElement>('button[value=cancel]')!.click();
    await waitFor(() => !dialog.open && document.activeElement === trigger);
    expect(api.requests.some((request) => request.method === 'DELETE')).to.equal(false);
    expect(feature.querySelectorAll('tbody tr')).to.have.length(4);
  });

  it('removes it with one DELETE carrying the CSRF token, shows the view that answers, and keeps focus on the list', async () => {
    csrfCookie('csrf-token');
    const heard: SettingsSummary[] = [];
    document.body.addEventListener('dl-settings-summary', (event) => { heard.push(event.detail); });
    const {feature, api} = await shown({
      ...listing(),
      'DELETE /web/api/agent-accounts/huggingface.co': removal(['huggingface.co']),
    });
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;

    removeButton(feature, 'huggingface.co').click();
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value=remove]')!.click();
    await waitFor(() => feature.querySelectorAll('tbody tr').length === 3);

    const deletes = api.requests.filter((request) => request.method === 'DELETE');
    expect(deletes.map((request) => request.path)).to.deep.equal(['/web/api/agent-accounts/huggingface.co']);
    expect(deletes[0]!.headers.get('X-CSRF-Token')).to.equal('csrf-token');
    expect([...feature.querySelectorAll('tbody [class*=siteName]')].map((name) => name.textContent!.trim()))
      .to.deep.equal(['discourse.org', 'ycombinator.com', 'example.net']);
    // The row the reader was on is gone: focus goes to the one that took its place.
    await waitFor(() => document.activeElement === removeButton(feature, 'ycombinator.com'));
    expect(heard.at(-1)).to.deep.equal({section: 'agent-accounts', count: 3});
  });

  it('shows the empty state and hands focus to the switch when the last one goes', async () => {
    const {feature} = await shown({
      ...listing([ACCOUNTS[0]!]),
      'DELETE /web/api/agent-accounts/discourse.org': () => Response.json(agentAccountsView([])),
    });
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;

    removeButton(feature, 'discourse.org').click();
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value=remove]')!.click();
    await waitFor(() => feature.textContent!.includes('No accounts yet'));

    await waitFor(() => document.activeElement === registration(feature));
  });

  it('hands focus to the note that nothing is left when the switch is out of reach for the deployment', async () => {
    const closed = {allowed: false, enabled: true};
    const {feature} = await shown({
      'GET /web/api/agent-accounts': () => Response.json(agentAccountsView([ACCOUNTS[0]!], closed)),
      'DELETE /web/api/agent-accounts/discourse.org': () => Response.json(agentAccountsView([], closed)),
    });
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;
    expect(registration(feature).disabled).to.equal(true);

    removeButton(feature, 'discourse.org').click();
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value=remove]')!.click();
    await waitFor(() => feature.textContent!.includes('No accounts yet'));

    const note = [...feature.querySelectorAll('h4')].find((heading) => heading.textContent!.trim() === 'No accounts yet')!;
    await waitFor(() => document.activeElement === note);
  });

  it('treats a 404 as "already gone": it reads the list again and says nothing', async () => {
    const heard = notices();
    let gone = false;
    const {feature, api} = await shown({
      'GET /web/api/agent-accounts': () => Response.json(agentAccountsView(
        ACCOUNTS.filter((account) => !gone || account.site !== 'huggingface.co'),
      )),
      'DELETE /web/api/agent-accounts/huggingface.co': () => {
        gone = true;
        return Response.json({detail: 'No account for this website', error_type: 'not_found', error_kind: null}, {status: 404});
      },
    });
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;

    removeButton(feature, 'huggingface.co').click();
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value=remove]')!.click();
    await waitFor(() => feature.querySelectorAll('tbody tr').length === 3);

    expect(api.requests.map((request) => request.method)).to.deep.equal(['GET', 'DELETE', 'GET']);
    expect(heard).to.deep.equal([]);
  });

  it('keeps the row and says so when the server refuses', async () => {
    const heard = notices();
    const {feature} = await shown({
      ...listing(),
      'DELETE /web/api/agent-accounts/huggingface.co': () => new Response('boom', {status: 500}),
    });
    const dialog = feature.querySelector<HTMLDialogElement>('#agent-accounts-remove')!;
    const trigger = removeButton(feature, 'huggingface.co');

    trigger.click();
    await waitFor(() => dialog.open);
    dialog.querySelector<HTMLButtonElement>('button[value=remove]')!.click();
    await waitFor(() => heard.length === 1);

    expect(heard).to.deep.equal(['Could not remove the account.']);
    expect(feature.querySelectorAll('tbody tr')).to.have.length(4);
  });
});

describe('what the page never shows', () => {
  it('renders no field at all, so nothing secret can be typed or revealed on it', async () => {
    const {feature} = await shown(listing());

    expect(feature.querySelectorAll('input, textarea, select, [type="password"]')).to.have.length(0);
    expect(feature.textContent).not.to.match(/hunter|secret|token/i);
  });

  it('refuses a reply that carries a credential, and never puts it on the page', async () => {
    const {feature, api} = await shown({
      'GET /web/api/agent-accounts': () => Response.json({
        ...agentAccountsView(ACCOUNTS),
        accounts: ACCOUNTS.map((account) => ({...account, password: 'hunter2-secret-password'})),
      }),
    });

    expect(feature.error).to.equal(true);
    expect(feature.querySelector('[role="alert"]')!.textContent).to.equal('Could not load agent accounts.');
    expect(document.documentElement.outerHTML).not.to.contain('hunter2-secret-password');
    expect(api.unexpected).to.deep.equal([]);
  });
});

it('tells the dialog how many accounts there are', async () => {
  const heard: SettingsSummary[] = [];
  document.body.addEventListener('dl-settings-summary', (event) => { heard.push(event.detail); });
  await shown(listing());

  expect(heard).to.deep.equal([{section: 'agent-accounts', count: 4}]);
});

it('speaks the reader\'s language, dates included', async () => {
  try {
    await setLanguagePreference('zh');
    const {feature} = await shown(listing());

    expect([...feature.querySelectorAll('thead th')].map((cell) => cell.textContent!.trim()))
      .to.deep.equal(['网站', '登录身份', '注册', '最近登录', '移除']);
    const first = feature.querySelector('tbody tr')!;
    expect(first.children[2]!.textContent!.trim()).to.equal('2025年3月4日');
    expect(first.children[3]!.textContent!.trim()).to.equal('今天');
    expect(removeButton(feature, 'discourse.org').getAttribute('aria-label')).to.equal('移除 discourse.org 的账号');
    expect(caption(feature)).to.equal('代理需要时可以在网站上自己注册');
  } finally {
    await setLanguagePreference('en');
  }
});

it('shows a notice in the dialog\'s own region when it lives in the dialog', async () => {
  window.fetch = wire({
    ...listing(),
    'PUT /web/api/agent-accounts/settings': () => new Response('unavailable', {status: 503}),
    'GET /web/api/memory/settings': () => Response.json({enabled: false, active_count: null}),
  }).fetch;
  const {settings} = mountSettings();
  await openSettings(settings, 'agent-accounts');
  const feature = settings.querySelector('dl-settings-agent-accounts')!;
  await waitFor(() => feature.view !== null);
  await feature.updateComplete;

  registration(feature).click();
  const own = settings.querySelector('dl-toast-region')!;
  await waitFor(() => own.textContent!.includes('Could not save the sign-up setting.'));
});

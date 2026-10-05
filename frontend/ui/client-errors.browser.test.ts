// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {CSRF_COOKIE_NAME} from '../api/csrf.ts';

type Page = Window & typeof globalThis & {
  reports: {url: string; method: string; headers: Record<string, string>; body: string; keepalive: boolean}[];
  respond: () => unknown;
};

// The reporter's listeners and its report budget belong to one page load, so
// each test loads the fixture page into a fresh frame and raises errors there.
async function load(): Promise<Page> {
  const frame = document.createElement('iframe');
  frame.src = '/ui/fixtures/client-errors.html';
  const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, {once: true}));
  document.body.append(frame);
  await loaded;
  return frame.contentWindow as Page;
}

const raise = (page: Page, value: unknown) => page.dispatchEvent(new page.CustomEvent('raise', {detail: value}));

function reject(page: Page, ...values: unknown[]): Promise<void> {
  return new Promise((resolve) => {
    let seen = 0;
    page.addEventListener('unhandledrejection', () => { if (++seen === values.length) resolve(); });
    for (const value of values) page.dispatchEvent(new page.CustomEvent('reject', {detail: value}));
  });
}

const detailOf = (page: Page): string => JSON.parse(page.reports[0].body).detail;
const quiet = () => new Promise((resolve) => setTimeout(resolve, 50));

function csrfCookie(token: string | null): void {
  // biome-ignore lint/suspicious/noDocumentCookie: the CSRF double-submit cookie is the channel under test
  document.cookie = token === null ? `${CSRF_COOKIE_NAME}=; max-age=0` : `${CSRF_COOKIE_NAME}=${token}`;
}

beforeEach(() => csrfCookie('token'));
afterEach(() => {
  csrfCookie(null);
  document.body.replaceChildren();
});

it('reports an uncaught error as its message and stack, with the CSRF token', async () => {
  const page = await load();
  raise(page, new page.Error('boom'));
  expect(page.reports).to.have.length(1);
  const [report] = page.reports;
  expect(report.url).to.equal('/web/api/client-errors');
  expect(report.method).to.equal('POST');
  expect(report.headers).to.deep.equal({'Content-Type': 'application/json', 'X-CSRF-Token': 'token'});
  expect(report.keepalive).to.equal(true);
  expect(Object.keys(JSON.parse(report.body))).to.deep.equal(['detail']);
  expect(detailOf(page)).to.match(/^boom\n.+/);
});

it('reports an unhandled rejection', async () => {
  const page = await load();
  await reject(page, new page.Error('later'));
  expect(page.reports).to.have.length(1);
  expect(detailOf(page)).to.match(/^later\n.+/);
});

it('reports a thrown value that is not an Error', async () => {
  const page = await load();
  raise(page, 'plain text');
  expect(detailOf(page)).to.equal('plain text');
});

it('reports nothing for a rejection that has no text', async () => {
  const page = await load();
  await reject(page, page.Object.create(null));
  expect(page.reports).to.have.length(0);
});

it('reports the same error once however often it is raised', async () => {
  const page = await load();
  const error = new page.Error('render failed');
  await reject(page, error, error);
  raise(page, error);
  expect(page.reports).to.have.length(1);
});

it('reports at most five different errors', async () => {
  const page = await load();
  for (let index = 0; index < 7; index += 1) raise(page, new page.Error(`failure ${index}`));
  expect(page.reports).to.have.length(5);
});

it('cuts a long report to the length the server accepts, never inside a character', async () => {
  const page = await load();
  raise(page, `${'x'.repeat(1999)}😀😀`);
  expect(detailOf(page)).to.equal(`${'x'.repeat(1999)}😀`);
});

it('raises no further report when sending one fails', async () => {
  const page = await load();
  page.respond = () => page.Promise.reject(new page.TypeError('Failed to fetch'));
  raise(page, new page.Error('boom'));
  await quiet();
  expect(page.reports).to.have.length(1);
});

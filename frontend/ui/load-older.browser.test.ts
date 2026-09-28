// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {render} from 'lit';
import type {KeysetPagerStatus} from '../lib/paged.ts';
import {loadOlderControl} from './load-older.ts';

function mount(pages: KeysetPagerStatus, onLoad: (event: Event) => void = () => {}): HTMLElement {
  const host = document.createElement('div');
  document.body.appendChild(host);
  render(loadOlderControl({
    list: 'notes',
    pages,
    label: 'Load older notes',
    retryLabel: 'Retry loading older notes',
    loading: 'Loading older notes…',
    loaded: 'Loaded 2 older notes.',
    failed: 'Older notes could not be loaded.',
    onLoad,
    rowClass: 'notes-more',
  }), host);
  return host;
}

afterEach(() => {
  document.body.replaceChildren();
});

it('offers the next page through one named, busy-aware button', () => {
  let loads = 0;
  const host = mount({state: 'idle', hasOlder: true, outcome: null}, () => { loads += 1; });
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;

  expect(button.type).to.equal('button');
  expect(button.textContent?.trim()).to.equal('Load older notes');
  expect(button.getAttribute('aria-busy')).to.equal('false');
  expect(button.closest('.notes-more')).not.to.equal(null);
  button.click();
  expect(loads).to.equal(1);

  const status = host.querySelector<HTMLElement>('[data-load-older-status="notes"]')!;
  expect(status.getAttribute('role')).to.equal('status');
  expect(status.getAttribute('aria-live')).to.equal('polite');
  expect(status.textContent?.trim()).to.equal('');
});

it('marks a loading page busy and announces it politely', () => {
  const host = mount({state: 'loading', hasOlder: true, outcome: null});
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;

  expect(button.disabled).to.equal(true);
  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Loading older notes…');
});

it('turns a failed page into a retry and says why', () => {
  const host = mount({state: 'error', hasOlder: true, outcome: 'failed'});

  expect(host.querySelector('[data-load-older="notes"]')?.textContent?.trim())
    .to.equal('Retry loading older notes');
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Older notes could not be loaded.');
});

it('keeps the last announcement after the final page removes the button', () => {
  const host = mount({state: 'idle', hasOlder: false, outcome: 'loaded'});

  expect(host.querySelector('[data-load-older="notes"]')).to.equal(null);
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Loaded 2 older notes.');
});

it('says nothing about a first page that is still loading or failed', () => {
  for (const state of ['loading', 'error'] as const) {
    const host = mount({state, hasOlder: false, outcome: null});
    expect(host.querySelector('[data-load-older="notes"]')).to.equal(null);
    expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim()).to.equal('');
  }
});

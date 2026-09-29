// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {render} from 'lit';
import type {KeysetPagerStatus} from '../lib/paged.ts';
import {loadOlderControl} from './load-older.ts';

const IDLE: KeysetPagerStatus = {state: 'idle', starting: false, hasOlder: true, outcome: null};

function draw(host: HTMLElement, pages: KeysetPagerStatus, onLoad: (event: Event) => void): void {
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
}

function mount(pages: KeysetPagerStatus, onLoad: (event: Event) => void = () => {}) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  draw(host, pages, onLoad);
  return {host, redraw: (next: KeysetPagerStatus) => { draw(host, next, onLoad); }};
}

afterEach(() => {
  document.body.replaceChildren();
});

it('offers the next page through one named, busy-aware button', () => {
  let loads = 0;
  const {host} = mount(IDLE, () => { loads += 1; });
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;

  expect(button.type).to.equal('button');
  expect(button.textContent?.trim()).to.equal('Load older notes');
  expect(button.getAttribute('aria-busy')).to.equal('false');
  expect(button.hasAttribute('aria-disabled')).to.equal(false);
  expect(button.closest('.notes-more')).not.to.equal(null);
  button.click();
  expect(loads).to.equal(1);

  const status = host.querySelector<HTMLElement>('[data-load-older-status="notes"]')!;
  expect(status.getAttribute('role')).to.equal('status');
  expect(status.getAttribute('aria-live')).to.equal('polite');
  expect(status.textContent?.trim()).to.equal('');
});

it('marks a loading page busy and announces it politely', () => {
  const {host} = mount({...IDLE, state: 'loading'});
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;

  expect(button.disabled).to.equal(false);
  expect(button.getAttribute('aria-disabled')).to.equal('true');
  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Loading older notes…');
});

it('keeps focus on a busy button and ignores its activation until the page lands', () => {
  let loads = 0;
  const {host, redraw} = mount(IDLE, () => { loads += 1; });
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;
  button.focus();

  redraw({...IDLE, state: 'loading'});
  button.click();
  expect(loads).to.equal(0);
  // Compare identities: a failing DOM-node equality would stall the reporter.
  expect(document.activeElement === button, 'the busy button keeps focus').to.equal(true);

  redraw({...IDLE, outcome: 'loaded'});
  expect(host.querySelector('[data-load-older="notes"]') === button).to.equal(true);
  expect(document.activeElement === button, 'focus stays after the page lands').to.equal(true);
  button.click();
  expect(loads).to.equal(1);
});

it('keeps a reloading list\'s button busy without announcing a next page', () => {
  const {host} = mount({state: 'loading', starting: true, hasOlder: true, outcome: null});
  const button = host.querySelector<HTMLButtonElement>('[data-load-older="notes"]')!;

  expect(button.getAttribute('aria-busy')).to.equal('true');
  expect(button.getAttribute('aria-disabled')).to.equal('true');
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim()).to.equal('');
});

it('turns a failed page into a retry and says why', () => {
  const {host} = mount({...IDLE, state: 'error', outcome: 'failed'});

  expect(host.querySelector('[data-load-older="notes"]')?.textContent?.trim())
    .to.equal('Retry loading older notes');
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Older notes could not be loaded.');
});

it('keeps the last announcement after the final page removes the button', () => {
  const {host} = mount({...IDLE, hasOlder: false, outcome: 'loaded'});

  expect(host.querySelector('[data-load-older="notes"]')).to.equal(null);
  expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim())
    .to.equal('Loaded 2 older notes.');
});

it('says nothing about a first page that is still loading or failed', () => {
  for (const [state, starting] of [['loading', true], ['error', false]] as const) {
    const {host} = mount({state, starting, hasOlder: false, outcome: null});
    expect(host.querySelector('[data-load-older="notes"]')).to.equal(null);
    expect(host.querySelector('[data-load-older-status="notes"]')?.textContent?.trim()).to.equal('');
  }
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlNotificationOffer} from './notifications.ts';
import './notifications.ts';
import {buttonNamed} from '../testing/dom.ts';

const notificationDescriptor = Object.getOwnPropertyDescriptor(window, 'Notification');
const originalHasFocus = document.hasFocus;

/** What the feature would have constructed, were it still a notification owner. */
let constructed: string[] = [];

function installNotification(permission: NotificationPermission): void {
  class FakeNotification {
    static permission = permission;
    static async requestPermission(): Promise<NotificationPermission> { return permission; }
    onclick: (() => void) | null = null;
    constructor(title: string) { constructed.push(title); }
    close(): void {}
  }
  Object.defineProperty(window, 'Notification', {
    configurable: true,
    value: FakeNotification,
  });
}

function offer(): DlNotificationOffer {
  const element = document.createElement('dl-notification-offer') as DlNotificationOffer;
  document.body.appendChild(element);
  return element;
}

beforeEach(() => {
  constructed = [];
  localStorage.removeItem('dlightrag-notify-asked');
});

afterEach(() => {
  document.body.replaceChildren();
  document.hasFocus = originalHasFocus;
  localStorage.removeItem('dlightrag-notify-asked');
  if (notificationDescriptor) Object.defineProperty(window, 'Notification', notificationDescriptor);
  else Reflect.deleteProperty(window, 'Notification');
});

it('leaves completion notifications to the one notifier: a missed answer constructs none', async () => {
  installNotification('granted');
  document.hasFocus = () => false;
  const element = offer();
  element.running = true;
  await element.updateComplete;
  element.running = false;
  await element.updateComplete;

  expect(constructed).to.deep.equal([], 'notifications.ts no longer constructs a Notification');

  element.remove();
  window.dispatchEvent(new Event('focus'));
  expect(constructed).to.deep.equal([], 'and not on the way back either');
});

it('offers permission when a missed answer is found on return, and remembers a decline', async () => {
  installNotification('default');
  document.hasFocus = () => false;
  const element = offer();
  element.running = true;
  await element.updateComplete;
  element.running = false;
  await element.updateComplete;
  expect(element.hidden, 'the offer waits for the reader to come back').to.equal(true);

  document.hasFocus = () => true;
  window.dispatchEvent(new Event('focus'));
  await element.updateComplete;

  expect(element.hidden).to.equal(false);
  expect(constructed).to.deep.equal([], 'asking permission is the offer, not a notification');
  buttonNamed(element, 'Not now')?.click();
  await element.updateComplete;
  expect(element.hidden).to.equal(true);
  expect(localStorage.getItem('dlightrag-notify-asked')).to.equal('1');
});

it('renders accessible permission intent', async () => {
  installNotification('default');
  const element = offer();
  element.setAttribute('role', 'group');
  element.setAttribute('aria-label', 'Answer notifications');
  element.visible = true;
  await element.updateComplete;

  expect(element.hidden).to.equal(false);
  expect(buttonNamed(element, 'Enable')).not.to.equal(null);
  expect(buttonNamed(element, 'Not now')).not.to.equal(null);
});

it('does not offer what was already answered, nor when permission is granted', async () => {
  installNotification('default');
  localStorage.setItem('dlightrag-notify-asked', '1');
  document.hasFocus = () => false;
  const asked = offer();
  asked.running = true;
  await asked.updateComplete;
  asked.running = false;
  await asked.updateComplete;
  document.hasFocus = () => true;
  window.dispatchEvent(new Event('focus'));
  await asked.updateComplete;
  expect(asked.hidden, 'a remembered answer is not asked again').to.equal(true);
  asked.remove();

  installNotification('granted');
  document.hasFocus = () => false;
  const granted = offer();
  granted.running = true;
  await granted.updateComplete;
  granted.running = false;
  await granted.updateComplete;
  document.hasFocus = () => true;
  window.dispatchEvent(new Event('focus'));
  await granted.updateComplete;
  expect(granted.hidden, 'permission already granted needs no offer').to.equal(true);
});

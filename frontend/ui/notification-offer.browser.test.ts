// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {DlNotificationOffer} from './notifications.ts';
import './notifications.ts';

const notificationDescriptor = Object.getOwnPropertyDescriptor(window, 'Notification');
const originalHasFocus = document.hasFocus;

function buttonNamed(root: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
    .find((button) => button.textContent?.trim() === name) ?? null;
}

function installNotification(permission: NotificationPermission, calls: string[]): void {
  class FakeNotification {
    static permission = permission;
    static async requestPermission(): Promise<NotificationPermission> { return permission; }
    onclick: (() => void) | null = null;
    constructor(title: string) { calls.push(title); }
    close(): void {}
  }
  Object.defineProperty(window, 'Notification', {
    configurable: true,
    value: FakeNotification,
  });
}

afterEach(() => {
  document.body.replaceChildren();
  document.hasFocus = originalHasFocus;
  localStorage.removeItem('dlightrag-notify-asked');
  if (notificationDescriptor) Object.defineProperty(window, 'Notification', notificationDescriptor);
  else Reflect.deleteProperty(window, 'Notification');
});

it('owns missed-run notification state and cleans its browser listeners', async () => {
  const notifications: string[] = [];
  installNotification('granted', notifications);
  document.hasFocus = () => false;
  const offer = document.createElement('dl-notification-offer') as DlNotificationOffer;
  document.body.appendChild(offer);

  offer.running = true;
  await offer.updateComplete;
  offer.running = false;
  await offer.updateComplete;

  expect(notifications).to.deep.equal(['Answer ready']);
  offer.remove();
  window.dispatchEvent(new Event('focus'));
  expect(notifications).to.deep.equal(['Answer ready']);
});

it('renders accessible permission intent and persists a decline', async () => {
  installNotification('default', []);
  const offer = document.createElement('dl-notification-offer') as DlNotificationOffer;
  offer.setAttribute('role', 'group');
  offer.setAttribute('aria-label', 'Answer notifications');
  document.body.appendChild(offer);
  offer.visible = true;
  await offer.updateComplete;

  expect(offer.hidden).to.equal(false);
  buttonNamed(offer, 'Not now')?.click();
  await offer.updateComplete;
  expect(offer.hidden).to.equal(true);
  expect(localStorage.getItem('dlightrag-notify-asked')).to.equal('1');
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {html, type TemplateResult} from 'lit';
import {LightElement, StoreController} from '../lib/lit-host.ts';

/** A store that counts how often it was followed and lets the test say it changed. */
class FakeStore {
  readonly followers = new Set<() => void>();
  subscriptions = 0;

  subscribe(handler: () => void): () => void {
    this.subscriptions += 1;
    this.followers.add(handler);
    return () => { this.followers.delete(handler); };
  }

  publish(): void {
    for (const follower of [...this.followers]) follower();
  }
}

class StoreProbe extends LightElement {
  static properties = {stores: {attribute: false}};
  declare stores: readonly FakeStore[];
  renders = 0;

  constructor() {
    super();
    this.stores = [];
    new StoreController(this, () => this.stores);
  }

  protected override render(): TemplateResult {
    this.renders += 1;
    return html``;
  }
}

customElements.define('dl-store-probe', StoreProbe);

function probe(...stores: FakeStore[]): StoreProbe {
  const element = document.createElement('dl-store-probe') as StoreProbe;
  element.stores = stores;
  return element;
}

afterEach(() => { document.body.replaceChildren(); });

describe('StoreController', () => {
  it('draws its host again when a store it follows changes, while the host is connected', async () => {
    const store = new FakeStore();
    const element = probe(store);
    expect(store.followers.size, 'a host that is not connected follows nothing').to.equal(0);

    document.body.append(element);
    await element.updateComplete;
    expect(store.followers.size).to.equal(1);
    expect(element.renders).to.equal(1);

    store.publish();
    await element.updateComplete;
    expect(element.renders).to.equal(2);

    element.remove();
    expect(store.followers.size, 'a host that left follows nothing').to.equal(0);
    store.publish();
    expect(element.renders).to.equal(2);

    document.body.append(element);
    await element.updateComplete;
    expect(store.followers.size, 'and follows again once it returns').to.equal(1);
  });

  it('follows the stores a host holds when it connects, however it came by them', async () => {
    const built = new FakeStore();
    const handed = new FakeStore();
    const element = probe(built);
    // The stores of a host are set by whoever composes it, after it is built and before it connects.
    element.stores = [handed];

    document.body.append(element);
    await element.updateComplete;

    expect(built.subscriptions, 'the store it was built with was followed').to.equal(0);
    expect(handed.followers.size).to.equal(1);
  });

  it('moves to the stores a connected host is given later, and lets the ones it had go', async () => {
    const first = new FakeStore();
    const second = new FakeStore();
    const element = probe(first);
    document.body.append(element);
    await element.updateComplete;

    element.stores = [second];
    await element.updateComplete;
    expect(first.followers.size, 'the stores it had were not let go').to.equal(0);
    expect(second.followers.size).to.equal(1);
    const drawn = element.renders;

    first.publish();
    await element.updateComplete;
    expect(element.renders, 'a store it let go still drew it').to.equal(drawn);
    second.publish();
    await element.updateComplete;
    expect(element.renders).to.equal(drawn + 1);
  });

  it('follows each store once however often the host draws, and follows a list as a list', async () => {
    const one = new FakeStore();
    const two = new FakeStore();
    const element = probe(one, two);
    document.body.append(element);
    await element.updateComplete;

    for (let draw = 0; draw < 3; draw += 1) {
      element.requestUpdate();
      await element.updateComplete;
    }
    expect([one.subscriptions, two.subscriptions]).to.deep.equal([1, 1]);

    // A list that changes in length is a different list.
    element.stores = [two];
    await element.updateComplete;
    expect(one.followers.size).to.equal(0);
    expect(two.followers.size, 'one follower for the store that stayed, not two').to.equal(1);
  });
});

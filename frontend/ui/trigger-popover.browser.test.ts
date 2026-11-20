// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The popover a trigger opens closes on Escape, leaves the Escape of an IME composition to the
 *  composition, and closes when its host stops showing the surface it belongs to. */

import {expect} from '@esm-bundle/chai';
import {html, nothing, type TemplateResult} from 'lit';
import {LightElement} from '../lib/lit-host.ts';
import {TriggerPopover} from '../lib/popover.ts';

class PopoverProbe extends LightElement {
  readonly popup = new TriggerPopover(this, {
    trigger: () => this.querySelector('button'),
    enter: () => { this.querySelector('input')?.focus(); },
  });

  protected override render(): TemplateResult {
    return html`<button type="button" @click=${this.popup.toggle}>Choose</button>
      ${this.popup.open ? html`<input aria-label="New name">` : nothing}`;
  }
}
customElements.define('dl-popover-probe', PopoverProbe);

/** A host whose surface goes away and comes back: the Files panel's workspace picker. */
class ShowingProbe extends LightElement {
  static properties = {
    shown: {state: true},
  };

  declare shown: boolean;

  readonly popup = new TriggerPopover(this, {
    trigger: () => this.querySelector('button'),
    enter: () => { this.querySelector('input')?.focus(); },
    showing: () => this.shown,
  });

  constructor() {
    super();
    this.shown = true;
  }

  protected override render(): TemplateResult {
    return html`<button type="button" @click=${this.popup.toggle}>Choose</button>
      ${this.popup.open ? html`<input aria-label="New name">` : nothing}`;
  }
}
customElements.define('dl-showing-probe', ShowingProbe);

afterEach(() => { document.body.replaceChildren(); });

async function openProbe(): Promise<{probe: PopoverProbe; trigger: HTMLButtonElement; input: HTMLInputElement}> {
  const probe = document.createElement('dl-popover-probe') as PopoverProbe;
  document.body.append(probe);
  await probe.updateComplete;
  const trigger = probe.querySelector('button')!;
  trigger.click();
  await probe.updateComplete;
  const input = probe.querySelector('input')!;
  await probe.updateComplete;
  expect(document.activeElement).to.equal(input);
  return {probe, trigger, input};
}

function pressEscape(target: HTMLElement, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true, ...init});
  target.dispatchEvent(event);
  return event;
}

it('closes on Escape from a field inside it and gives the trigger focus back', async () => {
  const {probe, trigger, input} = await openProbe();

  expect(pressEscape(input).defaultPrevented).to.equal(true);
  await probe.updateComplete;

  expect(probe.popup.open).to.equal(false);
  expect(document.activeElement).to.equal(trigger);
});

// Chromium marks the keys of a composition; WebKit sends them after it ended, as keyCode 229.
for (const [name, init] of [['isComposing', {isComposing: true}], ['keyCode 229', {keyCode: 229}]] as const) {
  it(`stays open for the Escape that cancels an IME composition (${name}), then closes on the next one`, async () => {
    const {probe, input} = await openProbe();
    const reached: string[] = [];
    input.addEventListener('keydown', () => { reached.push('field'); });

    const composing = pressEscape(input, init);
    await probe.updateComplete;

    expect(composing.defaultPrevented, 'the composition keeps its key').to.equal(false);
    expect(reached, 'the field still hears it').to.deep.equal(['field']);
    expect(probe.popup.open).to.equal(true);
    expect(probe.querySelector('input')).to.equal(input);

    expect(pressEscape(input).defaultPrevented).to.equal(true);
    await probe.updateComplete;
    expect(probe.popup.open).to.equal(false);
  });
}

it('closes with the surface that showed it, and comes back closed', async () => {
  const probe = document.createElement('dl-showing-probe') as ShowingProbe;
  document.body.append(probe);
  await probe.updateComplete;
  probe.querySelector('button')!.click();
  await probe.updateComplete;
  expect(probe.popup.open).to.equal(true);

  probe.shown = false;
  await probe.updateComplete;
  expect(probe.popup.open, 'a surface that is not shown has no open popover').to.equal(false);

  probe.shown = true;
  await probe.updateComplete;
  expect(probe.popup.open, 'the state is not kept for the surface to come back with').to.equal(false);
});

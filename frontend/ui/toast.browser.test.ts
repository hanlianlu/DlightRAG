// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys, sendMouse} from '@web/test-runner-commands';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {buttonNamed} from '../testing/dom.ts';
import {mountToast, withUndo} from '../testing/toast.ts';
import {type DlToastRegion, type ToastRequestDetail, toastRegionOf} from './toast.ts';
import './toast.ts';

defineDesignSystemElements();

const realSetTimeout = window.setTimeout;
const realClearTimeout = window.clearTimeout;
const realNow = performance.now.bind(performance);

interface Clock {
  /** The time left on each timer still waiting, soonest first. */
  waiting(): number[];
  /** Let this long pass: the timers that fall due run, in order. */
  advance(milliseconds: number): void;
}

/** Take the page's timers and clock into the test's hand: a toast's seconds pass when the test says so. */
function holdTime(): Clock {
  let now = 0;
  let nextId = 1;
  const timers = new Map<number, {due: number; handler: () => void}>();
  window.setTimeout = ((handler: () => void, delay = 0) => {
    timers.set(nextId, {due: now + delay, handler});
    return nextId++;
  }) as typeof window.setTimeout;
  window.clearTimeout = ((id: number) => { timers.delete(id); }) as typeof window.clearTimeout;
  performance.now = () => now;
  return {
    waiting: () => [...timers.values()].map((timer) => timer.due - now).sort((first, second) => first - second),
    advance(milliseconds) {
      now += milliseconds;
      const due = [...timers]
        .filter(([, timer]) => timer.due <= now)
        .sort(([, first], [, second]) => first.due - second.due);
      for (const [id, timer] of due) {
        if (!timers.delete(id)) continue;
        timer.handler();
      }
    },
  };
}

/** Everything queued behind the current task has run: promises settled, updates drawn. */
async function flush(): Promise<void> {
  await new Promise((resolve) => realSetTimeout.call(window, resolve, 0));
}

afterEach(() => {
  // The toasts go first: they clear their timers with the clock that set them.
  document.body.replaceChildren();
  window.setTimeout = realSetTimeout;
  window.clearTimeout = realClearTimeout;
  performance.now = realNow;
  document.documentElement.removeAttribute('data-color-mode');
});

describe('a toast', () => {
  for (const [kind, tone, duration] of [['a receipt', 'receipt', 1500], ['an error', 'error', 8000]] as const) {
    it(`stays ${duration} ms as ${kind}, waits for its Undo to answer, then keeps the answer as long`, async () => {
      const clock = holdTime();
      const toast = mountToast(tone);
      let answer: (words: string) => void = () => {};

      toast.show({message: 'Plain'});
      expect(clock.waiting()).to.deep.equal([duration]);
      toast.show(withUndo('Undo available', () => new Promise((resolve) => { answer = resolve; })));
      await toast.updateComplete;
      expect(clock.waiting(), 'a new toast starts the clock again').to.deep.equal([duration]);

      buttonNamed(toast, 'Undo')!.click();
      await toast.updateComplete;
      expect(clock.waiting(), 'nothing runs out while the Undo is out').to.deep.equal([]);
      answer('Undone.');
      await flush();
      expect(toast.textContent).to.contain('Undone.');
      expect(clock.waiting()).to.deep.equal([duration]);

      clock.advance(duration);
      await toast.updateComplete;
      expect(toast.visible).to.equal(false);
    });
  }

  it('puts focus on the action when the toast asks for it, and only then', async () => {
    const toast = mountToast();

    toast.show(withUndo('Plain change'));
    await toast.updateComplete;
    expect(document.activeElement).to.equal(document.body);

    toast.show(withUndo('Forgot: one', undefined, true));
    await toast.updateComplete;
    expect(document.activeElement).to.equal(toast.querySelector('.toast-action'));
    expect(toast.inert).to.equal(false);

    // A toast that replaced it first has nothing of the earlier one's to focus.
    (document.activeElement as HTMLElement).blur();
    toast.show(withUndo('Asked to focus', undefined, true));
    toast.show({message: 'Replaced at once'});
    await toast.updateComplete;
    expect(document.activeElement).to.equal(document.body);
  });

  it('renders escaped text and settles an asynchronous Undo in place', async () => {
    const toast = mountToast();
    let calls = 0;
    toast.show(withUndo('<img src=x> Remembered', async () => {
      calls += 1;
      return 'Profile Memory change undone.';
    }));
    await toast.updateComplete;

    expect(toast.querySelector('img')).to.equal(null);
    expect(toast.textContent).to.contain('<img src=x> Remembered');
    buttonNamed(toast, 'Undo')!.click();
    await flush();

    expect(calls).to.equal(1);
    expect(toast.textContent?.trim()).to.equal('Profile Memory change undone.');
    expect(toast.querySelector('button')).to.equal(null);
  });

  it('answers an Undo that supplies no words with a domain-neutral receipt', async () => {
    const toast = mountToast();
    toast.show(withUndo('First change'));
    await toast.updateComplete;
    buttonNamed(toast, 'Undo')!.click();
    await flush();

    expect(toast.textContent?.trim()).to.equal('Change undone.');
  });

  it('takes a failed Undo away and tells the failure as an error', async () => {
    const toast = mountToast();
    const asked: ToastRequestDetail[] = [];
    toast.addEventListener('dl-toast-request', (event) => { asked.push(event.detail); });
    toast.show(withUndo('Second change', async () => { throw new Error('conflict'); }));
    await toast.updateComplete;
    buttonNamed(toast, 'Undo')!.click();
    await flush();

    expect(asked).to.deep.equal([{message: 'Could not undo the change.', tone: 'error'}]);
    expect(toast.visible).to.equal(false);
  });

  it('replaces an actionable toast with a plain one without stale controls', async () => {
    const toast = mountToast();
    toast.show(withUndo('Remembered'));
    toast.show({message: 'Already remembered.'});
    await toast.updateComplete;

    expect(toast.textContent?.trim()).to.equal('Already remembered.');
    expect(toast.querySelector('button')).to.equal(null);
  });

  it('does not resume while hover ends but keyboard focus remains inside', async () => {
    const clock = holdTime();
    const toast = mountToast();
    toast.show(withUndo('Remembered'));
    await toast.updateComplete;
    const action = buttonNamed(toast, 'Undo')!;

    toast.dispatchEvent(new MouseEvent('mouseenter'));
    action.focus();
    toast.dispatchEvent(new MouseEvent('mouseleave'));
    expect(clock.waiting()).to.deep.equal([]);
    expect(document.activeElement).to.equal(action);

    action.blur();
    expect(clock.waiting()).to.deep.equal([1500]);
    clock.advance(1500);
    await toast.updateComplete;
    expect(toast.visible).to.equal(false);
  });

  it('pauses an actionable toast while Shell modality makes it unreachable', async () => {
    const clock = holdTime();
    const toast = mountToast();
    toast.show(withUndo('Remembered'));
    await toast.updateComplete;
    expect(toast.inert).to.equal(false);

    clock.advance(500);
    toast.shellInert = true;
    await toast.updateComplete;
    expect(toast.inert).to.equal(true);
    expect(clock.waiting(), 'the clock stops while nobody can reach Undo').to.deep.equal([]);
    clock.advance(60_000);
    expect(toast.visible).to.equal(true);

    toast.shellInert = false;
    await toast.updateComplete;
    expect(toast.inert).to.equal(false);
    expect(clock.waiting(), 'and goes on with the time it had left').to.deep.equal([1000]);

    clock.advance(1000);
    await toast.updateComplete;
    expect(toast.inert).to.equal(true);
    expect(toast.visible).to.equal(false);
  });

  it('keeps its words through the fade-out and drops them when the fade ends', async () => {
    // The fade-out lasts an hour: it ends where the test ends it, not when a clock says so.
    const style = document.createElement('style');
    style.textContent = '.toast { opacity: 0; } .toast.visible { opacity: 1; }'
      + ' .toast:not(.visible) { transition: opacity 3600s; }';
    document.body.append(style);
    const clock = holdTime();
    const toast = mountToast();
    toast.show(withUndo('Remembered'));
    await toast.updateComplete;
    // The page has drawn the toast showing, so hiding it is a change the browser fades.
    void getComputedStyle(toast).opacity;

    clock.advance(1500);
    await toast.updateComplete;
    const [fade] = toast.getAnimations();
    expect(fade, 'hiding the toast starts a fade').to.be.instanceOf(CSSTransition);
    // The toast has expired and is fading out: its words are still on it, and nobody can reach them.
    expect(toast.textContent).to.contain('Remembered');
    expect(toast.inert).to.equal(true);

    const faded = new Promise((resolve) => toast.addEventListener('transitionend', resolve, {once: true}));
    fade!.finish();
    await faded;
    await toast.updateComplete;
    expect(toast.textContent?.trim()).to.equal('');
    expect(toast.querySelector('button')).to.equal(null);
  });
});

describe('an error', () => {
  it('is announced at once from an alert region, and a receipt politely from a status region', () => {
    expect(mountToast('error').getAttribute('role')).to.equal('alert');
    expect(mountToast('receipt').getAttribute('role')).to.equal('status');
  });

  it('is dismissed by its close button, which a receipt does not have', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    const receipt = mountToast();
    error.show({message: 'Could not save.'});
    receipt.show({message: 'Saved.'});
    await Promise.all([error.updateComplete, receipt.updateComplete]);
    expect(buttonNamed(receipt, 'Dismiss')).to.equal(null);

    buttonNamed(error, 'Dismiss')!.click();
    await error.updateComplete;
    expect(error.visible).to.equal(false);
    expect(error.inert).to.equal(true);
    expect(clock.waiting(), 'only the receipt is still counting down').to.deep.equal([1500]);
  });

  it('goes by itself after eight seconds', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    error.show({message: 'Could not save.'});
    await error.updateComplete;

    clock.advance(7999);
    await error.updateComplete;
    expect(error.visible).to.equal(true);
    clock.advance(1);
    await error.updateComplete;
    expect(error.visible).to.equal(false);
  });

  it('waits while it is hovered or focused, and goes on with the time it had left', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    error.show({message: 'Could not save.'});
    await error.updateComplete;
    clock.advance(3000);

    error.dispatchEvent(new MouseEvent('mouseenter'));
    expect(clock.waiting(), 'hovered').to.deep.equal([]);
    error.dispatchEvent(new MouseEvent('mouseleave'));
    expect(clock.waiting()).to.deep.equal([5000]);

    buttonNamed(error, 'Dismiss')!.focus();
    expect(clock.waiting(), 'focused').to.deep.equal([]);
    (document.activeElement as HTMLElement).blur();
    expect(clock.waiting()).to.deep.equal([5000]);
  });

  it('waits out a Shell modal, so it never expires unseen, where a plain receipt passes', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    const receipt = mountToast();
    error.show({message: 'Could not save.'});
    receipt.show({message: 'Saved.'});
    await Promise.all([error.updateComplete, receipt.updateComplete]);

    clock.advance(1000);
    error.shellInert = true;
    receipt.shellInert = true;
    await Promise.all([error.updateComplete, receipt.updateComplete]);
    expect(error.inert, 'the modal stops its clock but does not put it out of reach').to.equal(false);
    expect(receipt.inert).to.equal(true);
    expect(clock.waiting(), 'the receipt goes on counting down; the error stops').to.deep.equal([500]);
    clock.advance(60_000);
    await Promise.all([error.updateComplete, receipt.updateComplete]);
    expect(receipt.visible, 'the receipt, a glance, ran out under the modal').to.equal(false);
    expect(error.visible, 'the error did not').to.equal(true);

    error.shellInert = false;
    await error.updateComplete;
    expect(clock.waiting(), 'and goes on with the time it had left').to.deep.equal([7000]);
    clock.advance(7000);
    await error.updateComplete;
    expect(error.visible).to.equal(false);
  });

  it('stays within reach under a Shell modal: an alert the page exposes, and a Dismiss that takes focus', async () => {
    const error = mountToast('error');
    const receipt = mountToast();
    error.shellInert = true;
    receipt.shellInert = true;
    error.show({message: 'Could not save.'});
    receipt.show(withUndo('Saved.'));
    await Promise.all([error.updateComplete, receipt.updateComplete]);

    // Inert content is out of the accessibility tree and the tab order; a live region in it is never announced.
    expect(error.getAttribute('role')).to.equal('alert');
    expect(error.closest('[inert]') === null, 'the alert is in the accessibility tree').to.equal(true);
    const dismiss = buttonNamed(error, 'Dismiss')!;
    dismiss.focus();
    expect(document.activeElement === dismiss, 'and its Dismiss can be reached').to.equal(true);

    buttonNamed(receipt, 'Undo')!.focus();
    expect(document.activeElement === dismiss, 'a receipt is out of reach under the same modal').to.equal(true);
  });

  it('opens under a Shell modal without a clock, and starts one when the modal goes', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    error.shellInert = true;
    error.show({message: 'Could not save.'});
    await error.updateComplete;
    expect(clock.waiting()).to.deep.equal([]);

    error.shellInert = false;
    await error.updateComplete;
    expect(clock.waiting()).to.deep.equal([8000]);
  });

  it('keeps its Undo working as a receipt\'s does', async () => {
    const clock = holdTime();
    const error = mountToast('error');
    error.show(withUndo('Could not finish.', async () => 'Finished.'));
    await error.updateComplete;

    buttonNamed(error, 'Undo')!.click();
    await flush();
    expect(error.textContent).to.contain('Finished.');
    expect(clock.waiting()).to.deep.equal([8000]);
  });

  it('survives a receipt that follows it, and is replaced by the next error', async () => {
    const receipt = mountToast();
    const error = mountToast('error');
    // As a host does: the request goes to the region of its tone.
    const show = (request: ToastRequestDetail): void => { toastRegionOf(document.body, request.tone)!.show(request); };

    show({message: 'Could not save.', tone: 'error'});
    show({message: 'Saved elsewhere.'});
    await Promise.all([error.updateComplete, receipt.updateComplete]);
    expect(error.visible).to.equal(true);
    expect(error.textContent).to.contain('Could not save.');
    expect(receipt.textContent).to.contain('Saved elsewhere.');

    show({message: 'Could not delete.', tone: 'error'});
    await error.updateComplete;
    expect(error.textContent).to.contain('Could not delete.');
    expect(error.textContent).not.to.contain('Could not save.');
    expect(receipt.textContent).to.contain('Saved elsewhere.');
  });
});

describe('an error dismissed from the keyboard', () => {
  /** A stop before the toast in the tab order: the place the reader was working. */
  function placeBeforeToast(): HTMLButtonElement {
    const place = document.createElement('button');
    place.textContent = 'Place';
    document.body.prepend(place);
    return place;
  }

  /** Mount an error, and count the times it asks the Shell to place focus. */
  async function mountedError(): Promise<{error: DlToastRegion; lost: () => number}> {
    const error = mountToast('error');
    let lost = 0;
    error.addEventListener('dl-toast-focus-lost', () => { lost += 1; });
    error.show({message: 'Could not save.'});
    await error.updateComplete;
    return {error, lost: () => lost};
  }

  it('gives focus back to where it was before it came into the toast', async () => {
    const place = placeBeforeToast();
    const {error, lost} = await mountedError();
    place.focus();

    await sendKeys({press: 'Tab'});
    expect(error.contains(document.activeElement), 'Tab reached the Dismiss').to.equal(true);
    await sendKeys({press: 'Enter'});
    await error.updateComplete;

    expect(error.visible).to.equal(false);
    expect(document.activeElement === place, 'focus is back on the place').to.equal(true);
    expect(lost(), 'nobody was asked').to.equal(0);
  });

  it('asks the Shell to place focus when the place it came from is gone', async () => {
    const place = placeBeforeToast();
    const {error, lost} = await mountedError();
    place.focus();
    await sendKeys({press: 'Tab'});

    place.remove();
    await sendKeys({press: 'Enter'});
    await error.updateComplete;

    expect(error.visible).to.equal(false);
    expect(lost()).to.equal(1);
  });

  it('asks the Shell too when focus came from outside the page, so there is no place to return it to', async () => {
    const {error, lost} = await mountedError();
    buttonNamed(error, 'Dismiss')!.focus();

    await sendKeys({press: 'Enter'});

    expect(lost()).to.equal(1);
  });

  it('remembers the place it came from, and not the Undo it moved on from', async () => {
    const place = placeBeforeToast();
    const error = mountToast('error');
    error.show(withUndo('Could not finish.'));
    await error.updateComplete;
    place.focus();
    await sendKeys({press: 'Tab'});
    expect(document.activeElement === buttonNamed(error, 'Undo'), 'the Undo comes first').to.equal(true);
    await sendKeys({press: 'Tab'});
    await sendKeys({press: 'Enter'});
    await error.updateComplete;

    expect(error.visible).to.equal(false);
    expect(document.activeElement === place, 'focus is back on the place').to.equal(true);
  });

  it('leaves a pointer alone: a click on Dismiss gives nothing back and asks nobody', async () => {
    const place = placeBeforeToast();
    const {error, lost} = await mountedError();
    place.focus();

    const box = buttonNamed(error, 'Dismiss')!.getBoundingClientRect();
    await sendMouse({type: 'click', position: [Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2)]});
    await error.updateComplete;

    expect(error.visible).to.equal(false);
    expect(lost()).to.equal(0);
    expect(document.activeElement === place, 'focus was not taken back to the place').to.equal(false);
  });
});

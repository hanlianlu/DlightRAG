// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The unsent message outlives the page: a provider return, a refresh, or a crash keeps it. The keys that send,
 *  break the line, stop a Run, and pick a skill answer to the reader, and not to an IME composition. */

import {expect} from '@esm-bundle/chai';
import {sendKeys} from '@web/test-runner-commands';
import {AGENT_EFFORT_STORAGE_KEY} from '../lib/agent-effort.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {policy, restoreChatFeatureFixtures} from '../testing/chat-feature.ts';
import {waitFor, chooseFiles} from '../testing/dom.ts';
import './chat-composer.ts';
import './chat-feature.ts';
import {forgetStoredDraft, type ComposerSubmitDetail, type DlChatComposer} from './chat-composer.ts';

restoreChatFeatureFixtures();

const DRAFT_KEY = 'dlightrag.composerDraft';

const originalFetch = window.fetch;

function mount(): DlChatComposer {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  document.body.appendChild(composer);
  return composer;
}

/** Type into the box the way a reader does, so the composer's own handler runs. */
async function type(composer: DlChatComposer, text: string): Promise<void> {
  await composer.updateComplete;
  const input = composer.querySelector<HTMLTextAreaElement>('.composer-input')!;
  input.value = text;
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
}

function boxOf(composer: DlChatComposer): string {
  return composer.querySelector<HTMLTextAreaElement>('.composer-input')!.value;
}

/** What the page does when the browser takes it away: the message is written before it goes. */
function pageGoesAway(): void {
  window.dispatchEvent(new Event('pagehide'));
}

const settle = (): Promise<void> => new Promise((resolve) => { setTimeout(resolve, 500); });

/** Every message the composer offers the page, in the order it was offered. */
function offered(composer: DlChatComposer): ComposerSubmitDetail[] {
  const messages: ComposerSubmitDetail[] = [];
  composer.addEventListener('dl-composer-submit', (event) => { messages.push(event.detail); });
  return messages;
}

/** The composer's box, focused, so the keys a test presses are the reader's. */
async function focused(composer: DlChatComposer): Promise<HTMLTextAreaElement> {
  await composer.updateComplete;
  const input = composer.querySelector<HTMLTextAreaElement>('.composer-input')!;
  input.focus();
  return input;
}

/** One keydown, as a browser delivers it: `init` says what an IME composition makes of it. */
function press(input: HTMLElement, key: string, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true, ...init});
  input.dispatchEvent(event);
  return event;
}

/** Chromium marks the keys of a composition; WebKit delivers them after it ended, as keyCode 229. */
const COMPOSING: readonly KeyboardEventInit[] = [{isComposing: true}, {keyCode: 229}];

const SKILLS = [
  {name: 'tdd', description: 'TDD loop', source: 'global'},
  {name: 'weekly-report', description: 'Weekly reports', source: 'owner'},
  {name: 'z-creator', description: 'Create skills', source: 'builtin'},
];

/** The skill catalog the server states, for the menu that opens on a slash. */
function serveSkills(skills: readonly object[] = SKILLS): void {
  window.fetch = async () => new Response(JSON.stringify({skills}), {
    status: 200,
    headers: {'Content-Type': 'application/json'},
  });
}

/** Type a slash into the box and wait for the menu to list `count` skills. */
async function openSkillMenu(composer: DlChatComposer, count = SKILLS.length): Promise<HTMLTextAreaElement> {
  const input = await focused(composer);
  await type(composer, '/');
  await waitFor(() => composer.querySelectorAll('.skill-menu-item').length === count);
  await composer.updateComplete;
  return input;
}

beforeEach(() => {
  localStorage.removeItem(DRAFT_KEY);
});

afterEach(() => {
  window.fetch = originalFetch;
  localStorage.removeItem(DRAFT_KEY);
  document.body.replaceChildren();
});

it('brings the unsent message back after a reload', async () => {
  const composer = mount();
  await type(composer, 'a question I had not sent');
  await settle();
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('a question I had not sent');

  // A reload builds the element again; what the reader typed is in the box.
  composer.remove();
  const reloaded = mount();
  await reloaded.updateComplete;
  expect(boxOf(reloaded)).to.equal('a question I had not sent');
});

it('writes the message the moment the page goes away, before typing settles', async () => {
  const composer = mount();
  await type(composer, 'half a thought, then the provider page');

  pageGoesAway();

  expect(localStorage.getItem(DRAFT_KEY)).to.equal('half a thought, then the provider page');
});

it('forgets the message when the box is cleared, and a reload comes back empty', async () => {
  const composer = mount();
  await type(composer, 'gone in a moment');
  pageGoesAway();

  composer.clearText();

  expect(localStorage.getItem(DRAFT_KEY)).to.equal(null);
  composer.remove();
  const reloaded = mount();
  await reloaded.updateComplete;
  expect(boxOf(reloaded)).to.equal('');
});

it('forgets the message when a sent one is cleared from the box', async () => {
  const composer = mount();
  await type(composer, 'accepted question');
  pageGoesAway();

  expect(composer.clearSubmittedText('accepted question')).to.equal(true);

  expect(localStorage.getItem(DRAFT_KEY)).to.equal(null);
});

it('does not hand back a message the reader sent', async () => {
  const composer = mount();
  await type(composer, 'this one goes to the agent');
  // Typing settles first, as it does when a reader pauses before sending.
  await settle();
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('this one goes to the agent');

  composer.querySelector<HTMLFormElement>('#query-form')!.requestSubmit();
  await composer.updateComplete;

  expect(boxOf(composer)).to.equal('');
  expect(localStorage.getItem(DRAFT_KEY)).to.equal(null);
  composer.remove();
  const reloaded = mount();
  await reloaded.updateComplete;
  expect(boxOf(reloaded)).to.equal('');
});

it('forgets the message when the sign-in ends', async () => {
  const composer = mount();
  await type(composer, 'words from a session that ended');
  pageGoesAway();

  forgetStoredDraft();

  expect(localStorage.getItem(DRAFT_KEY)).to.equal(null);
});

it('offers the message in a cancelable event, and clears the box once a listener has taken it', async () => {
  const composer = mount();
  let offer: CustomEvent | null = null;
  composer.addEventListener('dl-composer-submit', (event) => { offer = event; });
  await type(composer, 'taken');
  pageGoesAway();
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('taken');

  composer.querySelector<HTMLFormElement>('#query-form')!.requestSubmit();
  await composer.updateComplete;

  expect(offer).not.to.equal(null);
  expect(offer!.cancelable, 'a listener may refuse the message').to.equal(true);
  expect(offer!.bubbles && offer!.composed, 'and so may one above the composer').to.equal(true);
  expect(boxOf(composer)).to.equal('');
  expect(localStorage.getItem(DRAFT_KEY)).to.equal(null);
});

it('keeps the words in the box, and in storage, when a listener refuses the message', async () => {
  const composer = mount();
  let refusals = 0;
  composer.addEventListener('dl-composer-submit', (event) => {
    refusals += 1;
    event.preventDefault();
  });
  await type(composer, 'not just yet');
  // The words are stored before the refusal and must still be after it.
  pageGoesAway();
  expect(localStorage.getItem(DRAFT_KEY)).to.equal('not just yet');

  composer.querySelector<HTMLFormElement>('#query-form')!.requestSubmit();
  await composer.updateComplete;
  await settle();

  expect(refusals).to.equal(1);
  expect(boxOf(composer)).to.equal('not just yet');
  expect(composer.hasDraft).to.equal(true);
  expect(localStorage.getItem(DRAFT_KEY), 'a reload still brings the words back').to.equal('not just yet');
  composer.remove();
  const reloaded = mount();
  await reloaded.updateComplete;
  expect(boxOf(reloaded)).to.equal('not just yet');
});

it('keeps a skill directive in the box too when its message is refused, and sends it once it is taken', async () => {
  const composer = mount();
  const messages = offered(composer);
  let refuse = true;
  composer.addEventListener('dl-composer-submit', (event) => { if (refuse) event.preventDefault(); });
  await type(composer, '/skill:tdd write the test first');

  composer.querySelector<HTMLFormElement>('#query-form')!.requestSubmit();
  await composer.updateComplete;
  expect(boxOf(composer)).to.equal('/skill:tdd write the test first');

  refuse = false;
  composer.querySelector<HTMLFormElement>('#query-form')!.requestSubmit();
  await composer.updateComplete;
  expect(messages.map(({query, requestedSkill}) => [query, requestedSkill])).to.deep.equal([
    ['write the test first', 'tdd'],
    ['write the test first', 'tdd'],
  ]);
  expect(boxOf(composer)).to.equal('');
});

it('sends on Enter, breaks the line on Shift+Enter, and never sends on the Enter of an IME composition', async () => {
  const composer = mount();
  const messages = offered(composer);
  const input = await focused(composer);

  await sendKeys({type: 'first'});
  await sendKeys({press: 'Shift+Enter'});
  await sendKeys({type: 'second'});
  expect(input.value).to.equal('first\nsecond');
  expect(messages).to.deep.equal([]);

  // The Enter that commits a composition is the composition's.
  const composing = new InputEvent('beforeinput', {
    inputType: 'insertLineBreak', isComposing: true, bubbles: true, cancelable: true,
  });
  input.dispatchEvent(composing);
  expect(composing.defaultPrevented).to.equal(false);
  expect(messages).to.deep.equal([]);

  await sendKeys({press: 'Enter'});
  await composer.updateComplete;
  expect(messages.map(({query}) => query)).to.deep.equal(['first\nsecond']);
  expect(input.value).to.equal('');
});

it('stops a Run on Escape, but not on the Escape that cancels an IME composition', async () => {
  const composer = mount();
  composer.running = true;
  const input = await focused(composer);
  let stops = 0;
  composer.addEventListener('dl-composer-cancel', () => { stops += 1; });

  for (const init of COMPOSING) press(input, 'Escape', init);
  expect(stops, 'the Escape that closes a candidate window leaves the Run running').to.equal(0);

  press(input, 'Escape');
  expect(stops).to.equal(1);
});

it('leaves the keys of an IME composition alone while the skill menu is open', async () => {
  serveSkills();
  const composer = mount();
  const input = await openSkillMenu(composer);
  const menu = composer.querySelector<HTMLElement>('#skill-menu')!;
  const active = (): string | null => composer.querySelector('.skill-menu-item.active')
    ?.querySelector('.skill-menu-name')?.textContent ?? null;

  expect(press(input, 'ArrowDown').defaultPrevented, 'the reader\'s own arrow moves through the menu').to.equal(true);
  await composer.updateComplete;
  expect(active()).to.equal('tdd');

  for (const init of COMPOSING) {
    for (const key of ['ArrowDown', 'ArrowUp', 'Tab', 'Enter', 'Escape']) {
      expect(press(input, key, init).defaultPrevented, `${key} ${JSON.stringify(init)} is not taken`).to.equal(false);
    }
    await composer.updateComplete;
    expect(active(), 'the highlight stayed where it was').to.equal('tdd');
    expect(input.value, 'no skill was installed in the box').to.equal('/');
    expect(menu.hidden, 'the menu stayed open').to.equal(false);
  }

  // Once the composition is over, the same keys do their work: Escape closes the menu ...
  press(input, 'Escape');
  await composer.updateComplete;
  expect(menu.hidden).to.equal(true);
  // ... and Enter picks the highlighted skill.
  await type(composer, '/');
  press(input, 'ArrowDown');
  expect(press(input, 'Enter').defaultPrevented).to.equal(true);
  await composer.updateComplete;
  expect(input.value).to.equal('/skill:tdd ');
});

describe('the skill menu, to assistive technology', () => {
  const optionId = (composer: DlChatComposer, name: string): string => composer
    .querySelector(`.skill-menu-item[id="skill-option-${name}"]`)!.id;
  const commandOptionId = (composer: DlChatComposer, name: string): string => composer
    .querySelector(`.command-menu-item[id="command-option-${name}"]`)!.id;

  it('points the box at the menu and at the highlighted skill, with no combobox role the box may not take', async () => {
    serveSkills();
    const composer = mount();
    const input = await focused(composer);

    // A textarea may not take role=combobox or aria-expanded, so it states the list it autocompletes from
    // and where its highlight is, and nothing more.
    expect(input.getAttribute('aria-autocomplete')).to.equal('list');
    expect(input.hasAttribute('role')).to.equal(false);
    expect(input.hasAttribute('aria-expanded')).to.equal(false);
    expect(input.hasAttribute('aria-controls'), 'nothing to control while the menu is closed').to.equal(false);
    expect(input.hasAttribute('aria-activedescendant')).to.equal(false);

    await openSkillMenu(composer);
    const menu = composer.querySelector<HTMLElement>('#skill-menu')!;
    expect(menu.getAttribute('role')).to.equal('listbox');
    expect(input.getAttribute('aria-controls')).to.equal('skill-menu');
    expect(input.hasAttribute('aria-activedescendant'), 'no skill is highlighted yet').to.equal(false);
    // Every option has an id of its own, the same one each time the menu draws.
    const ids = [...composer.querySelectorAll('.skill-menu-item')].map((option) => option.id);
    expect(ids).to.deep.equal(['skill-option-tdd', 'skill-option-weekly-report', 'skill-option-z-creator']);

    const highlighted = (): string | null => input.getAttribute('aria-activedescendant');
    await sendKeys({press: 'ArrowDown'});
    await composer.updateComplete;
    expect(highlighted()).to.equal(optionId(composer, 'tdd'));
    expect(composer.querySelector(`#${highlighted()}`)?.getAttribute('aria-selected')).to.equal('true');
    await sendKeys({press: 'ArrowDown'});
    await composer.updateComplete;
    expect(highlighted()).to.equal(optionId(composer, 'weekly-report'));
    expect(composer.querySelector('#skill-option-tdd')?.getAttribute('aria-selected')).to.equal('false');
    await sendKeys({press: 'ArrowUp'});
    await composer.updateComplete;
    expect(highlighted()).to.equal(optionId(composer, 'tdd'));
    // ArrowUp wraps from the first to the last.
    await sendKeys({press: 'ArrowUp'});
    await composer.updateComplete;
    expect(highlighted()).to.equal(commandOptionId(composer, 'goal'));

    // Escape closes the menu, and the box stops pointing at what is gone.
    await sendKeys({press: 'Escape'});
    await composer.updateComplete;
    expect(menu.hidden).to.equal(true);
    expect(input.hasAttribute('aria-controls')).to.equal(false);
    expect(input.hasAttribute('aria-activedescendant')).to.equal(false);
  });

  it('stops pointing at the menu once a skill is picked, and when no skill matches what is typed', async () => {
    serveSkills();
    const composer = mount();
    const input = await openSkillMenu(composer);

    await sendKeys({press: 'ArrowDown'});
    await composer.updateComplete;
    expect(input.getAttribute('aria-activedescendant')).to.equal('skill-option-tdd');
    await sendKeys({press: 'Tab'});
    await composer.updateComplete;
    expect(input.value).to.equal('/skill:tdd ');
    expect(input.hasAttribute('aria-controls')).to.equal(false);
    expect(input.hasAttribute('aria-activedescendant')).to.equal(false);

    await type(composer, '/skill:q');
    await composer.updateComplete;
    expect(composer.querySelectorAll('.skill-menu-item')).to.have.length(0);
    expect(input.hasAttribute('aria-controls'), 'a menu with nothing in it is not pointed at').to.equal(false);
    expect(input.hasAttribute('aria-activedescendant')).to.equal(false);
  });

  it('says how many skills match when the menu opens and when that number changes, and not for every key', async () => {
    serveSkills();
    const composer = mount();
    await composer.updateComplete;
    const status = composer.querySelector<HTMLElement>('[data-skill-status]')!;
    expect(status.getAttribute('role')).to.equal('status');
    expect(status.getAttribute('aria-live')).to.equal('polite');
    expect(status.textContent?.trim(), 'a closed menu says nothing').to.equal('');

    // Whatever the status is told changes in the page, a screen reader may read out.
    const changes: string[] = [];
    const watch = new MutationObserver(() => { changes.push(status.textContent?.trim() ?? ''); });
    watch.observe(status, {childList: true, characterData: true, subtree: true});
    const settleChanges = async (): Promise<string[]> => {
      await composer.updateComplete;
      await new Promise((resolve) => setTimeout(resolve, 0));
      const seen = [...changes];
      watch.takeRecords();
      changes.length = 0;
      return seen;
    };

    await openSkillMenu(composer);
    expect(status.textContent?.trim()).to.equal('3 skills available');
    expect(await settleChanges()).to.deep.equal(['3 skills available']);

    // `/s` and `/sk` are still every skill: the same words are not said again.
    await type(composer, '/s');
    await type(composer, '/sk');
    expect(composer.querySelectorAll('.skill-menu-item')).to.have.length(3);
    expect(await settleChanges(), 'the same count is not announced again').to.deep.equal([]);

    await type(composer, '/skill:w');
    expect(composer.querySelectorAll('.skill-menu-item')).to.have.length(1);
    expect(status.textContent?.trim()).to.equal('1 skill available');
    expect(await settleChanges()).to.deep.equal(['1 skill available']);

    await type(composer, '/skill:');
    expect(status.textContent?.trim()).to.equal('3 skills available');
    expect(await settleChanges()).to.deep.equal(['3 skills available']);

    // No match, or no menu: nothing is said, and the status is empty for the next time the menu opens.
    await type(composer, '/skill:q');
    expect(status.textContent?.trim()).to.equal('');
    await type(composer, 'a question');
    expect(status.textContent?.trim()).to.equal('');
    await settleChanges();
    await type(composer, '/');
    expect(await settleChanges(), 'the menu opens again, and says so').to.deep.equal(['3 skills available']);
    watch.disconnect();
  });

  it('says the count in the language the reader chose', async () => {
    serveSkills();
    const composer = mount();
    await openSkillMenu(composer);
    const status = composer.querySelector<HTMLElement>('[data-skill-status]')!;
    try {
      await setLanguagePreference('zh');
      await composer.updateComplete;
      expect(status.textContent?.trim()).to.equal('3 个技能可用');
      await type(composer, '/skill:w');
      expect(status.textContent?.trim()).to.equal('1 个技能可用');
    } finally {
      await setLanguagePreference('en');
    }
  });
});

function effortRow(composer: DlChatComposer, level: string): HTMLButtonElement {
  return composer.querySelector<HTMLButtonElement>(`[data-effort="${level}"]`)!;
}

it('Composer owns draft, attachment, mode, and typed submission intent', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.attachmentAccept = 'image/*,.md,.pdf';
  document.body.appendChild(composer);
  await composer.updateComplete;

  chooseFiles(composer, [new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  await composer.updateComplete;
  expect(composer.hasDraft).to.equal(true);
  expect(composer.textContent).to.contain('notes.md');

  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = '  Explain this  ';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;

  let submitted: {query: string; mode: string | null; effort?: string | null} | null = null;
  composer.addEventListener('dl-composer-submit', (event) => {
    submitted = (event as CustomEvent).detail;
  });
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await composer.updateComplete;

  expect(submitted).to.deep.equal({
    query: 'Explain this',
    mode: null,
    requestedSkill: null,
    effort: null,
  });
  expect(composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')?.value).to.equal('');
  expect(composer.hasDraft).to.equal(true);

  composer.clearDraft();
  await composer.updateComplete;
  expect(composer.hasDraft).to.equal(false);
  composer.focusInput();
  await composer.updateComplete;
  expect(document.activeElement).to.equal(composer.querySelector('[aria-label="Message"]'));
});

it('Composer leaves /goal to the chat instead of reading it as a Skill', async () => {
  const composer = mount();
  await composer.updateComplete;
  await type(composer, '/goal finish the sealed model');

  let submitted: ComposerSubmitDetail | null = null;
  composer.addEventListener('dl-composer-submit', (event) => {
    submitted = (event as CustomEvent<ComposerSubmitDetail>).detail;
  });
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await composer.updateComplete;

  expect(submitted).to.deep.equal({
    query: '/goal finish the sealed model',
    mode: null,
    requestedSkill: null,
    effort: null,
  });
});

it('Composer sends /goal to the chat as a command while a Run is live, never as delivery', async () => {
  const composer = mount();
  composer.running = true;
  composer.live = true;
  const submitted: ComposerSubmitDetail[] = [];
  composer.addEventListener('dl-composer-submit', (event) => {
    submitted.push((event as CustomEvent<ComposerSubmitDetail>).detail);
  });
  await composer.updateComplete;
  await type(composer, '/goal pause');
  composer.querySelector<HTMLButtonElement>('[aria-label="Queue"]')?.click();
  await composer.updateComplete;

  expect(submitted).to.deep.equal([{
    query: '/goal pause',
    mode: null,
    requestedSkill: null,
    effort: null,
  }]);
  expect(boxOf(composer)).to.equal('');
});

it('Composer lists the /goal command, and keeps its hint while the reader types arguments', async () => {
  const composer = mount();
  await type(composer, '/go');
  const row = composer.querySelector('.command-menu-item');
  expect(row?.textContent).to.contain('/goal');
  expect(row?.textContent).to.contain('Set or view the goal for a long-running task');

  row?.dispatchEvent(new MouseEvent('click', {bubbles: true}));
  await composer.updateComplete;
  expect(boxOf(composer)).to.equal('/goal ');

  await type(composer, '/goal cap');
  expect(composer.querySelector('.command-hint')?.textContent)
    .to.contain('[<objective>|cap N|edit <objective>|pause|resume|clear]');
  expect(composer.querySelector('.command-menu-item')).to.equal(null);

  await type(composer, '/goalkeeper');
  expect(composer.querySelector('.command-hint')).to.equal(null);
  expect(composer.querySelector('.command-menu-item')).to.equal(null);
});

it('Composer delivers ordinary text while a Run is live: queue by default, steer on request', async () => {
  const composer = mount();
  composer.running = true;
  composer.live = true;
  const submitted: ComposerSubmitDetail[] = [];
  composer.addEventListener('dl-composer-submit', (event) => {
    submitted.push((event as CustomEvent<ComposerSubmitDetail>).detail);
  });
  await composer.updateComplete;
  await type(composer, 'keep going');
  composer.querySelector<HTMLButtonElement>('[aria-label="Queue"]')?.click();
  await composer.updateComplete;

  await type(composer, 'change the wording');
  // One quiet button states the delivery and switches it; no pair of chips sits in the row.
  expect(composer.querySelectorAll('.composer-delivery').length).to.equal(1);
  expect(composer.querySelector('.composer-delivery-option')).to.equal(null);
  composer.querySelector<HTMLButtonElement>('.composer-delivery')?.click();
  await composer.updateComplete;
  expect(composer.querySelector('.composer-delivery')?.textContent).to.contain('Steer');
  composer.querySelector<HTMLButtonElement>('[aria-label="Steer"]')?.click();
  await composer.updateComplete;

  expect(submitted).to.deep.equal([
    {
      query: 'keep going',
      mode: null,
      requestedSkill: null,
      effort: null,
      delivery: 'queue',
    },
    {
      query: 'change the wording',
      mode: null,
      requestedSkill: null,
      effort: null,
      delivery: 'steer',
    },
  ]);
});

it('Composer shows the delivery switch only while a Run is live and the reader is writing', async () => {
  const composer = mount();
  composer.running = true;
  composer.live = true;
  await composer.updateComplete;

  // Nothing typed: the row keeps no delivery control at all.
  expect(composer.querySelector('.composer-delivery')).to.equal(null);

  await type(composer, 'queued words');
  expect(composer.querySelectorAll('.composer-delivery').length).to.equal(1);

  // Words but no live Run is an ordinary send: there is no queue-or-steer choice to make.
  composer.live = false;
  composer.running = false;
  await composer.updateComplete;
  expect(composer.querySelector('.composer-delivery')).to.equal(null);

  // A live Run with words is the one state that carries the switch.
  composer.live = true;
  composer.running = true;
  await composer.updateComplete;
  expect(composer.querySelectorAll('.composer-delivery').length).to.equal(1);

  composer.live = false;
  composer.running = false;
  await composer.clearText();
  await composer.updateComplete;
  expect(composer.querySelector('.composer-delivery')).to.equal(null);
});

it('Composer hears the page only while it is connected, each time it connects', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  document.body.appendChild(composer);
  await composer.updateComplete;
  // The first connection's listeners ended with it, so a second connection must bind its own.
  composer.remove();
  document.body.appendChild(composer);
  await composer.updateComplete;

  const overlay = composer.querySelector('.drop-overlay')!;
  const files = new DataTransfer();
  files.items.add(new File(['notes'], 'notes.md', {type: 'text/markdown'}));
  document.dispatchEvent(new DragEvent('dragenter', {dataTransfer: files, cancelable: true}));
  await composer.updateComplete;
  expect(overlay.classList.contains('active')).to.equal(true);
  document.dispatchEvent(new DragEvent('drop', {dataTransfer: files, cancelable: true}));
  await composer.updateComplete;
  expect(overlay.classList.contains('active')).to.equal(false);

  const paste = new Event('paste');
  const clipboard = new DataTransfer();
  clipboard.items.add(new File(['png'], 'shot.png', {type: 'image/png'}));
  Object.defineProperty(paste, 'clipboardData', {value: clipboard});
  document.dispatchEvent(paste);
  await composer.updateComplete;
  expect(composer.querySelector('.thumbnail-strip img')?.getAttribute('alt')).to.equal('shot.png');

  const menu = composer.querySelector('#composer-mode-menu')!;
  composer.querySelector<HTMLButtonElement>('.composer-mode-trigger')!.click();
  await composer.updateComplete;
  expect(menu.hasAttribute('hidden')).to.equal(false);
  document.body.click();
  await composer.updateComplete;
  expect(menu.hasAttribute('hidden')).to.equal(true);

  composer.remove();
  document.dispatchEvent(new DragEvent('dragenter', {dataTransfer: files, cancelable: true}));
  await composer.updateComplete;
  expect(overlay.classList.contains('active')).to.equal(false);
});

it('Composer offers only bootstrap levels, remembers the choice, and hides effort for fast', async () => {
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
  document.body.appendChild(composer);
  await composer.updateComplete;

  // The deployment default is shown, checked, and marked without being stored.
  expect(composer.querySelector('.composer-effort-label')?.textContent).to.equal('High');
  expect(localStorage.getItem(AGENT_EFFORT_STORAGE_KEY)).to.equal(null);
  const rows = [...composer.querySelectorAll<HTMLButtonElement>('[data-effort]')];
  expect(rows.map((row) => row.dataset.effort)).to.deep.equal(['low', 'high', 'max']);
  expect(rows.map((row) => row.getAttribute('aria-checked'))).to.deep.equal(['false', 'true', 'false']);
  expect(rows[1]?.querySelector('.composer-effort-default')?.textContent).to.equal('Default');

  // ArrowUp opens on the last level, Escape closes and returns focus to the trigger.
  const trigger = composer.querySelector<HTMLButtonElement>('.composer-effort-trigger')!;
  trigger.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowUp', bubbles: true}));
  await composer.updateComplete;
  expect(document.activeElement).to.equal(rows[2]);
  rows[2]!.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  await composer.updateComplete;
  expect(composer.querySelector('.composer-effort-menu')?.hasAttribute('hidden')).to.equal(true);
  expect(document.activeElement).to.equal(trigger);

  // Each level shows its own dial: the trigger's needle follows the choice, and
  // every menu row carries that level's stop rather than one shared glyph.
  const needle = () => composer.querySelector('.composer-effort-trigger path:last-of-type')?.getAttribute('d');
  const rowNeedles = () => [...composer.querySelectorAll<HTMLButtonElement>('[data-effort]')]
    .map((row) => row.querySelector('path:last-of-type')?.getAttribute('d'));
  const stops = rowNeedles();
  expect(new Set(stops).size).to.equal(3, 'each level owns a distinct needle stop');
  expect(needle()).to.equal(stops[1], 'the deployment default is shown at its own stop');

  // Choosing a level stores the caller's own override and carries it in the intent.
  const intents: {effort: string | null}[] = [];
  composer.addEventListener('dl-composer-submit', (event) => {
    intents.push((event as CustomEvent<{effort: string | null}>).detail);
  });
  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'Compare both';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  effortRow(composer, 'max').click();
  await composer.updateComplete;
  expect(localStorage.getItem(AGENT_EFFORT_STORAGE_KEY)).to.equal('max');
  expect(composer.querySelector('.composer-effort-label')?.textContent).to.equal('Max');
  expect(needle()).to.equal(stops[2], 'the trigger shows the chosen level stop');
  expect(needle()).to.not.equal(stops[0]);
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await composer.updateComplete;
  expect(intents.at(-1)?.effort).to.equal('max');

  // Fast answers run no agent turns: the control disappears and sends no effort.
  composer.querySelector<HTMLButtonElement>('[data-mode="fast"]')!.click();
  await composer.updateComplete;
  expect(composer.querySelector('.composer-effort')).to.equal(null);
  input.value = 'Quick one';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  composer.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await composer.updateComplete;
  expect(intents.at(-1)?.effort).to.equal(null);
  expect(localStorage.getItem(AGENT_EFFORT_STORAGE_KEY)).to.equal('max');

  // Returning to an agent mode restores the stored choice.
  composer.querySelector<HTMLButtonElement>('[data-mode="auto"]')!.click();
  await composer.updateComplete;
  expect(composer.querySelector('.composer-effort-label')?.textContent).to.equal('Max');

  composer.remove();
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
});

it('Composer pickers share one keyboard contract for their menus', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
  document.body.appendChild(composer);
  await composer.updateComplete;

  // Keys come from the focused row, exactly as a keyboard user produces them.
  const key = async (name: string): Promise<void> => {
    (document.activeElement as HTMLElement).dispatchEvent(
      new KeyboardEvent('keydown', {key: name, bubbles: true, composed: true}),
    );
    await composer.updateComplete;
  };
  const row = (mode: string) => composer.querySelector<HTMLButtonElement>(`[data-mode="${mode}"]`)!;
  const trigger = composer.querySelector<HTMLButtonElement>('.composer-mode-trigger')!;
  expect(composer.querySelector('.composer-mode-menu')?.hasAttribute('hidden')).to.equal(true);

  // The mode switcher was the only picker before this change and had no keyboard
  // coverage; both now run the same Arrow/Home/End/Escape step.
  trigger.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true}));
  await composer.updateComplete;
  expect(document.activeElement).to.equal(row('auto'), 'ArrowDown opens on the first mode');
  await key('ArrowDown');
  expect(document.activeElement).to.equal(row('fast'));
  await key('End');
  expect(document.activeElement).to.equal(row('research'));
  await key('ArrowDown');
  expect(document.activeElement).to.equal(row('auto'), 'ArrowDown wraps');
  await key('Home');
  expect(document.activeElement).to.equal(row('auto'));
  await key('ArrowUp');
  expect(document.activeElement).to.equal(row('research'), 'ArrowUp wraps backwards');

  // An unrelated key leaves the row where it is instead of moving the selection.
  await key('x');
  expect(document.activeElement).to.equal(row('research'));

  await key('Escape');
  expect(composer.querySelector('.composer-mode-menu')?.hasAttribute('hidden')).to.equal(true);
  expect(document.activeElement).to.equal(trigger);

  // Closing from the trigger keeps the menu closed.
  trigger.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowUp', bubbles: true}));
  await composer.updateComplete;
  expect(document.activeElement).to.equal(row('research'), 'ArrowUp opens on the last mode');
  composer.querySelector<HTMLButtonElement>('.composer-mode-trigger')!.click();
  await composer.updateComplete;
  expect(composer.querySelector('.composer-mode-menu')?.hasAttribute('hidden')).to.equal(true);

  composer.remove();
});

it('Composer keeps one popup open at a time and gives the label room', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: 'high'};
  document.body.appendChild(composer);
  await composer.updateComplete;
  const open = (selector: string) => !composer.querySelector(selector)?.hasAttribute('hidden');
  const click = async (selector: string) => {
    composer.querySelector<HTMLButtonElement>(selector)!.click();
    await composer.updateComplete;
  };

  // Opening one picker settles the other: two menus never stack over the composer.
  await click('.composer-effort-trigger');
  expect(open('.composer-effort-menu')).to.equal(true);
  await click('.composer-mode-trigger');
  expect(open('.composer-mode-menu')).to.equal(true);
  expect(open('.composer-effort-menu')).to.equal(false);
  await click('.composer-effort-trigger');
  expect(open('.composer-effort-menu')).to.equal(true);
  expect(open('.composer-mode-menu')).to.equal(false);

  // A skill directive takes the same single slot.
  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = '/';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  expect(open('.composer-effort-menu')).to.equal(false);

  composer.remove();
});

it('Composer shows no effort picker when the deployment offers none', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.agentEffortOffer = {levels: [], default: null};
  document.body.appendChild(composer);
  await composer.updateComplete;

  expect(composer.querySelector('.composer-effort')).to.equal(null);
  expect(composer.querySelector('.composer-mode')).to.not.equal(null);
  composer.remove();
});

it('Composer starts unselected when the deployment default is not an offered level', async () => {
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  composer.agentEffortOffer = {levels: ['low', 'high', 'max'], default: null};
  document.body.appendChild(composer);
  await composer.updateComplete;

  expect(composer.querySelector('.composer-effort-label')?.textContent).to.equal('Effort');
  const rows = [...composer.querySelectorAll<HTMLButtonElement>('[data-effort]')];
  expect(rows.map((row) => row.getAttribute('aria-checked'))).to.deep.equal(['false', 'false', 'false']);
  expect(composer.querySelector('.composer-effort-default')).to.equal(null);

  // A level the deployment dropped is never reused from storage.
  localStorage.setItem(AGENT_EFFORT_STORAGE_KEY, 'xhigh');
  const second = document.createElement('dl-chat-composer') as DlChatComposer;
  second.attachmentPolicy = policy;
  second.agentEffortOffer = {levels: ['low', 'high'] as const, default: 'low'};
  document.body.appendChild(second);
  await second.updateComplete;
  expect(second.querySelector('.composer-effort-label')?.textContent).to.equal('Low');
  expect(second.querySelectorAll('[data-effort]').length).to.equal(2);
  second.remove();
  composer.remove();
  localStorage.removeItem(AGENT_EFFORT_STORAGE_KEY);
});

it('Composer clears steering text without discarding attachments and detects wrapped drafts', async () => {
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  document.body.appendChild(composer);
  await composer.updateComplete;
  chooseFiles(composer, [new File(['notes'], 'notes.md', {type: 'text/markdown'})]);
  await composer.updateComplete;
  const draft = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  draft.value = 'temporary steering text';
  draft.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;

  composer.clearText();
  await composer.updateComplete;

  expect(composer.hasDraft).to.equal(true);
  expect(composer.textContent).to.contain('notes.md');
  expect(composer.querySelector<HTMLInputElement>('[aria-label="Message"]')?.value).to.equal('');

  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.style.width = '60px';
  input.style.lineHeight = '20px';
  input.value = 'one very long single line that wraps over several visible rows';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(composer.querySelector('form')?.classList.contains('multiline')).to.equal(true);
});

it('Composer completes skill directives with ghost preview, Tab, and shorthand', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    skills: [
      {name: 'tdd', description: 'TDD loop', source: 'global'},
      {name: 'weekly-report', description: 'Weekly reports', source: 'owner'},
      {name: 'z-creator', description: 'Create skills', source: 'builtin'},
    ],
  }), {status: 200, headers: {'Content-Type': 'application/json'}});
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  document.body.appendChild(composer);
  await composer.updateComplete;
  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  const mirror = composer.querySelector<HTMLElement>('.composer-input-mirror')!;

  // Bare slash: panel opens, ghost teaches the full canonical form.
  input.value = '/';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 0));
  await composer.updateComplete;
  expect(mirror.textContent).to.equal('/skill:tdd');
  expect(composer.querySelectorAll('.skill-menu-item').length).to.equal(3);
  expect([...composer.querySelectorAll('.skill-menu-source')].map((item) => item.textContent))
    .to.deep.equal(['Global', 'Mine', 'Built-in']);

  // Tab commits the canonical form with a trailing space and closes the menu.
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Tab', bubbles: true, cancelable: true}));
  await composer.updateComplete;
  expect(input.value).to.equal('/skill:tdd ');
  expect(mirror.textContent).to.equal('/skill:tdd ');
  expect(composer.querySelector('.skill-menu[hidden]')).to.not.equal(null);

  // Shorthand narrows and commits in the shorthand form.
  input.value = '/w';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  expect(mirror.textContent).to.equal('/weekly-report');
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Tab', bubbles: true, cancelable: true}));
  await composer.updateComplete;
  expect(input.value).to.equal('/weekly-report ');

  // Arrows move the highlight and the ghost follows the highlighted skill.
  input.value = '/skill:';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await composer.updateComplete;
  expect(mirror.textContent).to.equal('/skill:tdd');
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true, cancelable: true}));
  await composer.updateComplete;
  expect(mirror.textContent).to.equal('/skill:tdd');
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown', bubbles: true, cancelable: true}));
  await composer.updateComplete;
  expect(mirror.textContent).to.equal('/skill:weekly-report');
  input.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter', bubbles: true, cancelable: true}));
  await composer.updateComplete;
  expect(input.value).to.equal('/skill:weekly-report ');

  composer.remove();
  globalThis.fetch = originalFetch;
});

it('Composer lists a skill published since the slash menu last opened', async () => {
  const originalFetch = globalThis.fetch;
  let skills = [{name: 'tdd', description: 'TDD loop', source: 'global'}];
  globalThis.fetch = async () => new Response(JSON.stringify({skills}), {
    status: 200,
    headers: {'Content-Type': 'application/json'},
  });
  const composer = document.createElement('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  document.body.appendChild(composer);
  await composer.updateComplete;
  const input = composer.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  const type = async (value: string) => {
    input.value = value;
    input.dispatchEvent(new Event('input', {bubbles: true}));
    await composer.updateComplete;
    await new Promise((resolve) => setTimeout(resolve, 0));
    await composer.updateComplete;
  };

  await type('/');
  expect(composer.querySelectorAll('.skill-menu-item').length).to.equal(1);

  // The user leaves the menu, publishes a skill in the conversation, and opens it again.
  await type('');
  skills = [...skills, {name: 'weekly-report', description: 'Weekly reports', source: 'owner'}];
  await type('/');
  expect(composer.querySelectorAll('.skill-menu-item').length).to.equal(2);

  composer.remove();
  globalThis.fetch = originalFetch;
});

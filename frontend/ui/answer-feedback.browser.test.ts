// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {sendKeys} from '@web/test-runner-commands';
import type {AnswerJudgement} from '../api/feedback.ts';
import {ICON_REGISTRY} from '../design-system/index.ts';
import {productionHandles} from '../stores/app-handles.ts';
import {waitFor} from '../testing/dom.ts';
import type {DlAnswerFeedback} from './answer-feedback.ts';
import './answer-feedback.ts';
import type {ToastRequestDetail} from './toast.ts';

interface Served {
  method: string;
  path: string;
  search: string;
  body: Record<string, unknown> | undefined;
}

const originalFetch = window.fetch;
const originalSetTimeout = window.setTimeout;
const originalClearTimeout = window.clearTimeout;
const FEEDBACK = '/web/api/answer/run-1/feedback';
const REASONS = ['unsupported', 'factual_error', 'too_verbose', 'off_topic', 'irrelevant_citations'];

let toasts: string[] = [];
let refusals = 0;
const noteToast = (event: Event): void => {
  toasts.push((event as CustomEvent<ToastRequestDetail>).detail.message);
};
const noteRefusal = (): void => { refusals += 1; };

beforeEach(() => {
  toasts = [];
  refusals = 0;
  document.addEventListener('dl-toast-request', noteToast);
  document.addEventListener('dl-answer-feedback-off', noteRefusal);
});

afterEach(() => {
  document.removeEventListener('dl-toast-request', noteToast);
  document.removeEventListener('dl-answer-feedback-off', noteRefusal);
  window.fetch = originalFetch;
  window.setTimeout = originalSetTimeout;
  window.clearTimeout = originalClearTimeout;
  productionHandles().judgements.forget();
  document.body.replaceChildren();
});

/** Answer the feedback routes the way the server does, and keep what was asked of them. A reply the test
 *  gives itself replaces the server's own for that request. */
function serve(reply?: (served: Served) => Response | Promise<Response> | undefined): Served[] {
  const requests: Served[] = [];
  window.fetch = async (input, init) => {
    const url = new URL(String(input), window.location.origin);
    const served: Served = {
      method: init?.method ?? 'GET',
      path: url.pathname,
      search: url.search,
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : undefined,
    };
    requests.push(served);
    const custom = await reply?.(served);
    if (custom) return custom;
    if (served.method === 'DELETE') return new Response(null, {status: 204});
    return Response.json({run_id: 'run-1', ...served.body, updated_at: '2026-10-09T12:00:00Z'});
  };
  return requests;
}

/** Record the reasons' own closing timer instead of running it, so a test can say where the reader's pause
 *  stands without waiting the three seconds it is measured in. */
function recordPauses() {
  const waiting = new Map<number, {delay: number; handler: () => void}>();
  const delays: number[] = [];
  let armed = 0;
  window.setTimeout = ((handler: TimerHandler, delay?: number) => {
    armed += 1;
    waiting.set(armed, {delay: delay ?? 0, handler: handler as () => void});
    delays.push(delay ?? 0);
    return armed;
  }) as typeof window.setTimeout;
  window.clearTimeout = ((id?: number) => {
    if (id !== undefined) waiting.delete(id);
  }) as typeof window.clearTimeout;
  return {
    /** Every pause armed, by its delay, in the order it was armed. */
    delays,
    /** The pauses still waiting, by their delay: where the reader's pause stands now. */
    pending: (): number[] => [...waiting.values()].map((timer) => timer.delay),
    /** The reader's pause is up: run the timers still waiting, as the browser would. */
    fire(): void {
      const timers = [...waiting.values()];
      waiting.clear();
      for (const timer of timers) timer.handler();
    },
  };
}

/** The element in the wrapping flex row that holds it in the product, after the row's Fork. */
function mount(seed: AnswerJudgement | null = null): DlAnswerFeedback {
  const row = document.createElement('div');
  row.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px;';
  const fork = document.createElement('button');
  fork.type = 'button';
  fork.textContent = 'Fork';
  row.append(fork);
  const element = document.createElement('dl-answer-feedback');
  element.runId = 'run-1';
  element.agent = 'main';
  element.sequence = 7;
  element.judgement = seed;
  element.enabled = true;
  row.append(element);
  document.body.append(row);
  return element;
}

const thumb = (element: DlAnswerFeedback, which: 'up' | 'down'): HTMLButtonElement => (
  element.querySelector<HTMLButtonElement>(`[data-thumb="${which}"]`)!
);
const chip = (element: DlAnswerFeedback, reason: string): HTMLButtonElement => (
  element.querySelector<HTMLButtonElement>(`[data-reason="${reason}"]`)!
);
const pressed = (button: Element): string | null => button.getAttribute('aria-pressed');
/** Whether a reason holds the check that says it is chosen in a shape, and not in colour alone. */
const hasCheck = (button: Element): boolean => {
  const mark = button.querySelector('svg');
  return mark?.getAttribute('aria-hidden') === 'true'
    && mark.querySelector('path')?.getAttribute('d') === ICON_REGISTRY.check.nodes[0][1].d;
};
const editor = (element: DlAnswerFeedback): HTMLElement | null => element.querySelector('[role="group"]');
const note = (element: DlAnswerFeedback): HTMLTextAreaElement => element.querySelector('textarea')!;
/** The note's own Send, which a note the message's judgement does not already hold has. */
const send = (element: DlAnswerFeedback): HTMLButtonElement | null => (
  element.querySelector<HTMLButtonElement>('[data-send]')
);
// What is there, said as a fact: a failure then prints a sentence, and not the whole element it could not describe.
const hasSend = (element: DlAnswerFeedback): boolean => send(element) !== null;
const isOpen = (element: DlAnswerFeedback): boolean => editor(element) !== null;
const isOn = (target: Element | null | undefined): boolean => document.activeElement === target;

/** Wait for the request a gesture sent to be answered and drawn. */
async function settled(element: DlAnswerFeedback): Promise<void> {
  await waitFor(() => !element.pending);
  await element.updateComplete;
}

async function press(element: DlAnswerFeedback, button: HTMLElement): Promise<void> {
  button.click();
  await settled(element);
}

function typeNote(element: DlAnswerFeedback, text: string): void {
  const field = note(element);
  field.value = text;
  field.dispatchEvent(new Event('input', {bubbles: true}));
}

it('rates a message good, takes the rating back, and says so each time the server has answered', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;
  expect(pressed(thumb(element, 'up'))).to.equal('false');
  expect(thumb(element, 'up').getAttribute('aria-label')).to.equal('Good answer');
  expect(thumb(element, 'down').getAttribute('aria-label')).to.equal('Bad answer');

  await press(element, thumb(element, 'up'));
  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {agent: 'main', sequence: 7, value: 'up', reasons: [], note: null},
  }]);
  expect(pressed(thumb(element, 'up'))).to.equal('true');
  // The pressed thumb is the filled one, so the state is not carried by colour alone.
  expect(thumb(element, 'up').querySelector('svg')!.getAttribute('fill')).to.equal('currentColor');
  expect(thumb(element, 'down').querySelector('svg')!.getAttribute('fill')).to.equal('none');
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);

  await press(element, thumb(element, 'up'));
  expect(requests[1]).to.deep.equal({
    method: 'DELETE', path: FEEDBACK, search: '?agent=main&sequence=7', body: undefined,
  });
  expect(pressed(thumb(element, 'up'))).to.equal('false');
  expect(toasts).to.deep.equal(['Thanks for the feedback.', 'Feedback removed.']);
});

it('replaces a thumb down with a thumb up, and closes the reasons that thumb down had open', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['off_topic'], note: 'rambles'});
  await element.updateComplete;
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  thumb(element, 'down').click();
  await element.updateComplete;
  expect(editor(element)).not.to.equal(null);

  // Open on a stored thumb down, a click on the same thumb would take it back: this one is the other.
  await press(element, thumb(element, 'up'));

  expect(requests.map((served) => [served.method, served.body])).to.deep.equal([
    ['PUT', {agent: 'main', sequence: 7, value: 'up', reasons: [], note: null}],
  ]);
  expect(pressed(thumb(element, 'up'))).to.equal('true');
  expect(pressed(thumb(element, 'down'))).to.equal('false');
  expect(editor(element)).to.equal(null);
});

it('opens the reasons on a thumb down and stores nothing until one is chosen', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;

  thumb(element, 'down').click();
  await element.updateComplete;

  expect(requests).to.deep.equal([]);
  expect(thumb(element, 'down').getAttribute('aria-expanded')).to.equal('true');
  expect(pressed(thumb(element, 'down'))).to.equal('false');
  const group = editor(element)!;
  expect(thumb(element, 'down').getAttribute('aria-controls')).to.equal(group.id);
  expect(group.getAttribute('aria-label')).to.equal('What was wrong with this answer?');
  expect([...group.querySelectorAll<HTMLElement>('[data-reason]')].map((each) => each.dataset.reason))
    .to.deep.equal(REASONS);
  expect([...group.querySelectorAll('[data-reason]')].map(pressed)).to.deep.equal(Array(5).fill('false'));
  expect([...group.querySelectorAll('[data-reason]')].map(hasCheck)).to.deep.equal(Array(5).fill(false));
  expect([...group.querySelectorAll('[data-reason]')].map((each) => each.textContent?.trim()))
    .to.deep.equal(['Unsupported claims', 'Factual error', 'Too verbose', 'Off topic', 'Irrelevant citations']);
  expect(note(element).getAttribute('aria-label')).to.equal('Add a note (optional)');
  expect(note(element).maxLength).to.equal(2000);
  // Nothing is typed yet, so the reasons are the only controls: words alone would bring Send of their own.
  expect(group.querySelectorAll('button')).to.have.length(5);

  // Closing them again, by the thumb that opened them or by Escape, stores nothing either.
  await press(element, thumb(element, 'down'));
  expect(editor(element)).to.equal(null);
  thumb(element, 'down').click();
  await element.updateComplete;
  const pressedEscape = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true});
  chip(element, 'off_topic').dispatchEvent(pressedEscape);
  await element.updateComplete;
  // The Inspector closes on an Escape nobody handled, and this one closed the reasons.
  expect(pressedEscape.defaultPrevented).to.equal(true);
  expect(editor(element)).to.equal(null);
  expect(requests).to.deep.equal([]);
});

it('stores a thumb down with the first reason chosen and the note as typed, and leaves the reasons to the reader', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  typeNote(element, '  cites nothing  ');

  chip(element, 'off_topic').focus();
  await press(element, chip(element, 'off_topic'));

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {agent: 'main', sequence: 7, value: 'down', reasons: ['off_topic'], note: 'cites nothing'},
  }]);
  // A choice stores the rating and states it in the pill: it never takes the form away from a reader who is
  // still in it. What closes the reasons is the reader's own pause, and only that.
  expect(isOpen(element)).to.equal(true);
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  expect(thumb(element, 'down').getAttribute('aria-expanded')).to.equal('true');
  expect(thumb(element, 'down').querySelector('svg')!.getAttribute('fill')).to.equal('currentColor');
  expect(isOn(chip(element, 'off_topic')), 'the reader stays on the pill they pressed').to.equal(true);
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);
});

it('adds a reason to a stored thumb down in the order the server keeps them, with the stored note to edit', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['off_topic'], note: 'rambles'});
  await element.updateComplete;

  thumb(element, 'down').click();
  await element.updateComplete;
  expect(requests).to.deep.equal([]);
  expect(chip(element, 'off_topic').getAttribute('aria-pressed')).to.equal('true');
  expect(chip(element, 'unsupported').getAttribute('aria-pressed')).to.equal('false');
  // A chosen reason is told from the others by a check as well as by colour, and the check adds no words.
  expect(hasCheck(chip(element, 'off_topic'))).to.equal(true);
  expect(hasCheck(chip(element, 'unsupported'))).to.equal(false);
  expect(chip(element, 'off_topic').textContent?.trim()).to.equal('Off topic');
  expect(note(element).value).to.equal('rambles');
  typeNote(element, 'rambles and cites nothing');
  await press(element, chip(element, 'unsupported'));

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {
      agent: 'main', sequence: 7, value: 'down', reasons: ['unsupported', 'off_topic'],
      note: 'rambles and cites nothing',
    },
  }]);
  expect(isOpen(element)).to.equal(true);
});

it('takes a reason away with the reasons left open, and the rating with the last one when no words are left', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['unsupported', 'off_topic'], note: null});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  await press(element, chip(element, 'unsupported'));
  expect(requests.at(-1)).to.deep.equal({
    method: 'PUT', path: FEEDBACK, search: '',
    body: {agent: 'main', sequence: 7, value: 'down', reasons: ['off_topic'], note: null},
  });
  expect(editor(element)).not.to.equal(null);
  expect(chip(element, 'unsupported').getAttribute('aria-pressed')).to.equal('false');
  expect(hasCheck(chip(element, 'unsupported'))).to.equal(false);
  expect(hasCheck(chip(element, 'off_topic'))).to.equal(true);
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  // The pill the reader pressed has already let go, so the smaller choice is stored without a receipt.
  expect(toasts).to.deep.equal([]);

  await press(element, chip(element, 'off_topic'));
  expect(requests.at(-1)).to.deep.equal({
    method: 'DELETE', path: FEEDBACK, search: '?agent=main&sequence=7', body: undefined,
  });
  // The thumb is unlit and the reasons are still there to choose again.
  expect(pressed(thumb(element, 'down'))).to.equal('false');
  expect(editor(element)).not.to.equal(null);
  expect([...element.querySelectorAll('[data-reason]')].map(pressed)).to.deep.equal(Array(5).fill('false'));
  expect(toasts).to.deep.equal(['Feedback removed.']);

  // Choosing again stores a rating again, and the reasons stay for the reader's pause.
  await press(element, chip(element, 'too_verbose'));
  expect(requests.at(-1)?.body).to.deep.equal(
    {agent: 'main', sequence: 7, value: 'down', reasons: ['too_verbose'], note: null},
  );
  expect(isOpen(element)).to.equal(true);
  expect(toasts).to.deep.equal(['Feedback removed.', 'Thanks for the feedback.']);
  const puts = requests.filter((served) => served.method === 'PUT').map((served) => served.body!);
  expect(
    puts.filter((body) => (body.reasons as string[]).length === 0),
    'no words were given, so no thumb down that says nothing was kept',
  ).to.deep.equal([]);
});

it('keeps the thumb down when the last reason goes and the words the reader typed still say why', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['off_topic'], note: null});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  typeNote(element, 'it answers a different question');
  await element.updateComplete;
  await press(element, chip(element, 'off_topic'));

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {
      agent: 'main', sequence: 7, value: 'down', reasons: [],
      note: 'it answers a different question',
    },
  }]);
  // The rating stands on the words alone, the reasons stay open for another, and nothing is receipted.
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  expect(editor(element)).not.to.equal(null);
  expect(toasts).to.deep.equal([]);
  expect(hasSend(element), 'the words stored are the words typed').to.equal(false);
});

it('takes a stored thumb down back when it is pressed with the reasons open, and only closes them when nothing is stored', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['too_verbose'], note: 'long'});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  await press(element, thumb(element, 'down'));

  expect(requests).to.deep.equal([
    {method: 'DELETE', path: FEEDBACK, search: '?agent=main&sequence=7', body: undefined},
  ]);
  expect(pressed(thumb(element, 'down'))).to.equal('false');
  expect(editor(element)).to.equal(null);

  // Nothing is stored now: the same two presses open the reasons and close them, without a word to the server.
  await press(element, thumb(element, 'down'));
  await press(element, thumb(element, 'down'));
  expect(requests).to.have.length(1);
  expect(editor(element)).to.equal(null);
});

it('waits for the server: while a request is out the controls hold still and nothing is shown as kept', async () => {
  let answer!: (response: Response) => void;
  const requests = serve(() => new Promise<Response>((resolve) => { answer = resolve; }));
  const element = mount();
  await element.updateComplete;

  thumb(element, 'up').click();
  await element.updateComplete;
  expect(pressed(thumb(element, 'up'))).to.equal('false');
  expect(thumb(element, 'up').getAttribute('aria-disabled')).to.equal('true');
  expect(thumb(element, 'down').getAttribute('aria-disabled')).to.equal('true');
  expect(thumb(element, 'up').closest('[aria-busy]')?.getAttribute('aria-busy')).to.equal('true');
  // A second press, of either thumb, is not a second request, and a disabled control keeps the reader's place.
  thumb(element, 'up').focus();
  thumb(element, 'up').click();
  thumb(element, 'down').click();
  await element.updateComplete;
  expect(requests).to.have.length(1);
  expect(editor(element)).to.equal(null);
  expect(document.activeElement).to.equal(thumb(element, 'up'));
  expect(toasts).to.deep.equal([]);

  answer(Response.json({
    run_id: 'run-1', agent: 'main', sequence: 7, value: 'up', reasons: [], note: null, updated_at: '',
  }));
  await settled(element);
  expect(pressed(thumb(element, 'up'))).to.equal('true');
  expect(thumb(element, 'up').hasAttribute('aria-disabled')).to.equal(false);
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);
});

it('leaves what shows alone when the server refuses, and says why in the server\'s own words', async () => {
  serve(() => Response.json({detail: 'Run not found', error_type: 'not_found'}, {status: 404}));
  const element = mount();
  await element.updateComplete;

  await press(element, thumb(element, 'up'));

  expect(pressed(thumb(element, 'up'))).to.equal('false');
  expect(toasts).to.deep.equal(['Run not found']);
  expect(refusals).to.equal(0);

  // A server that gives no reason at all leaves the page's own words.
  serve(() => new Response('', {status: 502}));
  await press(element, thumb(element, 'up'));
  expect(toasts.at(-1)).to.equal('Your feedback could not be saved.');
  serve(() => new Response('', {status: 502}));
  const rated = mount({value: 'up', reasons: [], note: null});
  await rated.updateComplete;
  await press(rated, thumb(rated, 'up'));
  expect(pressed(thumb(rated, 'up'))).to.equal('true');
  expect(toasts.at(-1)).to.equal('Your feedback could not be removed.');
});

it('stops offering the thumbs when the server says feedback is off for the reader, but not for any other 403', async () => {
  // Something in front of the server, a proxy or a firewall, refuses without the server's envelope.
  serve(() => new Response('<html>Forbidden</html>', {status: 403, headers: {'Content-Type': 'text/html'}}));
  const element = mount();
  await element.updateComplete;
  await press(element, thumb(element, 'up'));
  expect(element.querySelector('button')).not.to.equal(null);
  expect(refusals).to.equal(0);
  expect(toasts.at(-1)).to.equal('Your feedback could not be saved.');

  serve(() => Response.json(
    {detail: 'Cross-origin request rejected', error_type: 'auth', error_kind: 'cross_origin_rejected'},
    {status: 403},
  ));
  await press(element, thumb(element, 'up'));
  // A request the origin guard refused is a matter of reloading the page and not of feedback being off.
  expect(element.querySelector('button')).not.to.equal(null);
  expect(refusals).to.equal(0);
  expect(toasts.at(-1)).to.contain('origin could not be verified');

  serve(() => Response.json(
    {detail: 'Answer feedback is turned off for you', error_type: 'auth'},
    {status: 403},
  ));
  thumb(element, 'up').focus();
  await press(element, thumb(element, 'up'));

  expect(toasts.at(-1)).to.equal('Feedback is turned off for you.');
  expect(refusals).to.equal(1);
  expect(element.querySelector('button')).to.equal(null);
  expect(element.enabled).to.equal(false);
  // The thumb the reader was on is gone, and they are on the button before it in the row, not on the page.
  expect(document.activeElement).to.equal(element.previousElementSibling);
});

it('shows Send where the note typed is not the one the judgement holds, and never where nothing would be left to say', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['off_topic'], note: 'rambles'});
  await element.updateComplete;

  // The reasons open on the note as it is stored, and a reader who only clicks a reason is shown nothing more.
  thumb(element, 'down').click();
  await element.updateComplete;
  expect(note(element).value).to.equal('rambles');
  expect(hasSend(element)).to.equal(false);

  typeNote(element, 'rambles on');
  await element.updateComplete;
  expect(hasSend(element)).to.equal(true);
  expect(send(element)!.getAttribute('aria-label')).to.equal('Send note');
  expect(send(element)!.hasAttribute('aria-disabled')).to.equal(false);
  expect(send(element)!.querySelector('svg')?.getAttribute('aria-hidden')).to.equal('true');

  // Typed back to what is stored, or to it with the blanks the server trims, there is nothing to send again.
  typeNote(element, 'rambles');
  await element.updateComplete;
  expect(hasSend(element)).to.equal(false);
  typeNote(element, '  rambles  ');
  await element.updateComplete;
  expect(hasSend(element)).to.equal(false);
  expect(requests).to.deep.equal([]);

  // A thumb down stored without a note holds the empty one: any words differ from it, and blanks do not.
  const bare = mount({value: 'down', reasons: ['too_verbose'], note: null});
  await bare.updateComplete;
  thumb(bare, 'down').click();
  await bare.updateComplete;
  typeNote(bare, '   ');
  await bare.updateComplete;
  expect(hasSend(bare)).to.equal(false);
  typeNote(bare, 'Too long.');
  await bare.updateComplete;
  expect(hasSend(bare)).to.equal(true);

  // A thumb down that says why in words alone holds those words: clearing them leaves it nothing to say, so
  // Send is not there, and the lit thumb is how the reader takes that rating back.
  const words = mount({value: 'down', reasons: [], note: 'cites nothing'});
  words.sequence = 11;
  await words.updateComplete;
  thumb(words, 'down').click();
  await words.updateComplete;
  expect(note(words).value).to.equal('cites nothing');
  typeNote(words, '');
  await words.updateComplete;
  expect(hasSend(words), 'neither a reason nor words is no judgement').to.equal(false);
  typeNote(words, 'cites nothing at all');
  await words.updateComplete;
  expect(hasSend(words)).to.equal(true);
});

it('keeps a thumb down that says why in the reader\'s own words, sent from the thumb down that opened the reasons', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  // Nothing is typed and nothing is stored: there is nothing for Send to keep yet.
  expect(hasSend(element), 'the reasons are open on an empty note').to.equal(false);

  note(element).focus();
  await sendKeys({type: 'cites nothing'});
  expect(hasSend(element), 'the words alone are a judgement to keep').to.equal(true);
  await press(element, send(element)!);

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {agent: 'main', sequence: 7, value: 'down', reasons: [], note: 'cites nothing'},
  }]);
  // Send hides what it kept and thanks the reader in the same receipt, as a chosen reason does; the reasons
  // stay for the reader's own pause.
  expect(isOpen(element)).to.equal(true);
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);
  expect(note(element).value).to.equal('cites nothing');
  expect(hasSend(element)).to.equal(false);
});

it('keeps the note travelling with the reason chosen while nothing has been sent', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  note(element).focus();
  await sendKeys({type: 'cites nothing'});
  // Send is there, and Shift+Enter still breaks the line so the reader can say more than one thing.
  await sendKeys({press: 'Shift+Enter'});
  await sendKeys({type: 'at all'});
  expect(note(element).value).to.equal('cites nothing\nat all');
  expect(requests).to.deep.equal([]);

  await press(element, chip(element, 'off_topic'));
  expect(requests[0]?.body).to.deep.equal({
    agent: 'main', sequence: 7, value: 'down', reasons: ['off_topic'], note: 'cites nothing\nat all',
  });
  expect(isOpen(element), 'the reasons stay for the reader').to.equal(true);
});

it('replaces a thumb up with a words-alone thumb down where the reader changes their mind', async () => {
  const requests = serve();
  const element = mount({value: 'up', reasons: [], note: null});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  typeNote(element, 'On second thought, this answers the wrong question.');
  await element.updateComplete;
  expect(hasSend(element)).to.equal(true);
  await press(element, send(element)!);

  expect(requests[0]?.body).to.deep.equal({
    agent: 'main', sequence: 7, value: 'down', reasons: [],
    note: 'On second thought, this answers the wrong question.',
  });
  expect(pressed(thumb(element, 'up'))).to.equal('false');
  expect(pressed(thumb(element, 'down'))).to.equal('true');
});

it('sends a changed note with the reasons as they are stored, and thanks the reader as a chosen reason does', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['unsupported', 'off_topic'], note: 'rambles'});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  typeNote(element, '  rambles and cites nothing  ');
  await element.updateComplete;

  send(element)!.focus();
  await press(element, send(element)!);

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {
      agent: 'main', sequence: 7, value: 'down', reasons: ['unsupported', 'off_topic'],
      note: 'rambles and cites nothing',
    },
  }]);
  expect(isOpen(element)).to.equal(true);
  // Send hides with the note it kept, so the reader who was on it lands on the field it belonged to.
  expect(isOn(note(element)), 'focus lands on the note field Send belonged to').to.equal(true);
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);
  expect(pressed(thumb(element, 'down'))).to.equal('true');

  // What was sent is what is stored now, so there is nothing more to send, and the reasons show what is kept.
  // The field keeps the words as they were typed: the judgement holds them trimmed, and Send compares the two.
  expect(note(element).value).to.equal('  rambles and cites nothing  ');
  expect(hasSend(element)).to.equal(false);
  expect(chip(element, 'unsupported').getAttribute('aria-pressed')).to.equal('true');
});

it('sends no note at all when a stored note is cleared', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['too_verbose'], note: 'rambles'});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;

  typeNote(element, '');
  await element.updateComplete;
  expect(hasSend(element), 'a cleared note differs from the one stored').to.equal(true);
  await press(element, send(element)!);

  expect(requests).to.deep.equal([{
    method: 'PUT', path: FEEDBACK, search: '',
    body: {agent: 'main', sequence: 7, value: 'down', reasons: ['too_verbose'], note: null},
  }]);
  expect(isOpen(element), 'the reasons stay, on a note that is now the stored one').to.equal(true);
  expect(note(element).value).to.equal('');
  expect(hasSend(element)).to.equal(false);
});

it('sends the note on Enter, breaks the line on Shift+Enter, and never sends on the Enter of an IME composition', async () => {
  const requests = serve();
  const element = mount({value: 'down', reasons: ['too_verbose'], note: null});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  note(element).focus();

  await sendKeys({type: 'first'});
  await sendKeys({press: 'Shift+Enter'});
  await sendKeys({type: 'second'});
  expect(note(element).value).to.equal('first\nsecond');
  expect(requests).to.deep.equal([]);

  // The Enter that commits a composition is the composition's.
  const composing = new InputEvent('beforeinput', {
    inputType: 'insertLineBreak', isComposing: true, bubbles: true, cancelable: true,
  });
  note(element).dispatchEvent(composing);
  expect(composing.defaultPrevented).to.equal(false);
  expect(requests).to.deep.equal([]);

  await sendKeys({press: 'Enter'});
  await waitFor(() => requests.length === 1);
  expect(requests[0]?.body).to.deep.equal({
    agent: 'main', sequence: 7, value: 'down', reasons: ['too_verbose'], note: 'first\nsecond',
  });
  await settled(element);
  expect(isOpen(element), 'the reasons stay open for the reader').to.equal(true);
  expect(toasts).to.deep.equal(['Thanks for the feedback.']);
});

it('keeps the reasons open for the Escape that cancels an IME composition in the note, and closes them on the next', async () => {
  const requests = serve();
  const element = mount();
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  note(element).focus();
  typeNote(element, 'yanjiu');
  await element.updateComplete;

  // Chromium marks the keys of a composition; WebKit sends them after it ended, as keyCode 229.
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    const composing = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true, ...init});
    note(element).dispatchEvent(composing);
    await element.updateComplete;
    // The composition keeps its key: the reasons neither close on it nor mark it handled.
    expect(composing.defaultPrevented, `the Escape of a composition (${Object.keys(init)[0]}) is left alone`)
      .to.equal(false);
    expect(isOpen(element)).to.equal(true);
    expect(note(element).value).to.equal('yanjiu');
  }

  const plain = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true});
  note(element).dispatchEvent(plain);
  await element.updateComplete;
  expect(plain.defaultPrevented).to.equal(true);
  expect(editor(element)).to.equal(null);
  expect(requests, 'closing the reasons stores nothing').to.deep.equal([]);
});

it('waits for the server before the note lets go of Send, and keeps the words and the reasons when sending fails', async () => {
  let answer!: (response: Response) => void;
  const requests = serve(() => new Promise<Response>((resolve) => { answer = resolve; }));
  const element = mount({value: 'down', reasons: ['off_topic'], note: 'rambles'});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  typeNote(element, 'rambles, and more');
  await element.updateComplete;
  const idle = send(element)!.innerHTML;

  send(element)!.focus();
  send(element)!.click();
  await element.updateComplete;
  // Nothing is kept until the server says so; Send waits, in a shape of its own as well as dimmed, and holds focus.
  expect(requests).to.have.length(1);
  expect(send(element)!.getAttribute('aria-disabled')).to.equal('true');
  expect(send(element)!.innerHTML).not.to.equal(idle);
  expect(send(element)!.querySelector('circle') !== null, 'Send shows a dot while the request is out').to.equal(true);
  expect(isOn(send(element)), 'focus stays on Send').to.equal(true);
  expect(isOpen(element)).to.equal(true);
  // Neither a second press nor an Enter is a second request.
  send(element)!.click();
  note(element).focus();
  await sendKeys({press: 'Enter'});
  expect(requests).to.have.length(1);

  answer(new Response('', {status: 502}));
  await settled(element);
  expect(toasts).to.deep.equal(['Your feedback could not be saved.']);
  expect(isOpen(element), 'the reasons stay open for the reader to try again').to.equal(true);
  expect(note(element).value, 'the Enter that was swallowed added no line').to.equal('rambles, and more');
  expect(hasSend(element)).to.equal(true);
  expect(send(element)!.hasAttribute('aria-disabled')).to.equal(false);
  expect(send(element)!.innerHTML).to.equal(idle);
  expect(pressed(thumb(element, 'down'))).to.equal('true');
  expect(chip(element, 'off_topic').getAttribute('aria-pressed')).to.equal('true');

  // Sent again once the server answers, it keeps the reasons for the reader as before.
  serve();
  await press(element, send(element)!);
  expect(isOpen(element)).to.equal(true);
  expect(toasts.at(-1)).to.equal('Thanks for the feedback.');
});

// ── The reader's pause ──

it('closes the reasons three seconds after the reader\'s last act on a control, and never before one', async () => {
  const pauses = recordPauses();
  const requests = serve();
  const element = mount();
  await element.updateComplete;

  thumb(element, 'down').click();
  await element.updateComplete;
  expect(pauses.pending(), 'a reader who has only opened the reasons is still choosing').to.deep.equal([]);

  chip(element, 'off_topic').focus();
  await press(element, chip(element, 'off_topic'));
  expect(pauses.pending(), 'a chosen reason leaves one pause waiting, the full three seconds')
    .to.deep.equal([3000]);
  expect(isOpen(element), 'and the choice never closes the reasons itself').to.equal(true);

  // The pause is measured from the reader's last act, so another act starts it again.
  const arms = pauses.delays.length;
  await press(element, chip(element, 'unsupported'));
  expect(pauses.delays.length, 'the act arms the pause again').to.be.greaterThan(arms);
  expect(pauses.pending(), 'and only the newest pause is left waiting').to.deep.equal([3000]);
  expect(isOpen(element)).to.equal(true);

  pauses.fire();
  await element.updateComplete;
  expect(isOpen(element), 'the reader\'s pause is what closes them').to.equal(false);
  // The reader was on the pill they pressed, which went with the reasons: they land on the thumb.
  expect(isOn(thumb(element, 'down'))).to.equal(true);
  expect(requests).to.have.length(2);
});

it('counts only the controls as acts: a press outside them neither restarts the pause nor closes the reasons', async () => {
  const pauses = recordPauses();
  serve();
  const element = mount();
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  await press(element, chip(element, 'off_topic'));
  expect(pauses.pending()).to.deep.equal([3000]);

  // The box the reasons draw themselves in, and the page around the element: neither is a control.
  const arms = pauses.delays.length;
  editor(element)!.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
  element.parentElement!.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
  document.body.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true}));
  expect(pauses.delays.length, 'an act outside the controls arms nothing').to.equal(arms);
  expect(pauses.pending(), 'the pause the reader left runs on').to.deep.equal([3000]);
  expect(isOpen(element), 'and nothing outside takes the reasons away either').to.equal(true);

  pauses.fire();
  await element.updateComplete;
  expect(isOpen(element)).to.equal(false);
});

it('holds the reasons open over words the reader has not sent, and closes them on the pause after they are sent', async () => {
  const pauses = recordPauses();
  serve();
  const element = mount({value: 'down', reasons: ['off_topic'], note: 'rambles'});
  await element.updateComplete;
  thumb(element, 'down').click();
  await element.updateComplete;
  expect(pauses.pending()).to.deep.equal([]);

  typeNote(element, 'rambles on');
  expect(pauses.pending(), 'typing in the note field is an act on a control').to.deep.equal([3000]);
  typeNote(element, 'rambles on and on');
  expect(pauses.delays.length, 'every act arms the pause again').to.equal(2);
  expect(pauses.pending(), 'leaving only the newest one waiting').to.deep.equal([3000]);

  // The pause is up, but the words in the field are not the judgement's: closing would throw them away.
  pauses.fire();
  await element.updateComplete;
  expect(isOpen(element), 'unsent words hold the reasons open').to.equal(true);
  expect(note(element).value).to.equal('rambles on and on');
  expect(pauses.pending(), 'and another pause is armed for them').to.deep.equal([3000]);

  await press(element, send(element)!);
  pauses.fire();
  await element.updateComplete;
  expect(isOpen(element), 'the words are the judgement\'s now, so the pause may close them').to.equal(false);
});

it('draws nothing for a reader who is not offered feedback', async () => {
  const element = mount();
  element.enabled = false;
  await element.updateComplete;
  expect(element.querySelector('button')).to.equal(null);
  expect(element.textContent?.trim()).to.equal('');
});

it('shows a message the way the page last saw the server keep it, whatever an earlier read of it says', async () => {
  serve();
  const first = mount();
  await first.updateComplete;
  await press(first, thumb(first, 'up'));

  // The turn is drawn again from the history it was read from, which was read before the rating.
  first.parentElement!.remove();
  const again = mount(null);
  await again.updateComplete;
  expect(pressed(thumb(again, 'up'))).to.equal('true');

  // Another message is its own.
  const other = mount(null);
  other.sequence = 9;
  await other.updateComplete;
  expect(pressed(thumb(other, 'up'))).to.equal('false');
  expect(other.kept).to.equal(null);

  // The page learns nothing from a read about a message it has not judged, and the read stands.
  other.judgement = {value: 'down', reasons: ['off_topic'], note: null};
  await other.updateComplete;
  expect(pressed(thumb(other, 'down'))).to.equal('true');
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The live browser dock: the four states, the seat, and the frames one socket carries.

 * The frames socket is a stub the test drives, so what is asserted here is what the dock
 * does with what a real server sends: binary JPEG bytes become the picture on the stage,
 * text messages become the state, and closing the dock is what closes the socket. A page
 * nobody is looking at is the one thing that pauses it without ending it: the socket goes
 * with the page, the picture stays on the stage, and showing the page again resumes it.
 */

import {expect} from '@esm-bundle/chai';
import {defineDesignSystemElements} from '../design-system/index.ts';
import {waitFor} from '../testing/dom.ts';
import './inspector-live.ts';
import type {DlInspectorLive} from './inspector-live.ts';

defineDesignSystemElements();

/** One frames socket, as the dock opens it, with the test on the server's side. */
class LiveSocket {
  static readonly sockets: LiveSocket[] = [];
  static failNext = false;

  readonly url: string;
  binaryType = 'blob';
  closedWith: {code: number; reason: string} | null = null;
  onmessage: ((event: {data: unknown}) => void) | null = null;
  onclose: ((event: {code: number; reason: string}) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    LiveSocket.sockets.push(this);
    if (LiveSocket.failNext) {
      LiveSocket.failNext = false;
      // The network, not the server: the dock retries this one rather than giving up.
      queueMicrotask(() => this.onclose?.({code: 1006, reason: ''}));
    }
  }

  close(code = 1000, reason = ''): void {
    this.closedWith = {code, reason};
    this.onclose?.({code, reason});
  }

  /** One text message: the state a frame arrived in.
   *
   * A socket that has closed delivers nothing, exactly as a real one does: a message sent
   * to a finished socket is a programming error, and a stub that quietly swallowed it hid
   * the resume-after-hide defect for a whole review cycle.
   */
  text(update: Record<string, unknown>): void {
    if (this.closedWith !== null) {
      throw new Error('a message was sent to a socket that has already closed');
    }
    this.onmessage?.({data: JSON.stringify({
      at: 1, frames_per_second: null, jpeg_quality: null, max_width: null, ...update,
    })});
  }

  /** One heartbeat: a quiet moment, and the newest frame the session has delivered. */
  beat(at: number, sequence: number): void {
    if (this.closedWith !== null) {
      throw new Error('a heartbeat was sent to a socket that has already closed');
    }
    this.onmessage?.({data: JSON.stringify({at, sequence})});
  }

  /** One binary message: a frame's own JPEG bytes. */
  binary(bytes: ArrayBuffer): void {
    if (this.closedWith !== null) {
      throw new Error('a frame was sent to a socket that has already closed');
    }
    this.onmessage?.({data: bytes});
  }

  /** One frame as the server really sends it: the fixed header, then the JPEG. */
  frame(jpeg: ArrayBuffer, sequence: number, capturedAt = Date.now() / 1000): void {
    const body = new Uint8Array(jpeg);
    const message = new Uint8Array(16 + body.length);
    const header = new DataView(message.buffer);
    header.setBigUint64(0, BigInt(sequence));
    header.setFloat64(8, capturedAt);
    message.set(body, 16);
    this.binary(message.buffer);
  }

  /** The server, or the network, closing the socket. */
  drop(code: number, reason = ''): void {
    this.onclose?.({code, reason});
  }
}

const originalFetch = window.fetch;

/** A JPEG of one flat colour, so the dock's own decode can be read back off the canvas. */
async function jpeg(width = 8, height = 8): Promise<ArrayBuffer> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d')!;
  context.fillStyle = '#123456';
  context.fillRect(0, 0, width, height);
  const blob = await new Promise<Blob>((resolve) => {
    canvas.toBlob((made) => { resolve(made!); }, 'image/jpeg', 1);
  });
  return await blob.arrayBuffer();
}

interface Write {
  url: string;
  method: string;
  body: string | null;
  headers: Record<string, string>;
}

const writes: Write[] = [];
/** Every socket the dock opened, newest last, as the page's own WebSocket saw them. */
const sockets = LiveSocket.sockets;

function update(overrides: Record<string, unknown>): Record<string, unknown> {
  return {control: 'agent', driver: null, attached: true, ended: '', frame: null, ...overrides};
}

/** Show or hide the page this dock is in, as the browser does when a tab is switched. */
function pageVisible(visible: boolean): void {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    get: () => (visible ? 'visible' : 'hidden'),
  });
  document.dispatchEvent(new Event('visibilitychange'));
}

/** The state the dock draws, read from the surface that carries it. */
function liveState(dock: DlInspectorLive): string | null {
  return dock.querySelector('[data-live-state]')?.getAttribute('data-live-state') ?? null;
}

/** The seat credential this dock sent on a write, which is how a test plays the driver. */
function seatOf(request: {headers: Record<string, string>}): string {
  return request.headers['X-Live-Seat'] ?? '';
}

/** What the stubbed server mints for a take: a label every reader may see, and a secret. */
const SEAT_TOKEN = 'seat-token-9f2c41ab';
const SEAT_LABEL = 'viewer-3f9a1c04';

/** The pixel the dock painted at (0, 0), which is the frame it decoded and drew. */
function painted(dock: DlInspectorLive): [number, number, number] | null {
  const canvas = dock.querySelector('canvas');
  if (canvas === null || canvas.width === 0 || canvas.height === 0) return null;
  const pixel = canvas.getContext('2d')?.getImageData(0, 0, 1, 1).data;
  return pixel === undefined ? null : [pixel[0]!, pixel[1]!, pixel[2]!];
}

async function mounted(): Promise<DlInspectorLive> {
  const element = document.createElement('dl-inspector-live') as DlInspectorLive;
  document.body.append(element);
  element.runId = 'run-1';
  element.active = true;
  await element.updateComplete;
  await waitFor(() => sockets.length > 0);
  return element;
}

/** The socket the dock is watching through now. */
async function live(dock: DlInspectorLive): Promise<LiveSocket> {
  await waitFor(() => sockets.length > 0);
  const socket = sockets.at(-1)!;
  await dock.updateComplete;
  return socket;
}

beforeEach(() => {
  writes.length = 0;
  sockets.length = 0;
  LiveSocket.failNext = false;
  (window as unknown as {WebSocket: unknown}).WebSocket = LiveSocket;
  window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const headers: Record<string, string> = {};
    for (const [name, value] of Object.entries((init?.headers ?? {}) as Record<string, string>)) {
      headers[name] = value;
    }
    writes.push({
      url: String(input),
      method: init?.method ?? 'GET',
      body: typeof init?.body === 'string' ? init.body : null,
      headers,
    });
    const state = JSON.parse(String(init?.body)).state;
    // The seat's own answer: the credential on the take that minted it, and nothing on the
    // hand-back. This is the only place a credential ever reaches the dock.
    return new Response(JSON.stringify({
      control: state,
      driver: state === 'human' ? SEAT_LABEL : null,
      token: state === 'human' ? SEAT_TOKEN : null,
    }), {
      status: 200,
      headers: {'Content-Type': 'application/json'},
    });
  }) as typeof fetch;
});

afterEach(() => {
  window.fetch = originalFetch;
  document.body.replaceChildren();
  pageVisible(true);
});

it('shows four states and nothing else, and draws the newest frame it is sent', async () => {
  const dock = await mounted();
  expect(liveState(dock)).to.equal('idle');
  expect(dock.textContent).to.contain('No live page yet');

  const socket = await live(dock);
  expect(socket.url).to.contain('/web/api/runs/run-1/browser/live/frames');
  // The binary opcode is read as bytes: nothing here unwraps a Blob or a base64 string.
  expect(socket.binaryType).to.equal('arraybuffer');

  // The server sends the state first and the frame after it; the frame is the picture,
  // and the text is what says which state that picture is in.
  socket.text(update({}));
  socket.frame(await jpeg(), 1);
  await waitFor(() => painted(dock) !== null);
  await waitFor(() => liveState(dock) === 'live');
  const pixel = painted(dock)!;
  expect(Math.abs(pixel[0] - 0x12)).to.be.lessThan(24);
  expect(Math.abs(pixel[1] - 0x34)).to.be.lessThan(24);
  expect(Math.abs(pixel[2] - 0x56)).to.be.lessThan(24);
  expect(dock.querySelector('canvas')?.getAttribute('role')).to.equal('img');
  expect(dock.querySelector('canvas')?.getAttribute('aria-label')).to.contain('driving');

  // Taking the seat is what makes this dock the driver: the take answers with a credential,
  // and the label the stream names is the one that answer carried.
  expect(seatOf({headers: {}})).to.equal('');
  dock.querySelector('button')?.click();
  await waitFor(() => liveState(dock) === 'taken');
  const control = writes.filter((write) => write.url.endsWith('/control')).at(-1)!;
  expect(seatOf(control)).to.equal('');
  socket.text(update({control: 'human', driver: SEAT_LABEL}));
  await waitFor(() => liveState(dock) === 'taken');
  expect(dock.textContent).to.contain('the agent’s page actions are refused');
  expect(dock.querySelector('button')?.textContent).to.contain('Hand back');

  socket.text(update({control: 'human', driver: SEAT_LABEL, ended: 'expired'}));
  await waitFor(() => liveState(dock) === 'ended');
  expect(dock.textContent).to.contain('This live session has ended');
});

it('drives with the credential its own take was answered with, and never with a label', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  dock.querySelector('button')?.click();
  await waitFor(() => liveState(dock) === 'taken');
  socket.text(update({control: 'human', driver: SEAT_LABEL}));
  await waitFor(() => liveState(dock) === 'taken');

  socket.frame(await jpeg(), 1);
  await waitFor(() => painted(dock) !== null);
  // A click on the picture is an input event, and it is the dock's own driving state that
  // lets it through: this dock took the seat, so it is the one that types.
  const canvas = dock.querySelector('canvas')!;
  canvas.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: 1, clientY: 1}));
  await waitFor(() => writes.some((write) => write.url.endsWith('/input')));

  const input = writes.filter((write) => write.url.endsWith('/input')).at(-1)!;
  // The credential the take minted, and not the label every reader of the stream can see.
  expect(seatOf(input)).to.equal(SEAT_TOKEN);
  expect(seatOf(input)).to.not.equal(SEAT_LABEL);
  expect(input.headers['Idempotency-Key']).to.not.equal(undefined);
  const event = JSON.parse(input.body ?? '{}') as {type: string; sequence?: number};
  expect(event.type).to.equal('mouse');
  // Every event is numbered, so the page can drop one the wire overtook.
  expect(event.sequence).to.be.greaterThan(0);

  // The hand-back presents the same credential, and the answer keeps none.
  dock.querySelector('button')?.click();
  await waitFor(() => liveState(dock) === 'live');
  const controls = writes.filter((write) => write.url.endsWith('/control'));
  expect(seatOf(controls.at(-1)!)).to.equal(SEAT_TOKEN);
});

/** The lag the dock states, in seconds, or null while it states none.
 *
 * The readout is the only place the figure appears, so reading it back is what says whether
 * the number is the picture's age rather than the wall clock counted twice.
 */
function lagSeconds(dock: DlInspectorLive): number | null {
  const stated = /(\d+(?:\.\d+)?) s/.exec(dock.textContent ?? '');
  return stated === null ? null : Number(stated[1]);
}

it('says how far behind the page the stream is, and drops a stale frame', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.textContent).to.not.contain('behind the page');

  // A frame captured three seconds ago: the dock says so rather than showing it as current.
  const capturedAt = Date.now() / 1000 - 3;
  socket.frame(await jpeg(), 5, capturedAt);
  await waitFor(() => lagSeconds(dock) !== null);
  const drawn = lagSeconds(dock)!;
  expect(drawn).to.be.at.least(3);
  expect(drawn).to.be.at.most(3.8);

  // And it goes on counting the age the frame already had, one second per second. A readout
  // that subtracted the capture stamp from the wall clock on every tick added the time since
  // the frame was drawn a second time, so a picture ten seconds old read as twenty.
  await new Promise((resolve) => { setTimeout(resolve, 1500); });
  const truth = Date.now() / 1000 - capturedAt;
  const stated = lagSeconds(dock)!;
  expect(stated).to.be.at.least(truth - 0.6);
  expect(stated).to.be.at.most(truth + 0.3);

  // A frame numbered below the one already painted is dropped: the stage keeps the newer
  // picture rather than going back in time, and the readout never gets younger.
  socket.frame(await jpeg(), 4, Date.now() / 1000);
  await dock.updateComplete;
  expect(lagSeconds(dock)!).to.be.at.least(stated);
});

it('does not call a page behind for being still, however old its picture is', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  // The state and the frame carry the same count, exactly as the server sends them: the
  // state first, then the frame it describes.
  socket.text(update({sequence: 3}));
  await waitFor(() => liveState(dock) === 'live');
  socket.frame(await jpeg(), 3);
  await waitFor(() => painted(dock) !== null);

  // The frame was fresh when it was drawn and the session has delivered nothing newer since,
  // so the picture is the newest there is: the page is still, which is not the stream being
  // behind it. Counting the wall clock on a quiet page used to say "behind by 48 s" here.
  await new Promise((resolve) => { setTimeout(resolve, 1300); });
  expect(lagSeconds(dock)).to.equal(null);
  expect(dock.textContent).to.not.contain('behind the page');
  expect(dock.textContent).to.contain('Watching the agent’s browser');
});

it('stops calling a late frame behind once a quiet moment says nothing newer exists', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({sequence: 5}));
  await waitFor(() => liveState(dock) === 'live');

  // A frame that was already three seconds old when it arrived is the delivery being late,
  // and the dock says so while the stream has nothing newer to give.
  socket.frame(await jpeg(), 5, Date.now() / 1000 - 3);
  await waitFor(() => lagSeconds(dock) !== null);
  expect(dock.textContent).to.contain('behind the page');

  // The session's heartbeat repeats the number of the frame the dock drew and nothing
  // newer: the page has not changed since that capture, so what is on the stage is the
  // newest picture there is and a still page is not a stream that is behind it.
  socket.beat(Date.now(), 5);
  await waitFor(() => lagSeconds(dock) === null);
  expect(dock.textContent).to.not.contain('behind the page');
});

it('takes control and hands it back, and stops the stream when the dock closes', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  dock.querySelector('button')?.click();
  await waitFor(() => writes.some((write) => write.url.endsWith('/control')));
  const control = writes.filter((write) => write.url.endsWith('/control')).at(-1);
  expect(JSON.parse(control?.body ?? '{}')).to.deep.equal({state: 'human'});
  // A take carries no credential: the answer is what mints it, and the seat is one only
  // because nothing a reader can see holds one.
  expect(seatOf(control!)).to.equal('');
  await waitFor(() => liveState(dock) === 'taken');

  // Closing the dock unsubscribes: the socket it opened is closed, and the seat it took is
  // handed back, under the credential the take answered with, rather than left to refuse
  // the agent's next call for nothing.
  dock.active = false;
  await dock.updateComplete;
  await waitFor(() => socket.closedWith !== null);
  const controls = writes.filter((write) => write.url.endsWith('/control'));
  expect(controls.length).to.be.greaterThan(1);
  expect(JSON.parse(controls.at(-1)?.body ?? '{}')).to.deep.equal({state: 'agent'});
  expect(seatOf(controls.at(-1)!)).to.equal(SEAT_TOKEN);
});

it('opens the socket again when it drops, and draws what the new one sends', async () => {
  const dock = await mounted();
  const first = await live(dock);
  first.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  first.drop(1006);
  await waitFor(() => sockets.length > 1, 2000);
  const second = await live(dock);
  expect(second).to.not.equal(first);
  second.frame(await jpeg(), 2);
  await waitFor(() => painted(dock) !== null);
  expect(liveState(dock)).to.equal('live');
});

it('pauses when the page is hidden, keeping the last frame, and resumes when it is shown', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  socket.frame(await jpeg(), 1);
  await waitFor(() => painted(dock) !== null);
  await waitFor(() => liveState(dock) === 'live');

  pageVisible(false);
  await new Promise((resolve) => { setTimeout(resolve, 40); });
  await waitFor(() => liveState(dock) === 'paused');
  // The page is nobody looking: the socket is let go and nothing is opened in its place,
  // so no frames are asked for while there is no screen to put them on.
  await waitFor(() => socket.closedWith !== null);
  await new Promise((resolve) => { setTimeout(resolve, 20); });
  expect(sockets.length).to.equal(1);
  // The dock is paused, not ended: the last frame is still the picture on the stage.
  expect(painted(dock)).to.not.equal(null);
  expect(dock.textContent).to.contain('paused');
  expect(dock.textContent).to.not.contain('ended');

  pageVisible(true);
  await waitFor(() => sockets.length === 2, 2000);
  const resumed = sockets.at(-1)!;
  resumed.text(update({}));
  resumed.frame(await jpeg(), 2);
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.textContent).to.contain('Watching the agent’s browser');
});

it('stops listening to the page once the dock has left it', async () => {
  const dock = await mounted();
  await live(dock);
  dock.remove();
  const opened = sockets.length;

  pageVisible(false);
  pageVisible(true);
  window.dispatchEvent(new Event('pagehide'));
  window.dispatchEvent(new Event('resume'));
  await new Promise((resolve) => { setTimeout(resolve, 40); });

  expect(sockets.length, 'a dock that had left the page opened a socket for it').to.equal(opened);
});

it('keeps the driving seat through a pause, and gives it up only when the dock does', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  // This dock takes the seat, and the stream names it as the driver.
  dock.querySelector('button')?.click();
  await waitFor(() => liveState(dock) === 'taken');
  socket.text(update({control: 'human', driver: SEAT_LABEL}));
  await waitFor(() => liveState(dock) === 'taken');
  const taken = writes.filter((write) => write.url.endsWith('/control')).length;

  // Hiding the tab is nobody looking, not a hand-back: the socket goes and the seat stays.
  // Nobody asked for it back, and a seat released behind the reader's back spends a
  // credential they still expect to drive with — the session's own idle release is the
  // place that decision belongs.
  pageVisible(false);
  await waitFor(() => liveState(dock) === 'paused');
  await waitFor(() => socket.closedWith !== null);
  await new Promise((resolve) => { setTimeout(resolve, 30); });
  expect(writes.filter((write) => write.url.endsWith('/control')).length).to.equal(taken);

  // Shown again, the seat is still this dock's: the resumed stream names the same label,
  // and what drives the page is the credential the take answered with.
  pageVisible(true);
  await waitFor(() => sockets.length === 2, 2000);
  const resumed = sockets.at(-1)!;
  resumed.text(update({control: 'human', driver: SEAT_LABEL}));
  resumed.frame(await jpeg(), 2);
  await waitFor(() => liveState(dock) === 'taken');
  await waitFor(() => painted(dock) !== null);
  const canvas = dock.querySelector('canvas')!;
  canvas.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: 1, clientY: 1}));
  await waitFor(() => writes.some((write) => write.url.endsWith('/input')));
  expect(seatOf(writes.filter((write) => write.url.endsWith('/input')).at(-1)!))
    .to.equal(SEAT_TOKEN);

  // Closing the dock is what hands it back, under that same credential.
  dock.active = false;
  await dock.updateComplete;
  await waitFor(() => writes.filter((write) => write.url.endsWith('/control')).length > taken);
  const controls = writes.filter((write) => write.url.endsWith('/control'));
  expect(JSON.parse(controls.at(-1)?.body ?? '{}')).to.deep.equal({state: 'agent'});
  expect(seatOf(controls.at(-1)!)).to.equal(SEAT_TOKEN);
});

it('drops a seat the stream no longer names, so a pause cannot leave a stale credential', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');
  dock.querySelector('button')?.click();
  await waitFor(() => liveState(dock) === 'taken');
  socket.text(update({control: 'human', driver: SEAT_LABEL}));
  await waitFor(() => liveState(dock) === 'taken');

  // The session released the seat while this dock was hidden — its idle release, or the
  // session ending — and another viewer has it now. Holding on to a credential the server
  // has spent would leave the button reading "hand back" for a seat that is not there and
  // refuse every write, so the label the update names is what says whether it is still ours.
  socket.text(update({control: 'human', driver: 'viewer-someone-else'}));
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.textContent).to.contain('Another viewer is driving this page');
  expect(dock.querySelector<HTMLButtonElement>('button')!.disabled).to.equal(true);
});

it('judges the rung after a pause against the deployment’s own budget', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  // The deployment configures 24 fps at quality 70, and the session steps itself down to
  // keep up: 15 and 45 is a rung, not the ceiling.
  socket.text(update({frames_per_second: 24, jpeg_quality: 70, max_width: 1280}));
  await waitFor(() => liveState(dock) === 'live');
  socket.text(update({frames_per_second: 15, jpeg_quality: 45, max_width: 1280}));
  await waitFor(() => dock.textContent?.includes('stepped down') === true);

  // Hidden and shown again, the resumed socket opens on the rung the session is already
  // serving. The base is the session's, not that socket's: re-learning it from the first
  // update after a hide would make a stepped-down rung the deployment's configured ceiling,
  // and the dock would stop reporting the step-down it is looking at.
  pageVisible(false);
  await waitFor(() => liveState(dock) === 'paused');
  await waitFor(() => socket.closedWith !== null);
  pageVisible(true);
  await waitFor(() => sockets.length === 2, 2000);
  const resumed = sockets.at(-1)!;
  resumed.text(update({frames_per_second: 15, jpeg_quality: 45, max_width: 1280}));
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.rung).to.deep.equal([15, 45]);
  expect(dock.base).to.deep.equal([24, 70]);
  expect(dock.textContent).to.contain('stepped down');
});

it('hides and shows the page again and again, each show on a socket of its own', async () => {
  const dock = await mounted();
  const first = await live(dock);
  first.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  // Hide, show, hide, show. The bug this pins closed the *new* socket as soon as it opened:
  // the promise the previous hide had resolved was still carrying the previous socket's
  // conceal, so the pair that resumed belonged to a socket that was already gone.
  for (let round = 2; round <= 4; round += 1) {
    const before = sockets.length;
    pageVisible(false);
    await waitFor(() => liveState(dock) === 'paused');
    await new Promise((resolve) => { setTimeout(resolve, 20); });
    expect(sockets.length).to.equal(before, 'nothing opens while nobody is looking');

    pageVisible(true);
    await waitFor(() => sockets.length === before + 1, 2000);
    const socket = sockets.at(-1)!;
    expect(socket.closedWith).to.equal(null);
    socket.text(update({control: 'agent', driver: null, attached: true}));
    socket.frame(await jpeg(), round);
    await waitFor(() => liveState(dock) === 'live');
    expect(dock.textContent).to.contain('Watching the agent’s browser');
  }
});

it('reports why the picture got lighter when the session steps itself down', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({frames_per_second: 24, jpeg_quality: 70, max_width: 1280}));
  await new Promise((resolve) => { setTimeout(resolve, 50); });
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.textContent).to.not.contain('stepped down');

  socket.text(update({frames_per_second: 24, jpeg_quality: 45, max_width: 1280}));
  await waitFor(() => dock.textContent?.includes('stepped down') === true);
  expect(liveState(dock)).to.equal('live');
});

it('judges stepped-down against the budget the deployment configured', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  // A deployment whose own ceiling is 12 fps at quality 50 is serving all of it: the dock
  // must not call that stepped-down just because the ladder's old numbers were 24 and 70.
  socket.text(update({frames_per_second: 12, jpeg_quality: 50, max_width: 1280}));
  await waitFor(() => liveState(dock) === 'live');
  expect(dock.textContent).to.not.contain('stepped down');

  // One rung below the base it was told *is* a step-down.
  socket.text(update({frames_per_second: 12, jpeg_quality: 38, max_width: 1280}));
  await waitFor(() => dock.textContent?.includes('stepped down') === true);
});

it('leaves a second viewer read-only, naming whoever does drive', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({control: 'human', driver: 'viewer-someone-else'}));
  await waitFor(() => liveState(dock) === 'live');

  const button = dock.querySelector<HTMLButtonElement>('button')!;
  expect(button.disabled).to.equal(true);
  expect(dock.textContent).to.contain('Another viewer is driving this page');

  button.click();
  await dock.updateComplete;
  expect(writes.some((write) => write.url.endsWith('/control'))).to.equal(false);

  // When that viewer hands the seat back, this one may take it again.
  socket.text(update({control: 'agent', driver: null}));
  await waitFor(() => !dock.querySelector<HTMLButtonElement>('button')!.disabled);
  expect(dock.textContent).to.contain('Watching the agent’s browser');
});

it('never claims the seat when another viewer took it first', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');

  // The seat is taken between this dock's click and the server reading it: the refusal is
  // not a broken preview, and the update that follows is what says who drives.
  window.fetch = (async (input: RequestInfo | URL) => {
    if (String(input).endsWith('/control')) {
      return new Response(JSON.stringify({kind: 'taken_over', message: 'taken'}), {
        status: 409, headers: {'Content-Type': 'application/json'},
      });
    }
    return new Response('', {status: 204});
  }) as typeof fetch;
  dock.querySelector('button')?.click();
  await dock.updateComplete;
  await waitFor(() => dock.textContent?.includes('Watching') === true);
  expect(dock.textContent).to.not.contain('unavailable');
  expect(liveState(dock)).to.equal('live');
});

it('says why there is nothing to watch when the socket is refused', async () => {
  const dock = await mounted();
  const socket = await live(dock);

  // The server closes an unservable Run before any frame, naming the browser failure in
  // the close reason; the dock says which one instead of showing a blank stage.
  socket.drop(4409, 'not_configured');
  await waitFor(() => (dock.textContent ?? '').includes('This Run is not using a browser'), 2000);
  expect(liveState(dock)).to.equal('idle');
});

it('falls back to the Run’s last screenshot when nothing is attached', async () => {
  const dock = await mounted();
  const drawn = drawnFallback(dock);
  const socket = await live(dock);
  socket.text(update({attached: false}));
  await waitFor(() => (dock.textContent ?? '').includes('No page is open right now'));

  expect(drawn.sources.some((src) => src.endsWith('/screenshot'))).to.equal(true);
  expect(drawn.captions).to.contain('Last screenshot of this Run');
  expect(liveState(dock)).to.equal('live');

  // The live page coming back replaces the picture, rather than sitting beside it.
  socket.text(update({attached: true}));
  socket.frame(await jpeg(), 3);
  await waitFor(() => painted(dock) !== null);
  expect(dock.querySelectorAll('img')).to.have.length(0);
});

it('falls back to the last screenshot when the session has ended', async () => {
  const dock = await mounted();
  const socket = await live(dock);
  socket.text(update({}));
  await waitFor(() => liveState(dock) === 'live');
  const drawn = drawnFallback(dock);
  socket.text(update({ended: 'closed'}));
  await waitFor(() => liveState(dock) === 'ended');

  await waitFor(() => drawn.sources.length > 0);
  expect(drawn.sources.some((src) => src.endsWith('/screenshot'))).to.equal(true);
  expect(dock.textContent).to.contain('This live session has ended');
});

it('keeps a text-only stage when the last screenshot cannot be loaded', async () => {
  const dock = await mounted();
  const drawn = drawnFallback(dock);
  const socket = await live(dock);
  socket.text(update({attached: false}));
  await waitFor(() => drawn.sources.some((src) => src.endsWith('/screenshot')));

  // A Run that kept no screenshot answers 404, and the browser reports that as an error on
  // the image: the dock hides it rather than leaving a broken picture on the stage.
  await waitFor(() => dock.querySelector('img') === null);
  expect(dock.textContent).to.contain('No page is open right now');
});

/** Every stage picture the dock draws, recorded as it is drawn.
 *
 * The screenshot URL is answered by the test server itself — an image load never goes
 * through `window.fetch` — so the browser's own 404 can remove the picture before a poll
 * sees it. A test reads what was drawn from here instead of racing that error.
 */
function drawnFallback(dock: DlInspectorLive): {sources: string[]; captions: string[]} {
  const seen = {sources: [] as string[], captions: [] as string[]};
  const observer = new MutationObserver(() => {
    const src = dock.querySelector('img')?.getAttribute('src') ?? '';
    if (src !== '' && !seen.sources.includes(src)) seen.sources.push(src);
    const caption = dock.querySelector('figcaption')?.textContent?.trim() ?? '';
    if (caption !== '' && !seen.captions.includes(caption)) seen.captions.push(caption);
  });
  observer.observe(dock, {attributes: true, childList: true, subtree: true});
  return seen;
}

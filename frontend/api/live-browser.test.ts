// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The frames socket: what the client opens, what it sends, and when it gives up.

 * The socket is a stub the test drives, so what is asserted here is the protocol this
 * module speaks: the URL it opens, the frame header it decodes, the text opcode as state,
 * the heartbeat it reads as keepalive and nothing else, and the reconnect policy a dropped
 * socket meets — one that outlasts any number of drops while the dock is open, and that a
 * deliberate close code stops. A page nobody is looking at is the one thing that ends a
 * watch without ending the dock: the socket goes, nothing is scheduled behind it, and the
 * watch resumes only when `canWatch` says somebody is looking again.
 */

import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FRAME_HEADER_BYTES,
  LIVE_CLOSE_NO_RUN,
  LIVE_CLOSE_NO_SESSION,
  LIVE_CLOSE_UNAUTHENTICATED,
  LiveBrowserRefusal,
  decodeLiveFrame,
  sendLiveInput,
  setLiveControl,
  watchLiveBrowser,
  type LiveFrame,
  type LiveHeartbeat,
  type LiveUpdate,
} from './live-browser.ts';

/** One frames socket, as the pages sees it, with the test on the server's side. */
class StubSocket {
  static readonly sockets: StubSocket[] = [];
  static readonly closed: StubSocket[] = [];

  readonly url: string;
  binaryType = 'blob';
  closedWith: {code: number; reason: string} | null = null;
  onmessage: ((event: {data: unknown}) => void) | null = null;
  onclose: ((event: {code: number; reason: string}) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    StubSocket.sockets.push(this);
  }

  close(code = 1000, reason = ''): void {
    if (this.closedWith !== null) return;
    this.closedWith = {code, reason};
    StubSocket.closed.push(this);
    this.onclose?.({code, reason});
  }

  /** One text message: the state a frame arrived in. */
  text(payload: Record<string, unknown>): void {
    this.onmessage?.({data: JSON.stringify({at: 1, ...payload})});
  }

  /** One heartbeat text message: the moment the server sent it, and the newest frame. */
  beat(at: number, sequence = 0): void {
    this.onmessage?.({data: JSON.stringify({at, sequence})});
  }

  /** One binary message: a frame's own JPEG bytes. */
  binary(bytes: ArrayBuffer): void {
    this.onmessage?.({data: bytes});
  }

  /** One binary message as the server really sends it: the header, then the JPEG. */
  frame(jpeg: Uint8Array, sequence: bigint, capturedAt: number): void {
    const message = new Uint8Array(FRAME_HEADER_BYTES + jpeg.length);
    const header = new DataView(message.buffer);
    header.setBigUint64(0, sequence);
    header.setFloat64(8, capturedAt);
    message.set(jpeg, FRAME_HEADER_BYTES);
    this.binary(message.buffer);
  }

  /** The server, or the network, closing the socket. */
  drop(code: number, reason = ''): void {
    this.onclose?.({code, reason});
  }
}

function install(): void {
  StubSocket.sockets.length = 0;
  StubSocket.closed.length = 0;
  (globalThis as {WebSocket?: unknown}).WebSocket = StubSocket;
  (globalThis as {location?: unknown}).location = {protocol: 'http:', host: 'example.test'};
}

/** The socket a watcher opened at ``index``, waiting out the backoff in front of it.

 * Reconnect delays double, so the tenth socket of a test is seconds behind the first on a
 * slow machine: waiting for it is what makes that a wait rather than a race.
 */
async function opened(index: number, timeoutMs = 3000): Promise<StubSocket> {
  for (let waited = 0; waited < timeoutMs; waited += 5) {
    const socket = StubSocket.sockets[index];
    if (socket !== undefined) return socket;
    await new Promise((resolve) => { setTimeout(resolve, 5); });
  }
  throw new Error(`the watcher never opened socket ${index}`);
}

/** Wait for a condition the module under test reaches asynchronously. */
async function waitFor(condition: () => boolean, timeoutMs = 1000): Promise<void> {
  for (let waited = 0; waited < timeoutMs; waited += 2) {
    if (condition()) return;
    await new Promise((resolve) => { setTimeout(resolve, 2); });
  }
  throw new Error('the watcher never reached the state the test waited for');
}

/** A watcher that records what it was handed. */
function watcher(): {updates: LiveUpdate[]; frames: LiveFrame[]; beats: LiveHeartbeat[]} {
  return {updates: [], frames: [], beats: []};
}

function handlers(seen: {
  updates: LiveUpdate[];
  frames: LiveFrame[];
  beats: LiveHeartbeat[];
}) {
  return {
    update: (update: LiveUpdate) => { seen.updates.push(update); },
    frame: (frame: LiveFrame) => { seen.frames.push(frame); },
    quiet: (heartbeat: LiveHeartbeat) => { seen.beats.push(heartbeat); },
  };
}

/** The JPEG inside one delivered frame, as the header's own width says. */
function jpegOf(frame: LiveFrame): number[] {
  return [...new Uint8Array(frame.jpeg)];
}

test('frames arrive as bytes and updates as state, on one socket', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal);
  const socket = await opened(0);
  assert.equal(socket.url, 'ws://example.test/web/api/runs/run-1/browser/live/frames');
  // The binary opcode is read as bytes, so nothing has to unwrap a Blob afterwards.
  assert.equal(socket.binaryType, 'arraybuffer');

  socket.text({
    control: 'human', driver: 'viewer-1', attached: true, ended: '', sequence: 7,
    frames_per_second: 15, jpeg_quality: 45, max_width: 1280,
  });
  socket.frame(new Uint8Array([0xff, 0xd8, 0x6a]), 7n, 1_700_000_000.5);
  // The dock closing is what ends the watch; a socket that drops by itself is retried.
  controller.abort();
  await watching;

  assert.deepEqual(seen.updates, [
    {
      control: 'human', driver: 'viewer-1', attached: true, ended: '', at: 1,
      sequence: 7, framesPerSecond: 15, jpegQuality: 45, maxWidth: 1280,
    },
  ]);
  // The header states which frame it is and when the page was captured; the JPEG behind it
  // is handed on untouched, with no base64, no re-encoding and no copy of a string.
  assert.equal(seen.frames.length, 1);
  assert.equal(seen.frames[0]?.sequence, 7n);
  assert.equal(seen.frames[0]?.capturedAt, 1_700_000_000.5);
  assert.deepEqual(jpegOf(seen.frames[0]!), [0xff, 0xd8, 0x6a]);
});

test('a frame is decoded from its header, and one without a picture is not a frame', () => {
  const message = new Uint8Array(FRAME_HEADER_BYTES + 2);
  const header = new DataView(message.buffer);
  header.setBigUint64(0, 42n);
  header.setFloat64(8, 1_700_000_001.25);
  message.set(new Uint8Array([0xff, 0xd8]), FRAME_HEADER_BYTES);

  const frame = decodeLiveFrame(message.buffer);
  assert.equal(frame?.sequence, 42n);
  assert.equal(frame?.capturedAt, 1_700_000_001.25);
  assert.deepEqual(jpegOf(frame!), [0xff, 0xd8]);

  // A header with nothing behind it, a message too short to hold one, and a capture time
  // that is not a time are all dropped: there is no picture to put on the stage.
  assert.equal(decodeLiveFrame(new Uint8Array(FRAME_HEADER_BYTES).buffer), null);
  assert.equal(decodeLiveFrame(new Uint8Array(4).buffer), null);
  const timeless = new Uint8Array(FRAME_HEADER_BYTES + 2).buffer;
  new DataView(timeless).setFloat64(8, Number.NaN);
  assert.equal(decodeLiveFrame(timeless), null);
});

test('a binary message that is not a frame is not handed to the watcher', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal);
  const socket = await opened(0);

  socket.binary(new Uint8Array([1, 2, 3]).buffer);
  controller.abort();
  await watching;

  assert.deepEqual(seen.frames, []);
});

test('a state message without a rung is read, and reports no budget', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal, {
    reconnectDelayMs: 1,
  });
  const socket = await opened(0);

  socket.text({control: 'agent', driver: null, attached: true, ended: '', at: 4});
  controller.abort();
  await watching;

  assert.deepEqual(seen.updates, [
    {
      control: 'agent', driver: null, attached: true, ended: '', at: 4, sequence: null,
      framesPerSecond: null, jpegQuality: null, maxWidth: null,
    },
  ]);
});

test('a heartbeat is a quiet moment: no state, no repaint, and the newest frame restated', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal, {
    reconnectDelayMs: 1,
  });
  const socket = await opened(0);
  // A quiet page sends heartbeats and nothing else, for as long as it is quiet.
  socket.beat(1_700_000_000_000, 4);
  socket.beat(1_700_000_030_000, 4);
  await new Promise((resolve) => { setTimeout(resolve, 5); });

  // No state was applied and nothing was repainted; what the watcher heard is the one fact
  // a heartbeat has to say, which is that nothing newer than frame 4 has been delivered.
  assert.deepEqual(seen.updates, []);
  assert.deepEqual(seen.frames, []);
  assert.deepEqual(seen.beats, [
    {at: 1_700_000_000_000, sequence: 4},
    {at: 1_700_000_030_000, sequence: 4},
  ]);
  assert.equal(socket.closedWith, null);
  assert.equal(StubSocket.sockets.length, 1);

  // The socket is the same live one afterwards: state still arrives on it.
  socket.text({control: 'human', driver: 'viewer-1', attached: true, ended: ''});
  controller.abort();
  await watching;

  assert.deepEqual(seen.updates, [
    {
      control: 'human', driver: 'viewer-1', attached: true, ended: '', at: 1, sequence: null,
      framesPerSecond: null, jpegQuality: null, maxWidth: null,
    },
  ]);
});

test('a dropped socket is opened again, past the ceiling that used to end the dock', async () => {
  install();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
    reconnectDelayMs: 1,
  });
  try {
    // The old policy gave up after five failed reconnects; nine drops are not a reason to
    // end a dock that is still open, so a tenth socket is opened like the first.
    for (let drop = 0; drop < 9; drop += 1) {
      const socket = await opened(drop);
      socket.drop(1006);
    }
    const tenth = await opened(9);
    assert.equal(StubSocket.sockets.length, 10);
    assert.equal(tenth.closedWith, null);
  } finally {
    controller.abort();
    await watching;
  }
  // Closing the dock is what ends the watch, and no eleventh socket follows it.
  await new Promise((resolve) => { setTimeout(resolve, 5); });
  assert.equal(StubSocket.sockets.length, 10);
});

test('a deliberate close code is reported rather than retried', async () => {
  install();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
    reconnectDelayMs: 1,
  });
  (await opened(0)).drop(LIVE_CLOSE_NO_SESSION, 'unreachable');

  await assert.rejects(watching, (error: unknown) => (
    error instanceof LiveBrowserRefusal
    && error.status === LIVE_CLOSE_NO_SESSION
    && error.reason === 'unreachable'
  ));
  assert.equal(StubSocket.sockets.length, 1);
});

test('an unauthenticated or unknown Run is a refusal, not a session', async () => {
  for (const [code, reason] of [
    [LIVE_CLOSE_UNAUTHENTICATED, 'Authentication required'],
    [LIVE_CLOSE_NO_RUN, 'Run not found'],
  ] as const) {
    install();
    const controller = new AbortController();
    const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
      reconnectDelayMs: 1,
    });
    (await opened(0)).drop(code, reason);
    await assert.rejects(watching, (error: unknown) => (
      error instanceof LiveBrowserRefusal && error.status === code && error.reason === 'unknown'
    ));
    assert.equal(StubSocket.sockets.length, 1);
  }
});

test('a session that ends is a normal end: the socket closes and nothing reopens', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal, {
    reconnectDelayMs: 1,
  });
  const socket = await opened(0);
  socket.text({control: 'agent', driver: null, attached: false, ended: 'expired'});

  await watching;
  assert.equal(socket.closedWith?.code, 1000);
  assert.equal(StubSocket.sockets.length, 1);
  assert.equal(seen.updates[0]?.ended, 'expired');
});

/** A dock's own visibility, asked as `canWatch`: hidden means nobody is looking. */
function visibility() {
  let looking = true;
  let resolveWaiters: (() => void)[] = [];
  return {
    options: {
      // The dock answers the question afresh every time it is asked, which is the whole of
      // the pause: there is no promise belonging to a socket that has since closed.
      canWatch: () => (looking ? Promise.resolve(true) : new Promise<boolean>((resolve) => {
        resolveWaiters.push(() => { resolve(true); });
      })),
    },
    hide: () => { looking = false; },
    show: () => {
      looking = true;
      const waiters = resolveWaiters;
      resolveWaiters = [];
      for (const waiter of waiters) waiter();
    },
  };
}

test('a watch its dock closes while the page is hidden schedules nothing behind it', async () => {
  install();
  const controller = new AbortController();
  const page = visibility();
  const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
    reconnectDelayMs: 1,
    ...page.options,
  });
  const socket = await opened(0);

  page.hide();
  // The socket a hidden page already has is closed by the dock that owns it, which aborts
  // the watch and is asserted where that dock lives (`inspector-live.browser.test.ts`: a
  // hidden page closes its socket and opens no other). What belongs to this module is what
  // follows the deliberate close: the watch ends here, and nothing is scheduled behind it.
  controller.abort();
  await watching;
  await new Promise((resolve) => { setTimeout(resolve, 20); });
  assert.equal(StubSocket.sockets.length, 1);
  assert.equal(socket.closedWith?.code, 1000);
});

test('showing the page again opens a socket, and hiding it again pauses it again', async () => {
  install();
  const seen = watcher();
  const controller = new AbortController();
  const page = visibility();
  const watching = watchLiveBrowser('run-1', handlers(seen), controller.signal, {
    reconnectDelayMs: 1,
    ...page.options,
  });
  const first = await opened(0);
  first.text({control: 'agent', driver: null, attached: true, ended: '', at: 1});

  // Hide, show, hide, show: each show opens a socket of its own, and no socket of an
  // earlier round is closed by a promise the round before it left behind.
  for (let round = 0; round < 2; round += 1) {
    page.hide();
    const openedInRound = StubSocket.sockets.at(-1)!;
    openedInRound.drop(1000, 'page hidden');
    await new Promise((resolve) => { setTimeout(resolve, 10); });
    const before = StubSocket.sockets.length;
    assert.equal(before, round + 1, 'nothing opens while nobody is looking');

    page.show();
    const next = await opened(round + 1);
    assert.equal(next.closedWith, null);
    next.text({
      control: 'agent', driver: null, attached: true, ended: '', at: round + 2,
    });
  }

  await waitFor(() => seen.updates.length === 3);
  assert.equal(StubSocket.sockets.length, 3);
  controller.abort();
  await watching;
});

test('a dock that closes while the page is hidden ends the watch, not just the socket', async () => {
  install();
  const controller = new AbortController();
  const page = visibility();
  const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
    reconnectDelayMs: 1,
    ...page.options,
  });
  await opened(0);

  page.hide();
  controller.abort();
  await watching;

  await new Promise((resolve) => { setTimeout(resolve, 20); });
  assert.equal(StubSocket.sockets.length, 1);
});

test('closing the dock closes the socket and stops reconnecting', async () => {
  install();
  const controller = new AbortController();
  const watching = watchLiveBrowser('run-1', handlers(watcher()), controller.signal, {
    reconnectDelayMs: 1,
  });
  const socket = await opened(0);

  controller.abort();
  await watching;

  assert.equal(socket.closedWith?.code, 1000);
  // The close is the reader leaving: no second socket is opened after it.
  await new Promise((resolve) => { setTimeout(resolve, 5); });
  assert.equal(StubSocket.sockets.length, 1);
});

test('a refusal over HTTP names a reason this page knows, and anything else is unknown', async () => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  Object.defineProperty(globalThis, 'document', {configurable: true, value: {cookie: ''}});
  try {
    const refusedWith = async (body: BodyInit | null, status = 409): Promise<LiveBrowserRefusal> => {
      globalThis.fetch = async () => new Response(body, {status});
      const signal = new AbortController().signal;
      return await setLiveControl('run-1', 'human', null, signal).then(
        () => { throw new Error('the refusal was taken for a seat'); },
        (error: unknown) => error as LiveBrowserRefusal,
      );
    };
    for (const reason of ['not_configured', 'unreachable', 'no_page', 'taken_over', 'action_failed', 'expired']) {
      const error = await refusedWith(JSON.stringify({kind: reason, message: 'Refused.'}));
      assert.ok(error instanceof LiveBrowserRefusal);
      assert.equal(error.reason, reason);
      assert.equal(error.status, 409);
    }
    // A word the server adds later is no reason this page can act on, and a body that is not an
    // envelope names none; neither is taken for a reason it has not been told.
    for (const body of [
      JSON.stringify({kind: 'a_reason_added_later'}),
      JSON.stringify({kind: 7}),
      JSON.stringify({kind: 'constructor'}),
      JSON.stringify(['taken_over']),
      JSON.stringify('taken_over'),
      '<html>Bad gateway</html>',
      null,
    ]) {
      const error = await refusedWith(body, 502);
      assert.equal(error.reason, 'unknown', `${String(body)} named a reason`);
      assert.equal(error.status, 502);
    }

    globalThis.fetch = async () => Response.json({kind: 'taken_over', message: 'No.'}, {status: 403});
    await assert.rejects(
      sendLiveInput('run-1', {type: 'key', key: 'a'}, 'seat-token', new AbortController().signal),
      (error: unknown) => error instanceof LiveBrowserRefusal && error.reason === 'taken_over',
    );
  } finally {
    globalThis.fetch = originalFetch;
    Object.defineProperty(globalThis, 'document', {configurable: true, value: originalDocument});
  }
});

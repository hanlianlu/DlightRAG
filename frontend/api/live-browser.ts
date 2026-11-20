// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The live browser preview of one Run, and the co-drive that follows it.

 * One WebSocket per open dock carries the read stream: the binary opcode is the newest
 * frame, exactly the JPEG bytes the page's browser encoded, and the text opcode is the
 * state that frame arrived in. Nothing here decodes or re-encodes a frame, and nothing
 * wraps one in text. Input events and control handovers are ordinary same-origin POSTs
 * with the CSRF header every write carries, and each input event carries its own
 * idempotency key so a retry after a timeout runs once: the socket carries no write.
 *
 * A page that is not repainting produces no frame and no state, so the server also sends
 * a heartbeat over the text opcode — `{"at": <epoch ms>, "sequence": <newest frame>}` — which
 * holds a proxied socket open across a quiet page and repeats the count the last state
 * message carried. This module applies no state for one and repaints nothing; what it does
 * is hand the count to the watcher, because an unchanged sequence is a session that has
 * delivered nothing newer, and that is the difference between a page that has not changed
 * and a frame the reader has not drawn yet. A socket that drops for a reason other than a
 * deliberate close code is opened again, forever while the dock is open, backing off up to
 * a cap: the dock's own closing is the only thing that ends the watch, so a proxy's hiccup
 * or a server restart costs the reader a pause, not the dock.
 *
 * The one thing that ends a watch without ending the dock is a page nobody is looking at.
 * A hidden page is nobody looking: the dock that holds the open socket closes it when the
 * page goes away, and this module reopens nothing while it stays hidden, because the whole
 * cost of this stream is frames for a screen that is not there — the server stops the
 * page's capture when its last reader leaves, so closing the socket is the whole of the
 * protocol. What a caller whose page comes back does is resume the watch; until it does,
 * this module waits rather than backing off. The caller states that by answering
 * `canWatch`, which is asked before every socket rather than handed over as a promise:
 * what is hidden is a fact about the page now, not a future one socket owns.
 *
 * Every frame is a fixed header and then the JPEG the page encoded: the session's own
 * sequence number, then when the page was captured. A frame older than the one this dock
 * has already drawn is dropped rather than painted over it; the distance between the
 * capture time and the clock is how old the picture already was when it was drawn, and the
 * newest sequence the session has delivered is how much newer that picture could have been.
 *
 * A browser cannot set a header on an upgrade, so the handshake is authenticated by the
 * session cookie and by the `Origin` the browser sends on its own; the server closes a
 * socket it will not serve with a deliberate code, which is what this module turns back
 * into a refusal rather than reconnecting into the same wall.
 *
 * One seat holds one credential, and the stream never carries it. Taking the seat POSTs to
 * `/control` and the *answer* to that one request carries the token; every later write of
 * that seat presents it in `X-Live-Seat`. What the text updates carry is a display label
 * saying the seat is held, which is what tells every other dock it is an observer — and
 * nothing a reader can drive with, so a second reader of the same Run that replays every
 * byte of the broadcast still holds no credential at all.
 */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseData} from './sse.ts';
import {noteRefusal, parseWire, refusalBody, refusalEnvelope} from './wire.ts';

type LiveControlState = 'agent' | 'human';

/** One text update: the state the newest frame arrived in, never the frame itself. */
export interface LiveUpdate {
  control: LiveControlState;
  /** The seat holder's display label, null while the agent drives. Never a credential:
   * it is what every reader of the Run is told, so it is public by construction. */
  driver: string | null;
  attached: boolean;
  ended: string;
  /** The newest frame the session has delivered, as of this update, or null when the
   * message did not say.
   *
   * The same count a frame's header carries, and never behind it: a reader that compares
   * this with the newest frame it has drawn knows whether it is looking at a page that has
   * stopped changing — nothing newer was delivered — or at a stream it has not caught up
   * with. Null on a message from a session that names no count, which is read as before:
   * the number the dock states is the picture's age, and it cannot tell why.
   */
  sequence: number | null;
  /** What the session is delivering at now, and why the picture looks the way it does.
   *
   * A session whose viewers cannot keep up steps itself down, so the rung moves without
   * anybody asking. Null on a state message that carried no budget at all.
   */
  framesPerSecond: number | null;
  jpegQuality: number | null;
  maxWidth: number | null;
  at: number;
}

/** The newest frame the session has delivered, restated by a quiet moment.
 *
 * A heartbeat is the server saying that nothing — no frame, no state — has gone out since
 * the update whose sequence it repeats: a number unchanged from the frame a reader has
 * drawn is a page that has not changed, and one ahead of it is a frame the reader is behind.
 */
export interface LiveHeartbeat {
  /** The newest frame the session has delivered when this went out, and still is. */
  sequence: number;
  /** When the server sent it, as epoch milliseconds. */
  at: number;
}

/** What a watcher does with what the socket sends: a state update, or a frame. */
interface LiveWatcher {
  update(update: LiveUpdate): void;
  /** The newest frame, decoded: the JPEG, and what its header said about it.
   *
   * A frame whose sequence is at or below the last one painted is not delivered here at
   * all: it is older than what the dock already shows, and painting it would move the
   * picture backwards.
   */
  frame(frame: LiveFrame): void;
  /** A quiet moment: nothing newer than `sequence` has been delivered.
   *
   * A watcher that ignores this loses nothing but the distinction it carries: a page that
   * has stopped changing and a stream the reader has fallen behind look the same from the
   * frames alone, and this is the message that says which one it is.
   */
  quiet?(heartbeat: LiveHeartbeat): void;
}

/** How a watcher retries a socket that drops, and when it stops watching at all. */
interface LiveWatchOptions {
  /** The wait before the first reconnect; each further one doubles, up to the cap. */
  reconnectDelayMs?: number;
  /** Whether anybody is looking right now, asked again whenever the watch would go on.
   *
   * This is the whole of the pause: when it says no the socket is let go, and nothing is
   * opened in its place until it says yes — while the caller's `AbortSignal` still ends the
   * watch. There is deliberately no hidden/visible pair here: a promise belonging to a
   * socket that has since closed is exactly how a resuming watch came to close the socket
   * it had just opened.
   */
  canWatch?: () => Promise<boolean>;
}

/** Who drives the Run's browser now, and what the seat answered the caller that took it. */
interface LiveSeat {
  control: LiveControlState;
  driver: string | null;
  /** The credential, on the one answer that took the seat; null on every other answer. */
  token: string | null;
}

/** One frame as the socket delivered it: what the browser encoded, and when it was taken. */
export interface LiveFrame {
  /** The JPEG bytes, exactly as the page's browser encoded them. */
  jpeg: ArrayBuffer;
  /** The session's own count of the frames it delivered, oldest first, never repeated.
   *
   * BigInt because the header carries a full 64-bit counter; the server counts one session
   * and will never approach that, so comparing two of them is the whole of its use here.
   */
  sequence: bigint;
  /** When the page was captured, as epoch seconds on this machine's clock.
   *
   * CDP reports the moment in seconds since the epoch and the server hands it on as it
   * stands, so what arrives here is directly comparable with `Date.now()` and nothing has
   * to be calibrated client-side — no offset, and no unit to convert: the header carries the
   * browser's own reading of the clock this page counts in. A frame whose metadata carried
   * no reading is stamped with the moment it reached the server instead, which is a little
   * later than the capture and never earlier, so a dock is never told a picture is fresher
   * than it is.
   */
  capturedAt: number;
}

/** How many bytes every frame puts in front of its JPEG: a uint64 sequence number and a
 *
 * float64 capture time, both big-endian. That is the layout `docs/interfaces.md` states and
 * the server's own `FRAME_HEADER` writes; it is fixed width, so the frame and what
 * describes it cannot race the way a metadata message beside it could.
 */
export const FRAME_HEADER_BYTES = 16;

/** One binary message as a frame, or null when it is not one this dock can draw. */
export function decodeLiveFrame(bytes: ArrayBuffer): LiveFrame | null {
  if (bytes.byteLength <= FRAME_HEADER_BYTES) return null;
  const header = new DataView(bytes, 0, FRAME_HEADER_BYTES);
  const capturedAt = header.getFloat64(8);
  // A capture time the page never reported is not a frame worth painting or timing.
  if (!Number.isFinite(capturedAt)) return null;
  return {
    sequence: header.getBigUint64(0),
    capturedAt,
    jpeg: bytes.slice(FRAME_HEADER_BYTES),
  };
}

/** The reasons the server names for a live session it could not open or drive. */
const LIVE_REFUSAL_REASONS = [
  'not_configured',
  'unreachable',
  'no_page',
  'taken_over',
  'action_failed',
  'expired',
] as const;

/** Why a live session could not be opened or driven, as the server named it. */
type LiveRefusalReason = (typeof LIVE_REFUSAL_REASONS)[number] | 'unknown';

/** The reason a refusal names, or `unknown` for a word this page does not know. */
function liveRefusalReason(named: unknown): LiveRefusalReason {
  return LIVE_REFUSAL_REASONS.find((reason) => reason === named) ?? 'unknown';
}

/** One input event, exactly as the input route accepts it. */
export interface LiveInputEvent {
  type: 'mouse' | 'wheel' | 'key' | 'navigate';
  action?: string;
  x?: number;
  y?: number;
  button?: 'left' | 'right' | 'middle';
  click_count?: number;
  delta_x?: number;
  delta_y?: number;
  key?: string;
  text?: string;
  url?: string;
}

/** The close codes the frames socket is refused with, all before any frame. */
export const LIVE_CLOSE_UNAUTHENTICATED = 4401;
const LIVE_CLOSE_CROSS_ORIGIN = 4403;
export const LIVE_CLOSE_NO_RUN = 4404;
export const LIVE_CLOSE_NO_SESSION = 4409;

/** The first wait between reconnects; each further one doubles, up to the cap. */
const DEFAULT_RECONNECT_DELAY_MS = 500;
const MAX_RECONNECT_DELAY_MS = 8000;

const liveUpdate = v.object({
  control: v.picklist(['agent', 'human']),
  driver: v.nullable(v.string()),
  attached: v.boolean(),
  ended: v.string(),
  // The newest frame the session has delivered, which a quiet heartbeat repeats unchanged.
  // Optional as well as nullable: a message from a server that names no count is read for
  // everything else it carries, and a reader cannot yet tell a still page from a late frame.
  sequence: v.optional(v.nullable(v.number()), null),
  // The rung the session is on, which a session that steps itself down changes unasked.
  // Optional as well as nullable: a message from a session that named no budget carries
  // none of the three, and a reader is not a parser that refuses a preview over one.
  frames_per_second: v.optional(v.nullable(v.number()), null),
  jpeg_quality: v.optional(v.nullable(v.number()), null),
  max_width: v.optional(v.nullable(v.number()), null),
  at: v.number(),
});

/** One heartbeat: the moment it was sent, and the newest frame the session had delivered.
 *
 * It is strict because that is the whole of the distinction: a state update carries these
 * fields too, so only a message with nothing but them is a keepalive rather than state. A
 * heartbeat from a server that names no sequence is ignored like any other message this
 * page cannot read: a keepalive that says nothing costs the reader nothing.
 */
const liveHeartbeat = v.strictObject({at: v.number(), sequence: v.number()});

const liveSeat = v.object({
  control: v.picklist(['agent', 'human']),
  driver: v.nullable(v.string()),
  // The credential, on the one answer that minted it. Optional as well as nullable: a
  // hand-back has none to name, and an older answer without the field is still a seat.
  token: v.optional(v.nullable(v.string()), null),
});

/** A live session's refusal: the status or close code, and the reason the server named. */
export class LiveBrowserRefusal extends Error {
  readonly status: number;
  readonly reason: LiveRefusalReason;

  constructor(status: number, reason: LiveRefusalReason) {
    super(`live browser refused: ${reason}`);
    this.name = 'LiveBrowserRefusal';
    this.status = status;
    this.reason = reason;
  }
}

/** Read one live refusal, which wears the browser command envelope `{kind, message}`. */
async function liveRefusal(response: Response): Promise<Error> {
  noteRefusal(response.status);
  const {kind} = refusalEnvelope(await refusalBody(response));
  return new LiveBrowserRefusal(response.status, liveRefusalReason(kind));
}

function liveUrl(runId: string, suffix: string): string {
  return `/web/api/runs/${encodeURIComponent(runId)}/browser/live/${suffix}`;
}

/** Where one Run's frames socket is, on the scheme this page was served over. */
function liveFramesUrl(runId: string): string {
  const base = globalThis.location;
  const scheme = base?.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${base?.host ?? ''}${liveUrl(runId, 'frames')}`;
}

/** Follow one Run's frames until the signal aborts, the session ends, or nobody looks.

 * The stream carries the state beside every frame, so a frame this reader never sees
 * costs it nothing but resolution: the newest one is what it draws next. A socket that
 * drops for any other reason is opened again — for as long as the dock is open, not for a
 * fixed number of tries: the wait doubles up to a cap and stays there, so a proxy's hiccup
 * or a server restart costs the reader a pause rather than the preview, and a socket the
 * server refused by code is reported rather than retried into the same wall.

 * A page nobody is looking at is not watched: `canWatch` says whether anybody is, and when
 * it says no the socket is let go and nothing is opened in its place, however long that
 * lasts, because the server's capture costs frames for a screen that is not there. The
 * question is asked again before every socket, so a watch that resumes after a hide opens
 * the socket the page is actually looked at through — there is no state left over from the
 * socket that ended, and nothing to resolve out of order.
 */
export async function watchLiveBrowser(
  runId: string,
  watcher: LiveWatcher,
  signal: AbortSignal,
  options: LiveWatchOptions = {},
): Promise<void> {
  let delay = options.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS;
  for (;;) {
    if (!await mayWatch(options, signal)) return;
    const closed = await followFrames(runId, watcher, signal);
    if (signal.aborted || closed.ended) return;
    if (closed.refusal !== null) throw new LiveBrowserRefusal(closed.code, closed.refusal);
    // A socket that ended because nobody was looking is not a dropped one: the loop goes
    // straight back to the question above rather than scheduling a reconnect for it.
    if (!await mayWatch(options, signal)) return;
    if (!await suspend(delay, signal)) return;
    delay = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS);
  }
}

/** Whether anybody is looking, or False when the dock closes while the answer is waited on. */
function mayWatch(options: LiveWatchOptions, signal: AbortSignal): Promise<boolean> {
  const watching = options.canWatch?.() ?? Promise.resolve(true);
  return Promise.race([
    watching,
    new Promise<boolean>((resolve) => {
      signal.addEventListener('abort', () => resolve(false), {once: true});
    }),
  ]);
}

/** One socket's life: its messages, and how it closed. */
interface Closed {
  code: number;
  reason: string;
  /** Whether a text update said the session itself is over. */
  ended: boolean;
  /** The refusal a deliberate close code names, null when the socket was not refused. */
  refusal: LiveRefusalReason | null;
}

/** Which refusal a deliberate close code is, so a dock can say why there is nothing. */
function refusalOf(code: number, reason: string): LiveRefusalReason | null {
  if (
    code !== LIVE_CLOSE_UNAUTHENTICATED
    && code !== LIVE_CLOSE_CROSS_ORIGIN
    && code !== LIVE_CLOSE_NO_RUN
    && code !== LIVE_CLOSE_NO_SESSION
  ) {
    return null;
  }
  // The session's own reason travels in the close reason, where the HTTP refusal carries
  // it in the body: `not_configured`, `unreachable` and `expired` are the ones a dock names.
  return liveRefusalReason(reason);
}

/** Open one frames socket and follow it until it closes or the signal aborts. */
async function followFrames(
  runId: string,
  watcher: LiveWatcher,
  signal: AbortSignal,
): Promise<Closed> {
  const socket = new WebSocket(liveFramesUrl(runId));
  socket.binaryType = 'arraybuffer';
  return await new Promise<Closed>((resolve) => {
    let ended = false;
    let closed = false;
    const abort = (): void => {
      // The dock closing — or the page going away — is the reader leaving: the server
      // stops the capture with the last reader, so the socket is the whole protocol.
      socket.close(1000, 'dock closed');
    };
    const finish = (code: number, reason: string): void => {
      if (closed) return;
      closed = true;
      signal.removeEventListener('abort', abort);
      resolve({code, reason, ended, refusal: refusalOf(code, reason)});
    };
    signal.addEventListener('abort', abort, {once: true});
    socket.onmessage = (event: MessageEvent): void => {
      if (typeof event.data === 'string') {
        const message = parseData(event.data);
        // A quiet moment: no state is applied and nothing is repainted, and the only thing
        // the watcher learns is that the session has delivered nothing newer than the
        // sequence it repeats. It is the difference between a still page and a stream the
        // reader has fallen behind, and it is told rather than inferred from silence.
        if (v.is(liveHeartbeat, message)) {
          watcher.quiet?.(message);
          return;
        }
        const parsed = v.safeParse(liveUpdate, message);
        if (!parsed.success) return;
        const update = parsed.output;
        ended = update.ended !== '';
        watcher.update({
          control: update.control,
          driver: update.driver,
          attached: update.attached,
          ended: update.ended,
          sequence: update.sequence,
          framesPerSecond: update.frames_per_second,
          jpegQuality: update.jpeg_quality,
          maxWidth: update.max_width,
          at: update.at,
        });
        if (ended) socket.close(1000, 'session ended');
        return;
      }
      if (!(event.data instanceof ArrayBuffer)) return;
      const frame = decodeLiveFrame(event.data);
      // A frame this dock cannot read, or one the page somehow stamped with no time at
      // all, is dropped: painting it would say less than the frame already on the stage.
      if (frame !== null) watcher.frame(frame);
    };
    socket.onclose = (event: CloseEvent): void => finish(event.code, event.reason);
    socket.onerror = (): void => {
      // A failed connect or a broken socket reports itself as a close right after; the
      // close is what says whether reconnecting is worth it.
    };
    if (signal.aborted) abort();
  });
}

/** Wait out one reconnect delay; False when the dock closed while waiting. */
function suspend(delayMs: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', stop);
      resolve(true);
    }, delayMs);
    const stop = (): void => {
      clearTimeout(timer);
      resolve(false);
    };
    signal.addEventListener('abort', stop, {once: true});
  });
}

/** Take the Run's browser, or hand it back, presenting the seat's own credential.

 * Taking the seat needs no credential and the answer mints one; handing it back presents the
 * token that took it. A seat another caller holds is refused as `taken_over`, which leaves
 * this caller an observer with no credential at all.
 */
export async function setLiveControl(
  runId: string,
  state: LiveControlState,
  token: string | null,
  signal: AbortSignal,
): Promise<LiveSeat> {
  const response = await fetch(liveUrl(runId, 'control'), {
    method: 'POST',
    headers: {
      ...csrfHeaders('application/json'),
      ...(token === null ? {} : {'X-Live-Seat': token}),
    },
    body: JSON.stringify({state}),
    signal,
  });
  return parseWire(response, liveSeat, liveRefusal);
}

/** Send one input event to the page, under the seat's credential and a retry-proof key.

 * `sequence` is this dock's own count of the events it has sent: the session drops an event
 * numbered at or below one it has already run, so a move the wire overtook cannot land after
 * the click that came after it.
 */
export async function sendLiveInput(
  runId: string,
  event: LiveInputEvent,
  token: string,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(liveUrl(runId, 'input'), {
    method: 'POST',
    headers: {
      ...csrfHeaders('application/json'),
      'X-Live-Seat': token,
      'Idempotency-Key': crypto.randomUUID(),
    },
    body: JSON.stringify(event),
    signal,
  });
  if (!response.ok) throw await liveRefusal(response);
}

/** Where the Run's most recent browser screenshot is served from.

 * It is durable Run state, resolved on the server when the dock has nothing live to
 * show; a dock that never had a frame fetches it afresh rather than caching bytes.
 */
export function liveScreenshotUrl(runId: string): string {
  return liveUrl(runId, 'screenshot');
}

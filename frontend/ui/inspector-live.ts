// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {
  LiveBrowserRefusal,
  liveScreenshotUrl,
  sendLiveInput,
  setLiveControl,
  watchLiveBrowser,
  type LiveFrame,
  type LiveHeartbeat,
  type LiveInputEvent,
  type LiveUpdate,
} from '../api/live-browser.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import liveStyles from '../styles/inspector-live.module.css';

/** What the dock shows: nothing attached, a live page, this reader driving it, the page
 * hidden — nobody is looking at it — or the end. */
type LiveBrowserState = 'idle' | 'live' | 'taken' | 'paused' | 'ended';

/** Why there is nothing to show, when the session could not be opened at all. */
type LiveBrowserNotice = 'none' | 'noBrowser' | 'unreachable' | 'expired' | 'service';

/** How often a pointer move is forwarded; the page does not need every pixel of a drag. */
const MOVE_INTERVAL_MS = 40;

/** How far behind the page has to be before the dock says so, in seconds. */
const LAG_NOTICE_SECONDS = 1.0;
/** How often the notice is recomputed as the picture on the stage ages. */
const LAG_TICK_MS = 500;

/** The live browser preview of one Run: its frames, and the driving seat of its page.

 * The dock is the only thing that starts the stream: it subscribes when it becomes
 * active, and closing it — or the conversation moving on — unsubscribes, which is what
 * stops the capture on the server. Frames arrive over one socket, each behind a fixed
 * header carrying its sequence number and its capture time, and the newest one is decoded
 * with `createImageBitmap` and drawn: nothing here reads a base64 string, and nothing
 * builds a `data:` URL. A frame that arrives while the previous one is still decoding
 * replaces it, so a slow dock loses resolution rather than memory — and a frame older than
 * the one already painted is dropped rather than drawn over it.
 *
 * The age the dock states is the picture's, and only the stream's being behind is worth
 * saying: every state message says which frame is the newest the session has delivered, and
 * a heartbeat repeats that number unchanged while the page stays quiet. A picture whose age
 * is that stillness is the newest there is — a page that has not changed is current, not
 * late — so the notice names the distance only when a frame arrived late or a newer one is
 * known to exist and has not been drawn. A dock that counted the wall clock into every
 * quiet page called a still picture "behind the page" for as long as nothing moved, which
 * is the false alarm this readout exists not to raise.

 * One seat holds one driver, and this dock learns whether that is it from the *answer* to
 * its own take: the token that answer carried is what a write presents, and no update ever
 * carries one. So a dock that only watched — or that replayed everything the socket sent —
 * holds nothing it could drive with, and an update naming a held seat is all it takes for
 * this one to know it is an observer.

 * A dock in a hidden page is nobody watching: the socket is closed while the page is hidden
 * and no frame is asked for until it is shown again, and the last frame stays on the stage
 * in the meantime. What a pause is not is a hand-back: the seat this dock took survives it,
 * with the credential the take answered with, because the session's own idle release is
 * what decides a holder has gone — and so does the rung the session opened at, which is the
 * deployment's own budget rather than a property of the socket that was closed. The watch
 * asks whether anybody is looking before every socket it opens, so hide, show, hide and
 * show again all land in the same place. That is the same rule the server enforces from the
 * other side — its capture stops with the last reader — so nothing about it needs a message
 * of its own.
 */
export class DlInspectorLive extends LightElement {
  static properties = {
    runId: {attribute: false},
    active: {attribute: false},
    state: {state: true},
    notice: {state: true},
    frame: {state: true},
    url: {state: true},
    lastScreenshot: {state: true},
    attached: {state: true},
    lag: {state: true},
    rung: {state: true},
    base: {state: true},
  };

  declare runId: string | null;
  declare active: boolean;
  declare state: LiveBrowserState;
  declare notice: LiveBrowserNotice;
  /** The newest frame, decoded; the one it replaced is released as it is set. */
  declare frame: ImageBitmap | null;
  declare url: string;
  /** Whether the Run's last stored screenshot is worth drawing instead of a frame. */
  declare lastScreenshot: boolean;
  /** Whether the Run drives a page this dock can show right now. */
  declare attached: boolean;
  /** How far behind the page the newest frame is, in seconds, or null when nothing is. */
  declare lag: number | null;
  /** The rung the session is serving now, as `[fps, quality]`, so a move repaints the notice. */
  declare rung: [number | null, number | null];
  /** The rung it started at, which is the deployment's own ceiling for this session. */
  declare base: [number | null, number | null];

  #stream: AbortController | null = null;
  #input: AbortController | null = null;
  /** Whether a mobile browser has frozen this page, which is nobody looking at it too. */
  #frozen = false;
  /** Resolves when this dock's page is next looked at, replaced as the watch asks. */
  #shown: Promise<void> = Promise.resolve();
  #showing: (() => void) | null = null;
  /** The seat's own credential, held only while this dock took the seat; never sent on. */
  #seat: string | null = null;
  /** The label the take that minted that credential was answered with, so an update naming
   * somebody else is recognisable as a seat this dock no longer holds. */
  #label: string | null = null;
  /** The Run whose page that credential took, which is where a hand-back belongs: the dock
   * can be moved to another Run with the old seat still held. */
  #seatRun: string | null = null;
  /** Who holds the seat now, as the last update named them: a label, never a credential. */
  #driver: string | null = null;
  /** This dock's own count of the input events it has sent, oldest first. */
  #inputSequence = 0;
  #screenshotFailed = false;
  /** When the last pointer move was sent, and where the pointer is now. */
  #lastMove = 0;
  #lastPoint: {x: number; y: number} | null = null;
  #moveTimer: ReturnType<typeof setTimeout> | null = null;
  #ended = '';
  /** The newest frame a decode has not started on yet: the frame after them replaces them. */
  #pending: LiveFrame | null = null;
  #decoding = false;
  /** The bitmap the canvas is already showing, so a state change never repaints it. */
  #painted: ImageBitmap | null = null;
  /** The sequence number of the newest frame painted, so an older one is never drawn. */
  #paintedSequence: bigint | null = null;
  /** The newest sequence the session has said it delivered, from an update or a heartbeat. */
  #newestSequence: bigint | null = null;
  /** Whether the painted frame was already older than the notice threshold when it was drawn:
   * the delivery itself was late, which is the stream being behind however quiet the page
   * has been since. */
  #late = false;
  /** The age the newest painted frame already had when this dock drew it, in seconds, and
   * the `performance.now()` reading taken in the same breath: how old it is now, together. */
  #age: number | null = null;
  #ageAt = 0;
  #lagTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    super();
    this.runId = null;
    this.active = false;
    this.state = 'idle';
    this.notice = 'none';
    this.frame = null;
    this.url = '';
    this.lastScreenshot = false;
    this.attached = false;
    this.lag = null;
    this.rung = [null, null];
    this.base = [null, null];
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // The page is what a dock cannot see for itself: a hidden tab, a minimised window and
    // a locked phone are all one thing here, and all of them are nobody looking.
    const {lifetime: signal} = this;
    document.addEventListener('visibilitychange', this.#visibility, {signal});
    window.addEventListener('pagehide', this.#visibility, {signal});
    window.addEventListener('freeze', this.#visibility, {signal});
    window.addEventListener('resume', this.#visibility, {signal});
  }

  override disconnectedCallback(): void {
    this.#dismiss();
    super.disconnectedCallback();
  }

  /** Opening or moving the dock changes what it shows at once (the stage, the notice, the seat), so it
   *  is decided here, before the render that draws it, and not in `updated`, which would draw twice. */
  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('active') || changed.has('runId')) {
      if (this.active && this.runId) void this.#start();
      else this.#dismiss();
    }
  }

  protected override updated(): void {
    this.#paint();
  }

  /** A lifecycle event, which says either why nothing is being looked at or that it is. */
  #visibility = (event: Event): void => {
    if (event.type === 'freeze') this.#frozen = true;
    if (event.type === 'resume') this.#frozen = false;
    if (this.#interrupted()) this.#paused();
    else this.#resumed();
  };

  /** Whether anybody is looking at this page: hidden, frozen, or neither. */
  #interrupted(): boolean {
    return this.#frozen || document.visibilityState === 'hidden';
  }

  /** Nobody is looking at this page: let the socket go, and keep what it drew.
   *
   * The socket goes and nothing else does. The last frame stays on the stage, because the
   * stream is not ended and a reader who comes back finds the page as they left it; so does
   * the seat this dock holds, because hiding a tab is not handing the page back — a holder
   * who is really gone is what the session's own idle release is for, and a hand-back
   * nobody asked for would spend a credential the reader still expects to drive with; and so
   * does the budget the session is serving at, which belongs to the session rather than to
   * one socket's life. What a paused dock must not do is compare a stepped-down rung
   * against itself on the way back, which is what re-learning the base from the next
   * socket's first update would do.
   */
  #paused(): void {
    if (!this.active || this.runId === null) return;
    this.#stopStream();
    if (this.state === 'live' || this.state === 'taken') this.state = 'paused';
  }

  /** The page is being looked at again: resolve the wait, and watch it through a socket.
   *
   * Only a watch that is not open is started: a visible page that is already being watched
   * keeps the socket it has, so a lifecycle event that reports nothing new is not a
   * reconnect, and the seat and the budget it would have let go of stay where they are.
   */
  #resumed(): void {
    if (!this.active || this.runId === null) return;
    this.#show();
    if (this.#stream !== null) return;
    void this.#watch();
  }

  /** Whether anybody is looking now, in a promise that the next show resolves. */
  #looking(): Promise<boolean> {
    if (!this.#interrupted()) return Promise.resolve(true);
    return this.#nextShow().then(() => true);
  }

  /** A promise that resolves when this dock's page is next shown, and nothing before it. */
  #nextShow(): Promise<void> {
    if (this.#showing === null) {
      this.#shown = new Promise((resolve) => {
        this.#showing = resolve;
      });
    }
    return this.#shown;
  }

  /** Resolve the promise a hidden watch is waiting on: the page is being looked at. */
  #show(): void {
    const showing = this.#showing;
    this.#showing = null;
    this.#shown = Promise.resolve();
    showing?.();
  }

  /** Whether another viewer holds the seat, which leaves this dock read-only. */
  #observer(): boolean {
    return this.#driver !== null && this.#seat === null;
  }

  /** Whether this dock is the one driving. */
  #driving(): boolean {
    return this.state === 'taken';
  }

  /** Stop the socket, leaving the session and this dock's own state alone.
   *
   * What the socket was holding goes with it — the frame a decode had not started on, the
   * coalesced pointer move — because neither belongs to a stream that is over. What is on
   * the stage does not: a paused dock keeps the last picture it drew.
   */
  #stopStream(): void {
    this.#stream?.abort();
    this.#stream = null;
    this.#pending = null;
    this.#stopMoving();
  }

  /** Stop the stream and hand the driving seat back; closing the dock does both.

   * What the dock shows is not this function's to decide: only the caller — the property
   * change that closed the dock or moved it to another Run, and the disconnect — knows that
   * the dock itself is done. So this stops the socket, hands back the seat it held, forgets
   * everything the stream said, and leaves ``state`` to whoever asked. A pause is
   * deliberately not a caller of it: see ``#paused``.
   */
  #stop(): void {
    this.#stopStream();
    this.#show();
    this.#handBack();
    this.#input = null;
    this.#driver = null;
    this.attached = false;
    this.rung = [null, null];
    this.base = [null, null];
  }

  /** Give up the seat this dock holds, under the credential that took it.
   *
   * The request outlives the socket it was taken under — a hand-back must not be cancelled
   * by the stream the dock is closing — so it carries a signal of its own rather than the
   * one belonging to a watch that is already over.
   */
  #handBack(): void {
    const seat = this.#seat;
    const runId = this.#seatRun ?? this.runId;
    this.#seat = null;
    this.#label = null;
    this.#seatRun = null;
    if (runId === null || seat === null) return;
    void setLiveControl(runId, 'agent', seat, new AbortController().signal)
      .catch(() => undefined);
  }

  /** The dock itself is closing or moving on: there is nothing to show any more.
   *
   * The stage goes with it — that is the whole difference between this and a pause, which
   * keeps the last frame so a reader who comes back finds the page as they left it.
   */
  #dismiss(): void {
    this.#stop();
    this.#release();
    if (this.state !== 'idle' && this.state !== 'ended') this.state = 'idle';
  }

  /** Let go of the frame the canvas shows; the dock keeps no picture of its own. */
  #release(): void {
    this.frame?.close();
    this.frame = null;
    this.#painted = null;
    this.#paintedSequence = null;
    this.#newestSequence = null;
    this.#stopLag();
  }

  /** Open this Run from the top: the seat goes back, and everything the last stream said. */
  async #start(): Promise<void> {
    this.#stop();
    await this.#watch();
  }

  /** Follow this Run's frames until the watch ends, nobody looks, or the dock lets it go.
   *
   * The socket and nothing around it: the seat this dock holds, the budget the session is
   * serving at and the picture on the stage are the dock's own state, and none of them
   * belongs to one socket's life. That is what makes a show after a hide a resume rather
   * than a fresh start, and what keeps a pause from spending a credential nobody gave up.
   */
  async #watch(): Promise<void> {
    const runId = this.runId;
    if (runId === null || this.#interrupted()) return;
    const stream = new AbortController();
    this.#stream = stream;
    this.#input = new AbortController();
    this.notice = 'none';
    this.#ended = '';
    this.#screenshotFailed = false;
    this.lastScreenshot = false;
    try {
      await watchLiveBrowser(
        runId,
        {
          update: (update) => { this.#applied(update); },
          frame: (frame) => { this.#received(frame); },
          quiet: (heartbeat) => { this.#quiet(heartbeat); },
        },
        stream.signal,
        {
          // Asked before every socket rather than handed over as a promise: what this dock
          // is waiting for belongs to the watch, not to the socket that happened to be open
          // when the page was hidden.
          canWatch: () => this.#looking(),
        },
      );
      if (stream.signal.aborted || this.#stream !== stream) return;
      this.#ended = this.#ended || 'ended';
    } catch (error) {
      // A watch this dock has already let go of owns nothing: a newer one, or the pause
      // that replaced it, is what says what the dock shows now.
      if (stream.signal.aborted || this.#stream !== stream) return;
      this.state = 'idle';
      this.notice = this.#noticeFor(error);
      this.#syncFallback();
      return;
    }
    if (!stream.signal.aborted && this.#stream === stream) {
      this.state = 'ended';
      this.#driver = null;
      this.attached = false;
      this.#syncFallback();
    }
  }

  #applied(update: LiveUpdate): void {
    if (update.ended) this.#ended = update.ended;
    this.#driver = update.driver;
    this.attached = update.attached;
    this.rung = [update.framesPerSecond, update.jpegQuality];
    // The base is whatever the session names when it opens: it is serving all of the budget
    // the deployment configured then, and anything below it later is this session stepping
    // down to keep up. A deployment configured below the ladder's old numbers is not
    // stepped-down just for being configured that way.
    if (this.base[0] === null) this.base = [update.framesPerSecond, update.jpegQuality];
    // A seat an update names is this dock's only while it is the label its own take was
    // answered with. Anything else — nobody driving, or another viewer's label — means the
    // credential is spent: the session released it while the page was hidden, or someone
    // else took the seat in the meantime. Keeping it would leave the button reading "hand
    // back" for a seat this dock does not have, and every write it sent would be refused.
    if (this.#seat !== null && update.driver !== this.#label) {
      this.#seat = null;
      this.#label = null;
      this.#seatRun = null;
    }
    // The update that ends a session carries the state alone; the frame it described
    // stays on the stage, because it is still the last thing the page showed.
    this.state = update.ended !== ''
      ? 'ended'
      : this.#seat !== null ? 'taken' : 'live';
    // What the session has delivered, so a picture older than this message is measurable
    // against it: the state travels before the frame it describes, so the frame that follows
    // is the one the count is about.
    if (update.sequence !== null) this.#delivered(update.sequence);
    this.#syncFallback();
  }

  /** Whether the session is serving less than it started at, so the dock can say why. */
  #stepped(): boolean {
    const [frames, quality] = this.rung;
    const [baseFrames, baseQuality] = this.base;
    if (frames === null || quality === null) return false;
    return frames < (baseFrames ?? frames) || quality < (baseQuality ?? quality);
  }

  /** Take one frame: the newest one waits, and the one it replaces is dropped. */
  #received(frame: LiveFrame): void {
    if (!this.active) return;
    // A frame older than the one already painted describes a moment the page has moved on
    // from: dropping it is the whole point of the sequence number, and painting it would
    // put the stage back in time.
    if (this.#paintedSequence !== null && frame.sequence <= this.#paintedSequence) return;
    this.#pending = frame;
    if (!this.#decoding) void this.#decode();
  }

  /** Decode the frames this dock was sent, newest first and never a backlog. */
  async #decode(): Promise<void> {
    this.#decoding = true;
    try {
      while (this.#pending !== null && this.active) {
        const frame = this.#pending;
        this.#pending = null;
        const photo = await createImageBitmap(new Blob([frame.jpeg], {type: 'image/jpeg'}))
          .catch(() => null);
        if (photo === null) continue;
        if (!this.active) {
          // The dock closed while this frame was decoding: nothing keeps it.
          photo.close();
          return;
        }
        // A newer frame arrived while this one decoded, so this one is already stale: it
        // is dropped here for the same reason one that arrived out of order is dropped.
        if (this.#paintedSequence !== null && frame.sequence <= this.#paintedSequence) {
          photo.close();
          continue;
        }
        this.frame?.close();
        this.frame = photo;
        this.#paintedSequence = frame.sequence;
        // How old the picture already was when this dock finished turning it into something
        // it can draw: the frame's capture time is on the same clock as `Date.now()`, so one
        // subtraction here is the whole of it. Taking it once is what the readout is added
        // to, rather than subtracting the capture stamp from the wall clock again on every
        // tick, which counted each second the frame spent on the stage twice. A frame that
        // was already old when it arrived is the delivery being late, and no amount of
        // quiet after it makes the stream current: the fact is kept, and the session's own
        // "nothing newer" is what clears it.
        this.#age = Date.now() / 1000 - frame.capturedAt;
        this.#ageAt = performance.now();
        this.#late = this.#age >= LAG_NOTICE_SECONDS;
        this.#showLag();
      }
    } finally {
      this.#decoding = false;
    }
  }

  /** Draw the newest frame on the canvas, once per frame rather than once per render. */
  #paint(): void {
    const photo = this.frame;
    const canvas = this.querySelector('canvas');
    if (photo === null || canvas === null || photo === this.#painted) return;
    canvas.width = photo.width;
    canvas.height = photo.height;
    canvas.getContext('2d')?.drawImage(photo, 0, 0);
    this.#painted = photo;
  }

  /** Say how far behind the page the picture is, and keep saying it as time passes.
   *
   * The number is the painted frame's age, but the notice is about the stream, not about the
   * page: a picture that is old because the page has not changed is the newest picture there
   * is, and calling that "behind the page" is a false alarm. So the age is stated only while
   * `#behind` holds, and recomputed every tick because a quiet page would otherwise keep a
   * number the dock has already stopped meaning.
   */
  #showLag(): void {
    const age = this.#age;
    if (age === null) return;
    // How old the frame already was when this dock drew it, plus how long it has been on the
    // stage since. The capture time is epoch seconds on this machine's clock, so that first
    // distance is one subtraction; the second is this browser's own elapsed time. Adding the
    // wall clock's distance from the capture stamp again here would count the same seconds
    // twice, and a picture ten seconds old would read as twenty.
    const seconds = age + (performance.now() - this.#ageAt) / 1000;
    this.lag = seconds >= LAG_NOTICE_SECONDS && this.#behind() ? seconds : null;
    if (this.#lagTimer === null) {
      this.#lagTimer = setInterval(() => { this.#showLag(); }, LAG_TICK_MS);
    }
  }

  /** Whether the picture on the stage is behind the page, rather than merely old.
   *
   * Two things make it behind: the frame was already older than the notice threshold when
   * this dock drew it, so the delivery was late; or the session has delivered a newer frame
   * than the one painted, so a picture exists that the stage does not show yet. When neither
   * holds, the painted frame is the newest the session has delivered: the page is still, and
   * a still picture is current. What tells the two apart is the count every state message
   * carries and every heartbeat repeats, never the silence itself.
   */
  #behind(): boolean {
    if (this.#late) return true;
    const newest = this.#newestSequence;
    const painted = this.#paintedSequence;
    return newest !== null && painted !== null && newest > painted;
  }

  /** Take the newest count the session has delivered, and stop calling a still page late. */
  #delivered(sequence: number): void {
    this.#newestSequence = BigInt(sequence);
    const painted = this.#paintedSequence;
    // A frame that arrived late was late as it was drawn, and the count does not make it
    // younger: once the session states that no newer frame exists, the picture is the newest
    // there is, and a page that stopped changing is not a stream that is behind it.
    if (painted !== null && this.#newestSequence <= painted) this.#late = false;
    this.#showLag();
  }

  /** A quiet moment: the session has delivered nothing newer than the count it repeats. */
  #quiet(heartbeat: LiveHeartbeat): void {
    this.#delivered(heartbeat.sequence);
  }

  /** Stop counting the lag; nothing is on the stage to be behind. */
  #stopLag(): void {
    if (this.#lagTimer !== null) clearInterval(this.#lagTimer);
    this.#lagTimer = null;
    this.lag = null;
    this.#age = null;
    this.#late = false;
  }

  /** With nothing attached to draw, the Run's own last screenshot is what the stage shows.
   *
   * It is durable Run state and not this browser's cache: the address is fetched afresh,
   * and a Run that kept none simply keeps the text-only stage.
   */
  #syncFallback(): void {
    const drawing = this.attached && (this.state === 'live' || this.state === 'taken');
    this.lastScreenshot = !drawing
      && this.frame === null
      && !this.#screenshotFailed
      && this.runId !== null;
  }

  #screenshotBroken(): void {
    this.#screenshotFailed = true;
    this.lastScreenshot = false;
  }

  #noticeFor(error: unknown): LiveBrowserNotice {
    if (!(error instanceof LiveBrowserRefusal)) return 'service';
    if (error.reason === 'not_configured') return 'noBrowser';
    if (error.reason === 'unreachable' || error.reason === 'no_page') return 'unreachable';
    if (error.reason === 'expired') return 'expired';
    return 'service';
  }

  async #control(state: 'agent' | 'human'): Promise<void> {
    const runId = this.runId;
    const input = this.#input;
    if (runId === null || input === null) return;
    if (state === 'human' && this.#observer()) return;
    try {
      // Taking presents nothing and hands back what the take answered with: the token is
      // this dock's own, and it never travelled over anything another reader can see. The
      // label beside it is what the stream will name while this dock holds the seat, so it
      // is kept too — not as a credential, but as the one thing that says the seat is still
      // this dock's.
      const seat = await setLiveControl(runId, state, this.#seat, input.signal);
      this.#driver = seat.driver;
      this.#seat = state === 'human' ? seat.token : null;
      this.#label = this.#seat === null ? null : seat.driver;
      this.#seatRun = this.#seat === null ? null : runId;
      this.state = this.#seat !== null ? 'taken' : 'live';
    } catch (error) {
      // A seat somebody else took while this dock was asking is not a broken preview:
      // the stream already says who drives, and this dock stays the observer it is.
      if (error instanceof LiveBrowserRefusal && error.reason === 'taken_over') {
        this.#seat = null;
        this.#label = null;
        this.#seatRun = null;
        return;
      }
      this.notice = 'service';
    }
  }

  #send(event: LiveInputEvent): void {
    const runId = this.runId;
    const input = this.#input;
    const seat = this.#seat;
    if (runId === null || input === null || seat === null || !this.#driving()) return;
    // Numbered as they are sent, so the page can drop one the wire overtook rather than
    // running it over the state the event that overtook it already left.
    this.#inputSequence += 1;
    const numbered = {...event, sequence: this.#inputSequence};
    void sendLiveInput(runId, numbered, seat, input.signal).catch(() => undefined);
  }

  #pointer(event: MouseEvent): void {
    if (!this.#driving()) return;
    const canvas = this.#frameCanvas();
    if (canvas === null) return;
    const box = canvas.getBoundingClientRect();
    if (box.width === 0) return;
    const scale = canvas.width / box.width;
    const point = {
      x: Math.round((event.clientX - box.left) * scale),
      y: Math.round((event.clientY - box.top) * (canvas.height / box.height)),
    };
    const now = performance.now();
    if (now - this.#lastMove < MOVE_INTERVAL_MS) {
      // A burst of moves is coalesced to its newest position: sending every one of them
      // would queue them ahead of the click that follows, and the click is what the person
      // is waiting for. Only the latest position is kept, and it is sent when the interval
      // is up — so a drag still ends where the pointer actually is.
      this.#lastPoint = point;
      if (this.#moveTimer === null) {
        this.#moveTimer = setTimeout(() => {
          this.#moveTimer = null;
          const latest = this.#lastPoint;
          this.#lastPoint = null;
          if (latest === null) return;
          this.#lastMove = performance.now();
          this.#send({type: 'mouse', action: 'move', x: latest.x, y: latest.y});
        }, MOVE_INTERVAL_MS - (now - this.#lastMove));
      }
      return;
    }
    this.#lastMove = now;
    this.#lastPoint = null;
    this.#send({type: 'mouse', action: 'move', x: point.x, y: point.y});
  }

  /** Forget a coalesced move that has not been sent: the pointer is not being dragged. */
  #stopMoving(): void {
    if (this.#moveTimer !== null) clearTimeout(this.#moveTimer);
    this.#moveTimer = null;
    this.#lastPoint = null;
  }

  #clicked(event: MouseEvent): void {
    if (!this.#driving()) return;
    const canvas = this.#frameCanvas();
    if (canvas === null) return;
    const box = canvas.getBoundingClientRect();
    if (box.width === 0) return;
    this.#send({
      type: 'mouse',
      action: 'click',
      x: Math.round((event.clientX - box.left) * (canvas.width / box.width)),
      y: Math.round((event.clientY - box.top) * (canvas.height / box.height)),
      button: event.button === 2 ? 'right' : event.button === 1 ? 'middle' : 'left',
    });
    event.preventDefault();
  }

  /** The canvas a frame is drawn on, or null while the stage shows something else. */
  #frameCanvas(): HTMLCanvasElement | null {
    const canvas = this.querySelector<HTMLCanvasElement>('canvas');
    return canvas !== null && canvas.width > 0 ? canvas : null;
  }

  #keydown(event: KeyboardEvent): void {
    if (!this.#driving()) return;
    // Typing goes to the page as keys, and the dock keeps none of it.
    this.#send({type: 'key', action: 'press', key: event.key});
    event.preventDefault();
  }

  #wheelled(event: WheelEvent): void {
    if (!this.#driving()) return;
    this.#send({type: 'wheel', delta_x: Math.round(event.deltaX), delta_y: Math.round(event.deltaY)});
    event.preventDefault();
  }

  #navigate(event: Event): void {
    event.preventDefault();
    const url = this.url.trim();
    if (url === '') return;
    this.#send({type: 'navigate', url});
  }

  protected override render(): TemplateResult {
    return html`
      <div class=${liveStyles.live} data-live-state=${this.state}>
        ${this.#status()}
        ${this.#stage()}
        ${this.#controls()}
      </div>
    `;
  }

  #status(): TemplateResult {
    const running = this.state === 'live' || this.state === 'taken' || this.state === 'paused';
    // The notice is about the live stream, so it is only the live stream that states it: a
    // paused or ended dock has its own line, and a number that keeps counting behind one is
    // a number about nothing the reader can act on.
    const watching = this.state === 'live' || this.state === 'taken';
    const text = this.state === 'taken'
      ? msg('You have taken over this page', {id: 'inspectorLive.taken'})
      : this.state === 'ended'
        ? msg('This live session has ended', {id: 'inspectorLive.ended'})
        : this.state === 'paused'
          ? msg('The stream is paused while this page is hidden', {id: 'inspectorLive.paused'})
          : this.#observer()
            ? msg('Another viewer is driving this page', {id: 'inspectorLive.otherDriver'})
            : running
              ? this.attached
                ? msg('Watching the agent’s browser', {id: 'inspectorLive.watching'})
                : msg('No page is open right now', {id: 'inspectorLive.detached'})
              : this.#idleText();
    return html`
      <p class=${liveStyles.status} role="status">
        ${running ? html`<span class=${liveStyles.dot} aria-hidden="true">${icon('status-dot', {size: 'xs'})}</span>` : nothing}
        ${text}
      </p>
      ${this.#stepped() ? html`
        <p class=${liveStyles.notice} role="note">
          ${msg('The stream stepped down to keep up', {id: 'inspectorLive.stepped'})}
        </p>
      ` : nothing}
      ${this.lag === null || !watching ? nothing : html`
        <p class=${liveStyles.notice} role="note">
          ${msg('The stream is behind the page by', {id: 'inspectorLive.lag'})}
          ${this.#lagText()}
        </p>
      `}
    `;
  }

  /** How far behind the page the newest frame is, as one short figure. */
  #lagText(): string {
    const seconds = this.lag ?? 0;
    return seconds < 10 ? `${seconds.toFixed(1)} s` : `${Math.round(seconds)} s`;
  }

  #idleText(): string {
    switch (this.notice) {
      case 'noBrowser': return msg('This Run is not using a browser', {id: 'inspectorLive.noBrowser'});
      case 'unreachable': return msg('The browser is not reachable right now', {id: 'inspectorLive.unreachable'});
      case 'expired': return msg('This live session reached its time limit', {id: 'inspectorLive.expired'});
      case 'service': return msg('The live preview is unavailable', {id: 'inspectorLive.service'});
      default: return msg('No live page yet; the stream starts when one is open', {id: 'inspectorLive.waiting'});
    }
  }

  #stage(): TemplateResult {
    const frame = this.frame;
    if (frame === null) {
      const shot = this.runId !== null && this.lastScreenshot
        ? liveScreenshotUrl(this.runId)
        : null;
      return html`
        <div class=${liveStyles.stage}>
          ${shot === null
            ? html`<p class=${liveStyles.note}>${this.#idleText()}</p>`
            : html`
              <figure class=${liveStyles.lastFrame}>
                <img class=${liveStyles.frame} src=${shot}
                     alt=${msg('The last screenshot this Run took', {id: 'inspectorLive.lastScreenshotAlt'})}
                     @error=${() => this.#screenshotBroken()}>
                <figcaption class=${liveStyles.note}>
                  ${msg('Last screenshot of this Run', {id: 'inspectorLive.lastScreenshot'})}
                </figcaption>
              </figure>
            `}
        </div>
      `;
    }
    const taken = this.#driving();
    return html`
      <div class=${liveStyles.stage} tabindex=${taken ? '0' : nothing}
           @mousemove=${this.#pointer} @click=${this.#clicked}
           @keydown=${this.#keydown} @wheel=${this.#wheelled}
           @contextmenu=${(event: Event) => event.preventDefault()}>
        <canvas class=${liveStyles.frame} width=${frame.width} height=${frame.height}
                role="img"
                aria-label=${msg('The page this Run is driving', {id: 'inspectorLive.frameAlt'})}></canvas>
      </div>
    `;
  }

  #controls(): TemplateResult {
    const taken = this.#driving();
    const live = this.state === 'live' || this.state === 'taken';
    const observer = this.#observer();
    return html`
      <div class=${liveStyles.controls}>
        <button type="button" class=${liveStyles.action} ?disabled=${!live || observer}
                aria-label=${taken
                  ? msg('Hand the browser back to the agent', {id: 'inspectorLive.releaseAria'})
                  : observer
                    ? msg('Another viewer is driving this page', {id: 'inspectorLive.otherDriver'})
                    : msg('Take over the browser', {id: 'inspectorLive.takeOverAria'})}
                @click=${() => this.#control(taken ? 'agent' : 'human')}>
          ${taken
            ? msg('Hand back to the agent', {id: 'inspectorLive.release'})
            : msg('Take over', {id: 'inspectorLive.takeOver'})}
        </button>
      </div>
      ${taken ? html`
        <p class=${liveStyles.notice} role="note">
          ${msg('While you hold control, the agent’s page actions are refused.', {id: 'inspectorLive.notice'})}
        </p>
        <form class=${liveStyles.navigate} @submit=${this.#navigate}>
          <input class=${liveStyles.url} type="url" inputmode="url" .value=${this.url}
                 aria-label=${msg('Go to a URL', {id: 'inspectorLive.goToAria'})}
                 placeholder=${msg('https://…', {id: 'inspectorLive.urlPlaceholder'})}
                 @input=${(event: Event) => { this.url = (event.target as HTMLInputElement).value; }}>
          <button type="submit" class=${liveStyles.action}>${msg('Go', {id: 'inspectorLive.go'})}</button>
        </form>
      ` : nothing}
    `;
  }
}

customElements.define('dl-inspector-live', DlInspectorLive);

declare global {
  interface HTMLElementTagNameMap {
    'dl-inspector-live': DlInspectorLive;
  }
}

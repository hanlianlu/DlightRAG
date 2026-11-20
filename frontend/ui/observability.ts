// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Starts the browser observability channel: errors, RUM interactions, and replay. */

export type ObservabilityConfig = {
  url: string;
  apiKey: string;
  service: string;
  /** The share of sessions whose replay is recorded; errors and RUM ignore it. */
  replaySampleRate: number;
};

/** The SDK surface this module uses. Injected so a test can hold a load that fails. */
type ObservabilitySdk = {
  init(options: {
    url: string;
    apiKey: string;
    service: string;
    tracePropagationTargets: RegExp[];
    maskAllInputs: boolean;
    maskAllText: boolean;
    consoleCapture: boolean;
    recordCanvas: boolean;
    advancedNetworkCapture: boolean;
    disableReplay: boolean;
  }): void;
};

type SdkLoader = () => Promise<{default: ObservabilitySdk}>;

const loadBrowserSdk: SdkLoader = () => import('@hyperdx/browser');

/**
 * Where one session's replay decision is remembered. Injectable so a test can
 * hold the storage, and so the gate can state what a browser without one does.
 */
export type ReplayMemory = {read(): string | null; write(value: string): void};

const REPLAY_DECISION_KEY = 'dlightrag.replay-sampled';

function browserSession(): ReplayMemory {
  const storage = globalThis.sessionStorage;
  return {
    read: () => storage.getItem(REPLAY_DECISION_KEY),
    write: (value) => storage.setItem(REPLAY_DECISION_KEY, value),
  };
}

/**
 * Whether this browser session records replay: one coin flip, made once.
 *
 * Replay is the one signal whose volume grows with a session rather than with the
 * work in it, so it is the one signal a deployment may sample. The decision is
 * remembered for the session, so a reload inside one session does not record half
 * of it and pass over the other half; a storage that refuses (private mode, blocked
 * cookies) falls back to a coin flip per page rather than failing the page.
 * Sampling never touches errors or RUM: those are reported from every session.
 */
export function replayEnabledForSession(
  sampleRate: number,
  memory: ReplayMemory = browserSession(),
  random: () => number = Math.random,
): boolean {
  if (sampleRate >= 1) return true;
  if (sampleRate <= 0) return false;
  try {
    const remembered = memory.read();
    if (remembered === '1') return true;
    if (remembered === '0') return false;
    const sampled = random() < sampleRate;
    memory.write(sampled ? '1' : '0');
    return sampled;
  } catch {
    return random() < sampleRate;
  }
}

/**
 * Start the browser channel, once the server sent a destination and a key.
 *
 * `config` is the bootstrap's `observability`, which a deployment that has not
 * configured that channel does not send, so `null` starts nothing. The call
 * returns before the SDK exists: the load and the initialization sit behind a
 * dynamic import, and a blocked, slow, or failing SDK costs telemetry, never the
 * page (ADR 0039, docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md).
 */
export function initializeObservability(
  config: ObservabilityConfig | null,
  loadSdk: SdkLoader = loadBrowserSdk,
): void {
  if (config === null) return;
  // Nothing awaits this promise, and nothing may reject out of it.
  void start(config, loadSdk).catch(() => {});
}

async function start(config: ObservabilityConfig, loadSdk: SdkLoader): Promise<void> {
  try {
    const {default: hyperdx} = await loadSdk();
    hyperdx.init({
      url: config.url,
      apiKey: config.apiKey,
      service: config.service,
      // A browser request carries a `traceparent`, so a front-end error or a slow
      // call opens the backend trace that served it.
      tracePropagationTargets: [/\/web\/api/, /\/api\//],
      // ADR 0039: the browser channel states the trace seam's one rule — private
      // values are masked and monitoring has no off state. DOM text and input values
      // are private, so replay keeps the interactions without the text they happened
      // over; request and response bodies, console output, and canvas recording are
      // private, so they are not captured at all.
      maskAllInputs: true,
      maskAllText: true,
      consoleCapture: false,
      advancedNetworkCapture: false,
      recordCanvas: false,
      // The one sampled signal, decided once for this session. A session the coin
      // flip passed over still reports its errors and its RUM interactions; only
      // the recording is off, and the masking above is unchanged either way.
      disableReplay: !replayEnabledForSession(config.replaySampleRate),
    });
  } catch {
    // A load that fails, whether an ad blocker or a dead endpoint, installs no
    // telemetry and reports nothing: the page behaves exactly as it did before.
  }
}

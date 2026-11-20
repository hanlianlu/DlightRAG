// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {css, html, LitElement, nothing, type TemplateResult} from 'lit';
import {raise} from '../lib/dom.ts';
import type {ColorMode} from '../lib/theme.ts';

const PERMISSIONS = [
  'accelerometer', 'ambient-light-sensor', 'autoplay', 'battery', 'camera',
  'clipboard-read', 'clipboard-write', 'display-capture', 'encrypted-media',
  'fullscreen', 'geolocation', 'gyroscope', 'hid', 'idle-detection',
  'local-fonts', 'magnetometer', 'microphone', 'midi', 'payment',
  'picture-in-picture', 'publickey-credentials-get', 'screen-wake-lock',
  'serial', 'speaker-selection', 'usb', 'web-share', 'xr-spatial-tracking',
].map((name) => `${name} 'none'`).join('; ');

const ESCAPE_MESSAGE = 'dl-artifact-frame-escape';
const COLOR_MODE_MESSAGE = 'dl-artifact-frame-color-mode';

const BASE_CSP = [
  "default-src 'none'",
  "style-src 'unsafe-inline'",
  'img-src data: blob:',
  'media-src data: blob:',
  'font-src data:',
  "connect-src 'none'",
  "worker-src 'none'",
  "child-src 'none'",
  "frame-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "manifest-src 'none'",
];

/** The product's colour mode, which theme-init and the theme control keep on the root element. */
function hostColorMode(): ColorMode {
  const mode = document.documentElement.getAttribute('data-color-mode');
  if (mode === 'light' || mode === 'dark') return mode;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function wrapperDocument(
  source: string,
  active: boolean,
  escapeToken: string,
  colorMode: ColorMode,
): string {
  const script = active ? "script-src 'unsafe-inline'" : "script-src 'none'";
  const policy = [BASE_CSP[0], script, ...BASE_CSP.slice(1)].join('; ');
  // The frame cannot see the product's own toggle from inside its opaque origin, so the wrapper
  // hands the colour mode over as the root attribute the report's stylesheet keys off, and keeps it
  // current with a message after the reader switches. The same two checks as `isImeKey` keep the
  // Escape that cancels an IME composition with the composition, not with the reader leaving.
  const bridge = active
    ? `<script>(()=>{const token=${JSON.stringify(escapeToken)};` +
      'document.currentScript.remove();window.addEventListener("message",event=>{' +
      'if(event.source!==parent||!event.data)return;' +
      `if(event.data.type!=="${COLOR_MODE_MESSAGE}"||event.data.token!==token)return;` +
      'const mode=event.data.mode==="light"?"light":"dark";' +
      'document.documentElement.setAttribute("data-color-mode",mode);' +
      'document.documentElement.style.colorScheme=mode;},true);' +
      'window.addEventListener("keydown",event=>{' +
      'if(event.key==="Escape"&&!event.isComposing&&event.keyCode!==229)' +
      `parent.postMessage({type:"${ESCAPE_MESSAGE}",token},"*");` +
      '},true);})();</script>'
    : '';
  return `<!doctype html><html data-color-mode="${colorMode}"><head><meta charset="utf-8">` +
    `<meta http-equiv="Content-Security-Policy" content="${policy}">` +
    '<meta name="referrer" content="no-referrer">' +
    '<meta name="color-scheme" content="light dark"></head><body>' +
    bridge + source + '</body></html>';
}

/** The sole execution boundary for untrusted Artifact HTML. */
export class DlActiveArtifactFrame extends LitElement {
  static properties = {
    source: {attribute: false},
    active: {type: Boolean},
    label: {type: String},
  };

  static styles = css`
    :host { display: block; height: 100%; min-height: 20rem; }
    .boundary {
      border-radius: var(--radius-content, 8px);
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 20rem;
      overflow: hidden;
    }
    iframe { background: white; border: 0; flex: 1; min-height: 18rem; width: 100%; }
  `;

  declare source: string | null;
  declare active: boolean;
  /** The caller's name for the frame; the fallback is drawn at render. */
  declare label: string | null;

  readonly #escapeToken = crypto.randomUUID();
  readonly #colorObserver = new MutationObserver(() => this.#sendColorMode());

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.source = null;
    this.active = false;
    this.label = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener('message', this.#receiveMessage);
    this.#colorObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-color-mode'],
    });
  }

  override disconnectedCallback(): void {
    this.#colorObserver.disconnect();
    window.removeEventListener('message', this.#receiveMessage);
    super.disconnectedCallback();
  }

  destroy(): void {
    this.source = null;
    this.renderRoot.querySelector('iframe')?.remove();
  }

  protected override render(): TemplateResult | typeof nothing {
    if (this.source === null) return nothing;
    return html`
      <div class="boundary">
        <iframe
          title=${this.label ?? msg('File HTML preview', {id: 'activeArtifactFrame.defaultLabel'})}
          sandbox=${this.active ? 'allow-scripts' : ''}
          referrerpolicy="no-referrer"
          allow=${PERMISSIONS}
          .srcdoc=${wrapperDocument(this.source, this.active, this.#escapeToken, hostColorMode())}
          @load=${this.#sendColorMode}
        ></iframe>
      </div>
    `;
  }

  /** Hand the frame the product's colour mode, at load and whenever the reader switches it. */
  #sendColorMode = (): void => {
    const iframe = this.renderRoot.querySelector('iframe');
    iframe?.contentWindow?.postMessage(
      {type: COLOR_MODE_MESSAGE, token: this.#escapeToken, mode: hostColorMode()},
      '*',
    );
  };

  #receiveMessage = (event: MessageEvent): void => {
    const iframe = this.renderRoot.querySelector('iframe');
    const payload = event.data as {type?: unknown; token?: unknown} | null;
    if (
      !iframe || event.source !== iframe.contentWindow || !payload
      || payload.type !== ESCAPE_MESSAGE || payload.token !== this.#escapeToken
    ) return;
    raise(this, 'dl-artifact-frame-escape');
  };
}

customElements.define('dl-active-artifact-frame', DlActiveArtifactFrame);

declare global {
  interface HTMLElementTagNameMap {
    'dl-active-artifact-frame': DlActiveArtifactFrame;
  }

  interface HTMLElementEventMap {
    'dl-artifact-frame-escape': CustomEvent<void>;
  }
}

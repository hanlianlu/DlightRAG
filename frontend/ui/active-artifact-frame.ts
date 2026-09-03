// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {css, html, LitElement, nothing, type TemplateResult} from 'lit';

const PERMISSIONS = [
  'accelerometer', 'ambient-light-sensor', 'autoplay', 'battery', 'camera',
  'clipboard-read', 'clipboard-write', 'display-capture', 'encrypted-media',
  'fullscreen', 'geolocation', 'gyroscope', 'hid', 'idle-detection',
  'local-fonts', 'magnetometer', 'microphone', 'midi', 'payment',
  'picture-in-picture', 'publickey-credentials-get', 'screen-wake-lock',
  'serial', 'speaker-selection', 'usb', 'web-share', 'xr-spatial-tracking',
].map((name) => `${name} 'none'`).join('; ');

const ESCAPE_MESSAGE = 'dl-artifact-frame-escape';

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

function wrapperDocument(source: string, active: boolean, escapeToken: string): string {
  const script = active ? "script-src 'unsafe-inline'" : "script-src 'none'";
  const policy = [BASE_CSP[0], script, ...BASE_CSP.slice(1)].join('; ');
  const escapeBridge = active
    ? `<script>(()=>{const token=${JSON.stringify(escapeToken)};` +
      'document.currentScript.remove();window.addEventListener("keydown",event=>{' +
      `if(event.key==="Escape")parent.postMessage({type:"${ESCAPE_MESSAGE}",token},"*");` +
      '},true);})();</script>'
    : '';
  return '<!doctype html><html><head><meta charset="utf-8">' +
    `<meta http-equiv="Content-Security-Policy" content="${policy}">` +
    '<meta name="referrer" content="no-referrer">' +
    '<meta name="color-scheme" content="light dark"></head><body>' +
    escapeBridge + source + '</body></html>';
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
  declare label: string;

  readonly #escapeToken = crypto.randomUUID();

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.source = null;
    this.active = false;
    this.label = msg('Artifact HTML preview', {id: 'activeArtifactFrame.defaultLabel'});
  }

  override connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener('message', this.#receiveMessage);
  }

  override disconnectedCallback(): void {
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
          title=${this.label}
          sandbox=${this.active ? 'allow-scripts' : ''}
          referrerpolicy="no-referrer"
          allow=${PERMISSIONS}
          .srcdoc=${wrapperDocument(this.source, this.active, this.#escapeToken)}
        ></iframe>
      </div>
    `;
  }

  #receiveMessage = (event: MessageEvent): void => {
    const iframe = this.renderRoot.querySelector('iframe');
    const payload = event.data as {type?: unknown; token?: unknown} | null;
    if (
      !iframe || event.source !== iframe.contentWindow || !payload
      || payload.type !== ESCAPE_MESSAGE || payload.token !== this.#escapeToken
    ) return;
    this.dispatchEvent(new CustomEvent('dl-artifact-frame-escape', {
      bubbles: true,
      composed: true,
    }));
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

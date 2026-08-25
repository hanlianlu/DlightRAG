// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {css, html, LitElement, nothing, type TemplateResult} from 'lit';

const PERMISSIONS = [
  'accelerometer', 'ambient-light-sensor', 'autoplay', 'battery', 'camera',
  'clipboard-read', 'clipboard-write', 'display-capture', 'encrypted-media',
  'fullscreen', 'geolocation', 'gyroscope', 'hid', 'idle-detection',
  'local-fonts', 'magnetometer', 'microphone', 'midi', 'payment',
  'picture-in-picture', 'publickey-credentials-get', 'screen-wake-lock',
  'serial', 'speaker-selection', 'usb', 'web-share', 'xr-spatial-tracking',
].map((name) => `${name} 'none'`).join('; ');

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

function wrapperDocument(source: string, active: boolean): string {
  const script = active ? "script-src 'unsafe-inline'" : "script-src 'none'";
  const policy = [BASE_CSP[0], script, ...BASE_CSP.slice(1)].join('; ');
  return '<!doctype html><html><head><meta charset="utf-8">' +
    `<meta http-equiv="Content-Security-Policy" content="${policy}">` +
    '<meta name="referrer" content="no-referrer">' +
    '<meta name="color-scheme" content="light dark"></head><body>' +
    source + '</body></html>';
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
      border: 2px solid var(--color-danger-border, #a66);
      border-radius: var(--radius-content, 8px);
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 20rem;
      overflow: hidden;
    }
    .notice {
      background: var(--color-danger-surface, #291b1b);
      color: var(--color-text-primary, #fff);
      font: 500 var(--font-size-caption, 0.8rem)/1.4 system-ui;
      padding: var(--space-tight, 0.5rem) var(--space-component, 0.75rem);
    }
    iframe { background: white; border: 0; flex: 1; min-height: 18rem; width: 100%; }
  `;

  declare source: string | null;
  declare active: boolean;
  declare label: string;

  constructor() {
    super();
    this.source = null;
    this.active = false;
    this.label = 'Artifact HTML preview';
  }

  destroy(): void {
    this.source = null;
    this.renderRoot.querySelector('iframe')?.remove();
  }

  protected override render(): TemplateResult | typeof nothing {
    if (this.source === null) return nothing;
    const mode = this.active ? 'Untrusted active preview' : 'Static HTML preview';
    return html`
      <div class="boundary">
        <div class="notice" role="status">${mode} · isolated from DlightRAG</div>
        <iframe
          title=${this.label}
          sandbox=${this.active ? 'allow-scripts' : ''}
          referrerpolicy="no-referrer"
          allow=${PERMISSIONS}
          .srcdoc=${wrapperDocument(this.source, this.active)}
        ></iframe>
      </div>
    `;
  }
}

customElements.define('dl-active-artifact-frame', DlActiveArtifactFrame);

declare global {
  interface HTMLElementTagNameMap {
    'dl-active-artifact-frame': DlActiveArtifactFrame;
  }
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The shared-control reference page: primitives and tokens, rendered live. */

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/primitives.css';
import '../styles/layout.css';
import '../styles/panels.css';
import '../styles/files.css';
import '../styles/sources.css';
import '../styles/artifacts.css';
import '../styles/design_system.css';
import './active_artifact_frame.ts';
import './artifact_canvas.ts';
import './answer_presentation.ts';
import './run_dialogs.ts';
import type {DlImageLightbox} from './image_lightbox.ts';
import './image_lightbox.ts';
import type {DlNotificationOffer} from './notifications.ts';
import './notifications.ts';
import './theme.ts';
import type {DlToastRegion} from './toast.ts';
import './toast.ts';

interface PrimitiveSpec {
  name: string;
  markup: string;
}

const PRIMITIVES: PrimitiveSpec[] = [
  {
    name: 'Button (.ui-btn)',
    markup:
      '<div class="ui-dialog-actions" style="justify-content:flex-start">' +
      '<button class="ui-btn" type="button">Standard</button>' +
      '<button class="ui-btn ui-btn-danger-text" type="button">Destructive text</button>' +
      '<button class="ui-btn" type="button" disabled>Disabled</button></div>',
  },
  {
    name: 'Checkbox (.ui-dialog-checkbox)',
    markup:
      '<label class="ui-dialog-checkbox">' +
      '<input type="checkbox" checked /> Activate profile memories</label>',
  },
  {
    name: 'Confirm dialog (.confirm-dialog)',
    markup:
      '<button class="ui-btn" id="ds-confirm-open" type="button">Open confirm dialog</button>' +
      '<dialog class="confirm-dialog" id="ds-confirm" aria-labelledby="ds-confirm-title">' +
      '<form method="dialog"><h2 id="ds-confirm-title">Delete all conversations?</h2>' +
      '<p>This action cannot be undone.</p>' +
      '<label class="ui-dialog-checkbox"><input type="checkbox" /> Also clear Profile memory</label>' +
      '<div class="ui-dialog-actions">' +
      '<button type="submit" value="cancel">Cancel</button>' +
      '<button type="submit" value="delete" class="ui-dialog-danger">Delete all</button>' +
      '</div></form></dialog>',
  },
  {
    name: 'Settings drawer (.settings-dialog)',
    markup:
      '<button class="ui-btn" id="ds-settings-open" type="button">Open settings drawer</button>' +
      '<dialog class="settings-dialog" id="ds-settings" aria-labelledby="ds-settings-title">' +
      '<form method="dialog"><div class="settings-drawer-body">' +
      '<div class="settings-header"><h2 id="ds-settings-title">Settings</h2>' +
      '<button class="panel-close settings-close" type="submit" aria-label="Close">✕</button></div>' +
      '<section class="settings-section"><h3>Profile Memory</h3>' +
      '<label class="ui-dialog-checkbox"><input type="checkbox" /> Activate profile memories</label>' +
      '<div class="settings-actions"><button class="ui-btn ui-btn-danger-text" type="button">Clear memory</button></div>' +
      '</section></div></form></dialog>',
  },
  {
    name: 'Action toast (dl-toast-region + .ui-btn)',
    markup:
      '<dl-toast-region class="toast ds-toast-demo" role="status"></dl-toast-region>',
  },
  {
    name: 'Theme Control (dl-theme-control)',
    markup: '<dl-theme-control id="ds-theme-feature"></dl-theme-control>',
  },
  {
    name: 'Answer notification offer (dl-notification-offer)',
    markup:
      '<dl-notification-offer class="notify-offer ds-notify-demo" role="group" ' +
      'aria-label="Answer notifications"></dl-notification-offer>',
  },
  {
    name: 'Answer Mode menu',
    markup:
      '<div class="composer-mode"><button class="composer-mode-trigger" type="button" ' +
      'aria-haspopup="menu" aria-expanded="true">Research</button>' +
      '<div class="composer-mode-menu ds-mode-demo" role="menu" aria-label="Answer mode">' +
      '<button role="menuitemradio" aria-checked="true">Research</button>' +
      '<button role="menuitemradio" aria-checked="false">Quick</button></div></div>',
  },
  {
    name: 'Image Lightbox (dl-image-lightbox)',
    markup:
      '<button class="ui-btn" id="ds-lightbox-open" type="button">Open image viewer</button>' +
      '<dl-image-lightbox id="ds-image-lightbox"></dl-image-lightbox>',
  },
  {
    name: 'Workspace controls',
    markup:
      '<div class="ui-popover" role="dialog" aria-label="Workspaces">' +
      '<button class="ui-popover-item" type="button" aria-pressed="true">All workspaces</button>' +
      '<div class="ui-popover-create"><input class="ui-popover-input" aria-label="New workspace name" ' +
      'placeholder="New workspace..."><button class="ui-popover-create-btn" aria-label="Create workspace">+</button>' +
      '</div></div>',
  },
  {
    name: 'Dialog text input (.ui-dialog-input)',
    markup: '<textarea class="ui-dialog-input" rows="2" placeholder="Type a workspace name..."></textarea>',
  },
  {
    name: 'Run continuation (dl-continuation-dialog)',
    markup:
      '<button class="ui-btn" id="ds-continuation-open" type="button">Open continuation dialog</button>' +
      '<dl-continuation-dialog></dl-continuation-dialog>',
  },
  {
    name: 'Child roster (dl-children-roster)',
    markup:
      '<button class="ui-btn" id="ds-roster-open" type="button">Open child roster</button>' +
      '<dl-children-roster></dl-children-roster>',
  },
  {
    name: 'Panel close (.panel-close)',
    markup: '<button class="panel-close" type="button" aria-label="Close panel">✕</button>',
  },
  {
    name: 'Artifact card and Canvas controls',
    markup:
      '<article class="answer-artifact-card"><div><strong>Quarterly report</strong>' +
      '<span>report.html</span></div><button class="ui-btn" type="button">View report</button></article>' +
      '<div class="artifact-canvas-actions"><button class="ui-btn" aria-pressed="true">Side</button>' +
      '<button class="ui-btn" aria-pressed="false">Wide</button>' +
      '<button class="ui-btn" aria-pressed="false">Fullscreen</button></div>',
  },
  {
    name: 'Opaque Artifact frame (dl-active-artifact-frame)',
    markup: '<dl-active-artifact-frame id="ds-artifact-frame"></dl-active-artifact-frame>',
  },
];

const TOKEN_GROUPS: Array<[string, string[]]> = [
  ['color-text-primary', ['color-accent-action', 'color-danger-text', 'color-text-muted']],
  ['color-bg-surface', ['color-bg-hover', 'color-danger-surface', 'color-accent-surface']],
  ['radius-control', ['radius-dialog', 'radius-docked']],
  ['font-size-body', ['font-size-detail', 'font-size-caption', 'font-size-subhead']],
  ['space-section', ['space-component', 'space-tight']],
];

function mountPrimitives(): void {
  const section = document.getElementById('ds-primitives');
  if (!section) return;
  for (const primitive of PRIMITIVES) {
    const card = document.createElement('article');
    card.className = 'ds-card';
    const title = document.createElement('h2');
    title.textContent = primitive.name;
    const stage = document.createElement('div');
    stage.className = 'ds-stage';
    stage.innerHTML = primitive.markup;
    card.append(title, stage);
    section.appendChild(card);
  }
  document.querySelector<DlToastRegion>('.ds-toast-demo')?.showAction(
    'Remembered: Prefers concise answers.',
    {
      actionLabel: 'Undo',
      onAction: async () => 'Profile Memory change undone.',
      duration: 3000,
    },
  );
  const notificationOffer = document.querySelector<DlNotificationOffer>('.ds-notify-demo');
  if (notificationOffer) notificationOffer.visible = true;
  document.getElementById('ds-confirm-open')?.addEventListener('click', () => {
    (document.getElementById('ds-confirm') as HTMLDialogElement | null)?.showModal();
  });
  document.getElementById('ds-settings-open')?.addEventListener('click', () => {
    (document.getElementById('ds-settings') as HTMLDialogElement | null)?.showModal();
  });
  document.getElementById('ds-continuation-open')?.addEventListener('click', () => {
    document.querySelector<HTMLElement & {open(kind: string): void}>('dl-continuation-dialog')
      ?.open('fork');
  });
  document.getElementById('ds-roster-open')?.addEventListener('click', () => {
    document.querySelector<HTMLElement & {open(fetcher: () => Promise<unknown[]>): void}>(
      'dl-children-roster',
    )?.open(async () => [{status: 'succeeded', objective: 'Example child'} as never]);
  });
  document.getElementById('ds-lightbox-open')?.addEventListener('click', (event) => {
    void document.querySelector<DlImageLightbox>('#ds-image-lightbox')?.open(
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=',
      event.currentTarget as HTMLElement,
    );
  });
  const frame = document.querySelector<HTMLElement & {source: string; active: boolean}>(
    '#ds-artifact-frame',
  );
  if (frame) {
    frame.source = '<main><h1>Static Artifact example</h1><p>Scripts are disabled here.</p></main>';
    frame.active = false;
  }
}

function mountTokens(): void {
  const section = document.getElementById('ds-tokens');
  if (!section) return;
  const card = document.createElement('article');
  card.className = 'ds-card';
  const title = document.createElement('h2');
  title.textContent = 'Design tokens';
  card.appendChild(title);
  const root = getComputedStyle(document.documentElement);
  for (const [primary, rest] of TOKEN_GROUPS) {
    const group = document.createElement('div');
    group.className = 'ds-token-group';
    for (const name of [primary, ...rest]) {
      const row = document.createElement('div');
      row.className = 'ds-token-row';
      const swatch = document.createElement('span');
      swatch.className = 'ds-token-swatch';
      swatch.style.background = `var(--${name})`;
      const label = document.createElement('code');
      label.textContent = `--${name}: ${root.getPropertyValue(`--${name}`).trim()}`;
      row.append(swatch, label);
      group.appendChild(row);
    }
    card.appendChild(group);
  }
  section.appendChild(card);
}

mountPrimitives();
mountTokens();

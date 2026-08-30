// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Product-component showcase; deliberately separate from the design-system catalog. */

import {html, render} from 'lit';
import '../design-system/index.css';
import '../styles/app.css';
import '../styles/design_system.css';
import {defineDesignSystemElements} from '../design-system/index.ts';
import './notifications.ts';
import './theme.ts';
import type {DlToastRegion} from './toast.ts';
import './toast.ts';

defineDesignSystemElements();

const host = document.getElementById('product-showcase');
if (host) {
  render(html`
    <article class="ds-card">
      <h2>Theme feature</h2>
      <div class="ds-stage"><dl-theme-control></dl-theme-control></div>
    </article>
    <article class="ds-card">
      <h2>Product notification surfaces</h2>
      <div class="ds-stage">
        <dl-notification-offer class="notify-offer" role="group"
          aria-label="Answer notifications"></dl-notification-offer>
        <dl-toast-region class="toast ds-toast-demo" role="status"></dl-toast-region>
      </div>
    </article>
  `, host);
}

document.querySelector<DlToastRegion>('dl-toast-region')?.show('Product showcase ready.', 5000);

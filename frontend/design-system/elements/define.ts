// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Explicit, idempotent registration for stateful design-system elements. */

import {DlIconButton} from './icon-button.ts';
import {DlMenu} from './menu.ts';
import {DlSplitLayout} from './split-layout.ts';

export function defineDesignSystemElements(registry: CustomElementRegistry = customElements): void {
  if (!registry.get('dl-split-layout')) registry.define('dl-split-layout', DlSplitLayout);
  if (!registry.get('dl-icon-button')) registry.define('dl-icon-button', DlIconButton);
  if (!registry.get('dl-menu')) registry.define('dl-menu', DlMenu);
}

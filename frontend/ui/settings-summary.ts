// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What a Settings page tells the dialog's navigation about itself.
 *
 * Every page lives in its own element and owns its data. The navigation row beside a page shows a
 * short status of that data without asking for it: a page reports a typed summary whenever the
 * data behind it changes, and the dialog keeps the latest one per section and finds the words.
 */

import type {LanguagePreference} from '../lib/language.ts';

export type SettingsSection =
  | 'connections'
  | 'agent-accounts'
  | 'memory'
  | 'conversations'
  | 'language';

export type SettingsSummary =
  | {section: 'connections'; enabled: number; total: number}
  | {section: 'agent-accounts'; count: number}
  /** `enabled` is null while the switch is unread, and `count` is null while Memory is off. */
  | {section: 'memory'; enabled: boolean | null; count: number | null}
  | {section: 'conversations'; count: number}
  | {section: 'language'; preference: LanguagePreference};

/** Tell the dialog what this page's navigation row should say now. */
export function reportSettingsSummary(host: HTMLElement, summary: SettingsSummary): void {
  host.dispatchEvent(new CustomEvent<SettingsSummary>('dl-settings-summary', {
    detail: summary,
    bubbles: true,
    composed: true,
  }));
}

declare global {
  interface HTMLElementEventMap {
    'dl-settings-summary': CustomEvent<SettingsSummary>;
  }
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The pieces more than one Settings page draws, so each is drawn one way. */

import {html, type TemplateResult} from 'lit';
import styles from '../styles/settings-page.module.css';

export interface SwitchCard {
  /** The switch's own id; the ids of its two texts derive from it. */
  id: string;
  label: string;
  caption: string;
  checked: boolean;
  disabled?: boolean;
  /** The switch is off for a reason the owner cannot change, so its label reads quieter. */
  muted?: boolean;
  onToggle: (event: Event) => void;
}

/** A card holding one switch and the sentence that says what it does now.
 *
 * The card is the switch's label, so a tap anywhere on it turns the switch: the control itself is
 * a few pixels tall, and the row it sits in is the target a finger can find. The switch keeps its
 * own name and description for assistive technology.
 */
export function switchCard(card: SwitchCard): TemplateResult {
  const label = `${card.id}-label`;
  const caption = `${card.id}-caption`;
  return html`
    <label class="${styles.card} ${styles.row}">
      <span class=${styles.rowText}>
        <span id=${label} class="${styles.rowLabel} ${card.muted ? styles.rowLabelMuted : ''}"
          >${card.label}</span>
        <span id=${caption} class=${styles.rowCaption}>${card.caption}</span>
      </span>
      <button id=${card.id} class="dl-switch dl-switch--sm" type="button" role="switch"
        aria-checked=${String(card.checked)} aria-labelledby=${label} aria-describedby=${caption}
        ?disabled=${card.disabled} @click=${card.onToggle}></button>
    </label>`;
}

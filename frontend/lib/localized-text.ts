// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Client-owned words carried as their English source plus the catalog id that
 *  translates them, so a view holds no resolved language: a component applies the
 *  words at render and a language change redraws them. Text the server wrote carries
 *  no id and shows as it arrived. `ToolRow` states the same contract as verb/verbId. */

import {msg} from '@lit/localize';

export interface LocalizedText {
  /** The English source, which is also what a catalog miss falls back to. */
  readonly source: string;
  /** The catalog id, or null for text the server wrote. */
  readonly id: string | null;
}

/** The words to draw, in the language showing now. */
export function localizedText(text: LocalizedText): string {
  return text.id === null ? text.source : msg(text.source, {id: text.id});
}

/** The same helper for the words a component owns outright. */
export function catalogText(source: string, id: string): LocalizedText {
  return {source, id};
}

/** Text the server wrote, shown as it arrived. */
export function serverText(source: string): LocalizedText {
  return {source, id: null};
}

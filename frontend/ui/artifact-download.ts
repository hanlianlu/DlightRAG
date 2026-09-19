// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The one download control for a published Artifact.

 *  A download URL is only usable when it is this origin's; anything else renders
 *  inert rather than becoming a dead or cross-origin link. `download` keeps the
 *  save-to-file contract the delivery layer already set on the response.
 */

import {html, nothing, type TemplateResult} from 'lit';
import {safeSameOriginHref} from '../lib/urls.ts';

export function artifactDownloadLink(
  downloadUrl: string | null | undefined,
  content: unknown,
  className = 'dl-btn',
): TemplateResult | typeof nothing {
  if (!downloadUrl) return nothing;
  return html`
    <a class=${className} href=${safeSameOriginHref(downloadUrl) || '#'} download>
      ${content}
    </a>`;
}

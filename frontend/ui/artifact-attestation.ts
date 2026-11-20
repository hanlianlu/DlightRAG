// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One Artifact's attestation record, and the two shapes a reader meets it in.
 *
 * The server derives the record from the Run's own durable records -- which Skill the Run loaded,
 * which command it ran -- and never from anything the model wrote, so the record repeats a fact
 * about the Run and never a claim the answer made about itself. The three statuses are the whole
 * vocabulary, and each says one thing:
 *
 * * `attested` names the Skill, and the verifier script a recorded command named. That is the
 *   strongest thing the records establish; it is not proof the verifier produced this file, so the
 *   verifier is drawn beside the claim wherever the reader sees the claim's basis.
 * * `unattested` names why, in the server's own closed vocabulary, because the reason is the
 *   finding: "not attested" alone leaves the reader to guess at an accusation.
 * * `unknown` is not a finding at all. It says the records cannot settle the question, so it is
 *   never dressed as a failure -- no danger tone, no warning role.
 *
 * The two shapes divide the work by surface. The Canvas draws one Artifact to read, so its badge is
 * the status alone: the reason, the verifier and the skipped reads belong to the Answer. The Answer
 * draws the Run's delivery record once, under the placements it covers, as one line and one
 * disclosure: the reason and the per-read evidence fold behind it, because they are evidence beside
 * the verdict rather than a second verdict, and a line under a reader's files has to stay a line. A
 * record can also carry `requiredReadsMissing`: documents a loaded Skill says a Run must read and
 * this Run did not read to the end, one line each -- `not read: <path> (<when>)`. It is evidence
 * beside the verdict and never part of it, so it is drawn in the same neutral tone and with no alert
 * role: a Run that skipped a reading is not thereby accused of anything the status says. A Skill
 * that declares no verifier made no claim for the status to judge, so for it the missing-read lines
 * are the whole record -- no "Not attested", no reason -- and the server publishes that record only
 * because a read is missing.
 *
 * `null` is not a fourth status: it means the server published no record to draw, so a surface draws
 * nothing at all. That covers both "the server derived nothing" (an older payload, or a surface that
 * does not derive one) and "the server derived a verdict there is no claim behind" -- a Skill that
 * declares no verifier for what it produces and a Run that read everything it declared. The two are
 * deliberately one value here, because the reader-facing consequence is the same either way: nothing
 * was checked and nothing was withheld. Drawing "not attested" for either would accuse a Run the
 * server never found fault with.
 */

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import type {ArtifactAttestation} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';

/** What each reason in the server's closed vocabulary says in a reader's words. The tokens are the
 *  ones `application/answer/provenance.py` publishes; `ui/i18n-catalog.structure.test.ts` holds
 *  this map and the catalog to each other, so a reason cannot arrive without copy. */
const REASON_TEXT: Record<string, string> = {
  no_skill_loaded: 'the Run loaded no Agent Skill at all',
  skill_loaded_without_verifier: 'no command recorded in this Run named the verifier the Skill declares today',
  skill_has_no_verifier_contract: 'the Skill that was loaded declares no verifier for what it produces',
  contract_unavailable: 'a Skill the Run loaded is no longer in the catalog, so its declaration cannot be read',
  verifier_reported_failures: 'the Skill\'s verifier ran in this Run and reported failures',
  records_pruned: 'the records read here are not the whole Run, so no verifier can be ruled in or out',
  skill_name_unavailable: 'a Skill load is recorded without the Skill\'s name',
  arguments_truncated:
    'part of the Run\'s record is missing, so whether the Skill\'s verifier ran cannot be confirmed',
};

function statusText(attestation: ArtifactAttestation): string | TemplateResult {
  if (attestation.status === 'attested') {
    return attestation.skill === null
      ? msg('Attested', {id: 'artifactAttestation.attested'})
      : msg(str`Attested by ${attestation.skill}`, {id: 'artifactAttestation.attestedBy'});
  }
  if (attestation.status === 'unattested') {
    return msg('Not attested', {id: 'artifactAttestation.notAttested'});
  }
  return msg('Attestation unknown', {id: 'artifactAttestation.unknown'});
}

/** What the Run's delivery record folds under its disclosure: the verifier an attested record
 *  rests on, or the reason every other status must name. A reason this client does not know is
 *  shown as the server's own token, which is the thing the wire actually said. */
function detailText(attestation: ArtifactAttestation): string | TemplateResult | null {
  if (attestation.status === 'attested') {
    return attestation.verifier === null
      ? null
      : msg(str`verifier ${attestation.verifier}`, {id: 'artifactAttestation.verifier'});
  }
  const reason = attestation.reason;
  if (reason === null) return null;
  const known: string | undefined = REASON_TEXT[reason];
  return known === undefined
    ? msg(str`the server gave a reason this client does not know (${reason})`, {
      id: 'artifactAttestation.reasonUnrecognised',
    })
    : msg(known, {id: `artifactAttestation.reason.${reason}`});
}

/** One line per declared read the Run did not make. Neutral on purpose: it states what the records
 *  show and names the Skill author's own words for when the read was due. */
function notReadLines(attestation: ArtifactAttestation): TemplateResult[] {
  return (attestation.requiredReadsMissing ?? []).map((missing) => {
    const path = missing.paths.join(' or ');
    // A skipped read can belong to another loaded Skill: the Run-level record is stamped with
    // the attesting Skill, so name the declaring Skill whenever the two differ.
    const declared = missing.skill === attestation.skill ? '' : `; declared by ${missing.skill}`;
    return html`<span class="artifact-attestation-detail artifact-attestation-unread">${
      msg(str`not read: ${path} (${missing.when}${declared})`, {id: 'artifactAttestation.notRead'})
    }</span>`;
  });
}

/** A record whose only finding is a skipped read: the loaded Skill made no verifier claim, so there is
 *  no status to name and no reason to give and the unread lines are the whole record. */
function isClaimlessRecord(attestation: ArtifactAttestation): boolean {
  return attestation.status === 'unattested' && attestation.reason === 'skill_has_no_verifier_contract';
}

/** The status alone, for the Canvas header.
 *
 * The Canvas draws one Artifact to read, so the record there is the one word the records settle on:
 * the reason, the verifier and the skipped-read lines are the Answer's delivery record to draw, once
 * for the Run. A record that made no verifier claim has no status to name and draws nothing here,
 * and neither does `null` (see the module comment). */
export function artifactAttestationBadge(
  attestation: ArtifactAttestation | null,
): TemplateResult | typeof nothing {
  if (attestation === null || isClaimlessRecord(attestation)) return nothing;
  return html`
    <p class="artifact-attestation" data-attestation=${attestation.status}>
      <span class="artifact-attestation-status">${statusText(attestation)}</span>
    </p>
  `;
}

/** What a reader can ask of the Run's record: whether its evidence is showing. */
export interface DeliveryRecordOptions {
  /** The reason and the per-read evidence start folded: the record is one quiet line until a
   *  reader asks what it rests on. */
  expanded: boolean;
  /** The element id the disclosure names in `aria-controls`. */
  evidenceId: string;
  /** Flip the evidence. The Answer owns the state, so a redraw does not open it again. */
  onToggle: () => void;
}

/** The Run's delivery record, drawn once under the files it covers.
 *
 * `stamp_projected_artifacts` puts one record on every Artifact of a Run -- the same verdict, the
 * same reason, the same skipped reads -- so a record drawn inside each card says one Run-level
 * sentence as many times as the Answer lists files. The Answer draws it once instead, and as one
 * line: the verdict, then one disclosure. The reason and the per-read lines fold behind it -- the
 * count when reads are missing, "Details" when only the reason or the verifier is -- because they
 * are evidence beside the verdict and not a second verdict, and the line under a reader's files has
 * to stay a line. Opening the disclosure draws them exactly as the server published them.
 *
 * `null` (the server derived no record) and a claimless record with no skipped read draw nothing,
 * exactly as the per-Artifact badge did: both mean the reader is owed nothing here.
 */
export function artifactDeliveryRecord(
  attestation: ArtifactAttestation | null,
  options: DeliveryRecordOptions,
): TemplateResult | typeof nothing {
  if (attestation === null) return nothing;
  const unread = notReadLines(attestation);
  const claimless = isClaimlessRecord(attestation);
  if (claimless && unread.length === 0) return nothing;
  const status = claimless ? null : statusText(attestation);
  const detail = claimless ? null : detailText(attestation);
  const folded = unread.length > 0 || detail !== null;
  const evidenceLabel = unread.length > 0
    ? (unread.length === 1
      ? msg('1 read skipped', {id: 'artifactAttestation.oneReadSkipped'})
      : msg(str`${unread.length} reads skipped`, {id: 'artifactAttestation.readsSkipped'}))
    : msg('Details', {id: 'artifactAttestation.details'});
  // A claimless record has no verdict to lead with: the count is the whole line and the control
  // sits where the words would, not pushed to the far edge of it.
  const solo = status === null;
  const toggleClass = [
    'dl-text-action',
    'dl-text-action--disclosure',
    'answer-artifact-record-toggle',
    ...(solo ? ['answer-artifact-record-toggle--solo'] : []),
  ].join(' ');
  return html`
    <div class="answer-artifact-record">
      <p class="artifact-attestation" data-attestation=${attestation.status}>
        <span class="answer-artifact-record-mark" aria-hidden="true"></span>
        ${status === null ? nothing : html`<span class="artifact-attestation-status">${status}</span>`}
        ${folded ? html`
          <button class=${toggleClass} type="button"
                  aria-expanded=${String(options.expanded)} aria-controls=${options.evidenceId}
                  @click=${options.onToggle}>
            ${evidenceLabel}
            ${icon('chevron-down', {size: 'xs'})}
          </button>` : nothing}
      </p>
      ${folded ? html`
        <div class="answer-artifact-record-evidence" id=${options.evidenceId} ?hidden=${!options.expanded}>
          ${detail === null ? nothing : html`<p class="answer-artifact-record-reason">${detail}</p>`}
          ${unread}
        </div>` : nothing}
    </div>
  `;
}

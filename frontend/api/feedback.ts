// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What a reader thought of one answered message: the judgement the server keeps, and the two calls that
 *  give it and take it back.
 *
 * A judgement names its message by the Run, the agent that wrote it (`main` for the Run's own agent, else
 * a Child Session's id) and the message's own `sequence` in that agent's Session. The server states the
 * sequence of a final answer with the turn or the child, so the page never works one out.
 */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {apiError, parseWire} from './wire.ts';

/** The reasons a thumb down is given, in the order the server keeps them and the reader is offered them. */
export const FEEDBACK_REASONS = [
  'unsupported',
  'factual_error',
  'too_verbose',
  'off_topic',
  'irrelevant_citations',
] as const;
export type FeedbackReason = (typeof FEEDBACK_REASONS)[number];

/** The longest note the server keeps, in characters. */
export const FEEDBACK_NOTE_LIMIT = 2000;

/** The reader's judgement of one message, as a turn or a child states it and as a rating answers. A thumb up
 *  carries no reasons and no note; a thumb down says why in at least one reason or in a note. */
export const answerJudgement = v.object({
  value: v.picklist(['up', 'down']),
  reasons: v.array(v.picklist(FEEDBACK_REASONS)),
  note: v.nullable(v.string()),
});
export type AnswerJudgement = v.InferOutput<typeof answerJudgement>;

/** What a rating asks the server to keep. */
interface AnswerRating {
  readonly value: AnswerJudgement['value'];
  readonly reasons: readonly FeedbackReason[];
  readonly note: string | null;
}

/** Keep the judgement of one message, replacing an earlier one whole; the answer is what was kept. */
export async function putAnswerFeedback(
  runId: string,
  agent: string,
  sequence: number,
  rating: AnswerRating,
  signal?: AbortSignal,
): Promise<AnswerJudgement> {
  const response = await fetch(`/web/api/answer/${encodeURIComponent(runId)}/feedback`, {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({
      agent,
      sequence,
      value: rating.value,
      reasons: rating.reasons,
      note: rating.note,
    }),
    signal,
  });
  // The answer also echoes the message and when it was judged; the page needs only the judgement.
  return parseWire(response, answerJudgement);
}

/** Take the judgement of one message back. The server answers 204 whether or not there was one. */
export async function deleteAnswerFeedback(
  runId: string,
  agent: string,
  sequence: number,
  signal?: AbortSignal,
): Promise<void> {
  const query = new URLSearchParams({agent, sequence: String(sequence)});
  const response = await fetch(
    `/web/api/answer/${encodeURIComponent(runId)}/feedback?${query.toString()}`,
    {method: 'DELETE', headers: csrfHeaders(), signal},
  );
  if (!response.ok) throw await apiError(response);
}

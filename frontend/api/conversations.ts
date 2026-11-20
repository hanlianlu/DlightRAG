// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {type AgentId, readAgents } from '../lib/agent-id.ts';
import {csrfHeaders} from './csrf.ts';
import {answerJudgement} from './feedback.ts';
import {videoPlaybackLink} from './video-playback.ts';
import {webCommandError} from './web-command-error.ts';
import {apiError, parseWire} from './wire.ts';

const conversationSummary = v.pipe(
  v.object({
    conversation_id: v.string(),
    title: v.nullable(v.string()),
    created_at: v.string(),
    updated_at: v.string(),
    forked_from_conversation_id: v.optional(v.nullable(v.string())),
    forked_from_title: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    conversationId: w.conversation_id,
    title: w.title,
    createdAt: w.created_at,
    updatedAt: w.updated_at,
    forkedFromConversationId: w.forked_from_conversation_id ?? null,
    forkedFromTitle: w.forked_from_title ?? null,
  })),
);
export type ConversationSummary = v.InferOutput<typeof conversationSummary>;

const attachmentReference = v.pipe(
  v.object({
    attachment_id: v.string(),
    ordinal: v.number(),
    kind: v.string(),
    filename: v.string(),
    mime_type: v.string(),
    byte_size: v.number(),
    url: v.string(),
    thumbnail_url: v.nullable(v.string()),
    label: v.string(),
  }),
  v.transform((w) => ({
    attachmentId: w.attachment_id,
    ordinal: w.ordinal,
    kind: w.kind,
    filename: w.filename,
    mimeType: w.mime_type,
    byteSize: w.byte_size,
    url: w.url,
    thumbnailUrl: w.thumbnail_url,
    label: w.label,
  })),
);
export type ConversationAttachmentReference = v.InferOutput<typeof attachmentReference>;

const presentationImage = v.pipe(
  v.object({
    id: v.string(),
    chunk_id: v.string(),
    source_ref: v.string(),
    url: v.string(),
    thumbnail_url: v.string(),
    label: v.string(),
    answer_image_sent: v.boolean(),
  }),
  v.transform((w) => ({
    id: w.id,
    chunkId: w.chunk_id,
    sourceRef: w.source_ref,
    url: w.url,
    thumbnailUrl: w.thumbnail_url,
    label: w.label,
    answerImageSent: w.answer_image_sent,
  })),
);
export type PresentationImage = v.InferOutput<typeof presentationImage>;

const artifactIssue = v.object({
  kind: v.string(),
  description: v.string(),
  resource_id: v.nullable(v.string()),
});

const artifactOutcome = v.pipe(
  v.object({
    status: v.picklist(['complete', 'partial', 'failed']),
    issues: v.array(artifactIssue),
  }),
  v.transform((w) => ({
    status: w.status,
    issues: w.issues.map((issue) => ({
      kind: issue.kind,
      description: issue.description,
      resourceId: issue.resource_id,
    })),
  })),
);

/** One document a loaded Skill says a Run must read and this Run did not read to the end: the
 *  Skill, the Skill-relative files that would have satisfied it (several means any one would), and
 *  the Skill author's own words for the step or condition. */
const missingRead = v.object({
  skill: v.string(),
  paths: v.array(v.string()),
  when: v.string(),
});

/** What the Run's own records settle about whether an Agent Skill attested this Artifact, derived
 *  by the server and never declared by the model.
 *
 *  `status` is the three values the contract names; a value this client has not met reads as
 *  `unknown`, because "cannot be told" is the honest reading of a vocabulary the server has moved
 *  past and a badge must not take the whole Answer down with it. `reason` stays a plain string for
 *  the same reason: the closed vocabulary is the server's, and a token added since this client was
 *  built is shown as itself rather than dropped.
 *
 *  `required_reads_missing` is evidence beside that verdict and never part of it. `null` means the
 *  records could not settle it (and an absent field, or a value this client cannot read, is the
 *  same honest answer); an empty list means every declared read was made; an entry names a read
 *  that was not. The browser model keeps it as `requiredReadsMissing`, always present. */
const artifactAttestation = v.pipe(
  v.object({
    status: v.fallback(v.picklist(['attested', 'unattested', 'unknown']), 'unknown'),
    skill: v.nullable(v.string()),
    verifier: v.nullable(v.string()),
    reason: v.nullable(v.string()),
    basis: v.nullable(v.string()),
    required_reads_missing: v.fallback(v.optional(v.nullable(v.array(missingRead))), null),
  }),
  v.transform((w) => ({
    status: w.status,
    skill: w.skill,
    verifier: w.verifier,
    reason: w.reason,
    basis: w.basis,
    requiredReadsMissing: w.required_reads_missing ?? null,
  })),
);
export type ArtifactAttestation = v.InferOutput<typeof artifactAttestation>;

const answerArtifact = v.pipe(
  v.object({
    resource_id: v.string(),
    media_type: v.string(),
    label: v.string(),
    filename: v.string(),
    byte_size: v.number(),
    digest: v.string(),
    presentation: v.picklist(['image', 'video', 'markdown', 'html', 'pdf', 'text', 'download']),
    status: v.picklist(['available', 'unavailable']),
    uri: v.string(),
    width: v.nullable(v.number()),
    height: v.nullable(v.number()),
    data_url: v.nullable(v.string()),
    download_url: v.nullable(v.string()),
    presentation_url: v.nullable(v.string()),
    issue: v.nullable(artifactIssue),
    /** The server's record for this Artifact, or null when it derived none. */
    attestation: v.nullable(artifactAttestation),
  }),
  v.transform((w) => ({
    resourceId: w.resource_id,
    mediaType: w.media_type,
    label: w.label,
    filename: w.filename,
    byteSize: w.byte_size,
    digest: w.digest,
    presentation: w.presentation,
    status: w.status,
    uri: w.uri,
    width: w.width,
    height: w.height,
    dataUrl: w.data_url,
    downloadUrl: w.download_url,
    presentationUrl: w.presentation_url,
    issue: w.issue === null ? null : {
      kind: w.issue.kind,
      description: w.issue.description,
      resourceId: w.issue.resource_id,
    },
    /** `null` means the server published no record to draw -- a surface that does not derive one, or
     *  a verdict with no claim behind it because the loaded Skill declares no verifier. It is never
     *  "the Run loaded no Skill": that is a record with its own reason. */
    attestation: w.attestation,
  })),
);
export type AnswerArtifact = v.InferOutput<typeof answerArtifact>;

/** Page-declared video metadata (ADR 0026); playback permission is separate (ADR 0028). */
const linkCard = v.pipe(
  v.object({
    url: v.string(),
    title: v.string(),
    description: v.string(),
    site: v.string(),
    image: v.nullable(v.string()),
  }),
);
export type LinkCard = v.InferOutput<typeof linkCard>;

const presentationPart = v.pipe(
  v.object({
    type: v.picklist(['markdown', 'artifact', 'evidence_image']),
    text: v.string(),
    html: v.string(),
    artifact: v.nullable(answerArtifact),
    evidence_image: v.nullable(presentationImage),
    inline: v.boolean(),
    slot: v.optional(v.nullable(v.pipe(v.number(), v.integer(), v.minValue(0)))),
  }),
  v.transform((w) => ({
    type: w.type,
    text: w.text,
    html: w.html,
    artifact: w.artifact,
    evidenceImage: w.evidence_image,
    inline: w.inline,
    ...(w.slot == null ? {} : {slot: w.slot}),
  })),
);
export type PresentationPart = v.InferOutput<typeof presentationPart>;

const presentationSourceChunk = v.pipe(
  v.object({
    chunk_idx: v.nullable(v.number()),
    page_number: v.nullable(v.number()),
    content_html: v.string(),
    image_url: v.nullable(v.string()),
    thumbnail_url: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    chunkIdx: w.chunk_idx,
    pageNumber: w.page_number,
    contentHtml: w.content_html,
    imageUrl: w.image_url,
    thumbnailUrl: w.thumbnail_url,
  })),
);

const presentationSource = v.pipe(
  v.object({
    id: v.string(),
    title: v.string(),
    source_url: v.nullable(v.string()),
    download_url: v.nullable(v.string()),
    chunks: v.array(presentationSourceChunk),
  }),
  v.transform((w) => ({
    id: w.id,
    title: w.title,
    sourceUrl: w.source_url,
    downloadUrl: w.download_url,
    chunks: w.chunks,
  })),
);
export type PresentationSource = v.InferOutput<typeof presentationSource>;

const answerPresentation = v.pipe(
  v.object({
    answer_text: v.string(),
    parts: v.array(presentationPart),
    sources: v.array(presentationSource),
    link_cards: v.optional(v.array(linkCard), []),
    video_links: v.optional(v.array(videoPlaybackLink), []),
    evidence_images: v.array(presentationImage),
    artifacts: v.array(answerArtifact),
    artifact_outcome: artifactOutcome,
  }),
  v.transform((w) => ({
    answerText: w.answer_text,
    ...(w.video_links.length ? {videoLinks: w.video_links} : {}),
    parts: w.parts,
    sources: w.sources,
    linkCards: w.link_cards,
    evidenceImages: w.evidence_images,
    artifacts: w.artifacts,
    artifactOutcome: w.artifact_outcome,
  })),
);
export type AnswerPresentation = v.InferOutput<typeof answerPresentation>;

/** Return the typed presentation a live `done` event carries, or null for a
 *  payload this client cannot interpret. */
export function parseAnswerPresentation(wire: unknown): AnswerPresentation | null {
  const parsed = v.safeParse(answerPresentation, wire);
  return parsed.success ? parsed.output : null;
}

const agentChildStatus = v.pipe(
  v.object({
    child_session_id: v.optional(v.string()),
    status: v.string(),
    objective: v.optional(v.nullable(v.string())),
    model_role: v.optional(v.nullable(v.string())),
    usage: v.optional(v.nullable(v.record(v.string(), v.number()))),
    operation_id: v.optional(v.nullable(v.string())),
    operation_sequence: v.optional(v.nullable(v.number())),
    operation_status: v.optional(v.nullable(v.string())),
    cancellation_origin: v.optional(v.nullable(v.string())),
    summary: v.optional(v.nullable(v.string())),
    result_handles: v.optional(v.array(v.string())),
    started_at: v.optional(v.nullable(v.string())),
    finished_at: v.optional(v.nullable(v.string())),
    pending_questions: v.optional(v.nullable(v.number())),
    result_sequence: v.optional(v.nullable(v.number())),
    feedback: v.optional(v.nullable(answerJudgement)),
  }),
  v.transform((w) => ({
    childSessionId: w.child_session_id,
    status: w.status,
    objective: w.objective ?? undefined,
    modelRole: w.model_role ?? undefined,
    usage: w.usage ?? null,
    operationId: w.operation_id ?? null,
    operationSequence: w.operation_sequence ?? null,
    operationStatus: w.operation_status ?? null,
    cancellationOrigin: w.cancellation_origin ?? null,
    summary: w.summary ?? null,
    resultHandles: w.result_handles ?? [],
    /** When the child's current operation started and (once it has settled) finished, ISO-8601 UTC. */
    startedAt: w.started_at ?? null,
    finishedAt: w.finished_at ?? null,
    /** Questions to the parent that are still waiting and not yet expired. */
    pendingQuestions: w.pending_questions ?? 0,
    /** The `sequence` of the child's final answer message in its own Session, which a judgement names;
     *  null while the child has not answered. */
    resultSequence: w.result_sequence ?? null,
    /** The reader's judgement of that message, null when they have given none. */
    feedback: w.feedback ?? null,
  })),
);
export type AgentChildStatus = v.InferOutput<typeof agentChildStatus>;

/** One tool call of an assistant message: the server sends its id, name, and the Tool's own bounded
 *  subject for the call (a command, query or path label, never its raw argument or result payload). */
const toolCall = v.pipe(
  v.object({
    id: v.optional(v.nullable(v.string())),
    name: v.optional(v.nullable(v.string())),
    /** The bounded, tool-reported subject from the Run's event log; left out when the Tool derived none. */
    subject: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({id: w.id ?? '', name: w.name ?? '', subject: w.subject ?? ''})),
);

const activityMessage = v.pipe(
  v.object({
    sequence: v.number(),
    role: v.string(),
    content: v.optional(v.unknown()),
    tool_calls: v.optional(v.array(toolCall)),
    tool_call_id: v.optional(v.string()),
    name: v.optional(v.string()),
    is_error: v.optional(v.boolean()),
    /** Which agent of the Run the message came from, when a search spanned more than one. The wire
     *  says this three ways; `ActivityMessage` states it as one value, decided in the mapper. */
    agent: v.optional(v.nullable(v.string())),
    /** What that agent is called: its task, which the dock may not have loaded itself. */
    agent_label: v.optional(v.nullable(v.string())),
    /** The words in their neighbourhood, for a hit rather than a whole message. */
    snippet: v.optional(v.string()),
    /** When the entry behind the message was written, ISO-8601 UTC; left out of a message a search
     *  built rather than read. */
    at: v.optional(v.nullable(v.string())),
    /** The model's reasoning for this assistant message, masked at the server edge and bounded there.
     *  Left out of every message that is not an assistant turn, and of a page that does not state it. */
    reasoning: v.optional(v.string()),
  }),
  v.transform((w) => ({
    /** Where the message sits in its Session: the cursor of the page before it. */
    sequence: w.sequence,
    role: w.role,
    content: typeof w.content === 'string' ? w.content : '',
    agent: w.agent,
    agentLabel: w.agent_label ?? null,
    snippet: w.snippet ?? '',
    toolCalls: w.tool_calls ?? [],
    toolCallId: w.tool_call_id ?? '',
    name: w.name ?? '',
    isError: w.is_error ?? false,
    at: w.at ?? null,
    reasoning: w.reasoning ?? '',
  })),
);
export type ActivityMessage = Omit<v.InferOutput<typeof activityMessage>, 'agent'> & {
  /** Who wrote it, as one value: see `agent-id.ts` for why the wire cannot be trusted to say. */
  agent: AgentId;
};

/** One page of an agent's transcript, oldest message first; the newest page has no cursor. */
const activityPage = v.pipe(
  v.object({
    messages: v.array(activityMessage),
    next_before: v.nullable(v.number()),
    running: v.boolean(),
  }),
  v.transform((w) => ({
    messages: w.messages,
    nextCursor: w.next_before === null ? null : String(w.next_before),
    /** Whether the agent is still working, so a call without a result is in flight. */
    running: w.running,
  })),
);
export type ActivityPage = Omit<v.InferOutput<typeof activityPage>, 'messages'> & {
  messages: ActivityMessage[];
};
/** One mechanical decision the Run states about a step, as the server judges it. */
const decisionMark = v.picklist(['failed', 'degraded', 'repeated', 'redirect']);
export type DecisionMark = v.InferOutput<typeof decisionMark>;

/** The kinds a Run-timeline step can be; a page and the step-detail read state the same set. */
const TIMELINE_KINDS = ['thinking', 'say', 'tool', 'answer', 'instruction', 'delivery'] as const;
export type TimelineKind = (typeof TIMELINE_KINDS)[number];

/** How many steps of a Run state each mechanical decision, keyed by the four marks. The keys are
 *  always present, because a Run that states no mark states four zeroes rather than a narrower
 *  object a row would have to guess at. */
const runMarkCounts = v.pipe(
  v.object({
    failed: v.optional(v.number(), 0),
    degraded: v.optional(v.number(), 0),
    repeated: v.optional(v.number(), 0),
    redirect: v.optional(v.number(), 0),
  }),
  v.transform((w) => ({
    failed: w.failed,
    degraded: w.degraded,
    repeated: w.repeated,
    redirect: w.redirect,
  })),
);
export type RunMarkCounts = v.InferOutput<typeof runMarkCounts>;

/** How many steps of a Run are of each kind, the same six kinds a timeline step can be. */
const runKindCounts = v.pipe(
  v.object({
    thinking: v.optional(v.number(), 0),
    tool: v.optional(v.number(), 0),
    say: v.optional(v.number(), 0),
    answer: v.optional(v.number(), 0),
    instruction: v.optional(v.number(), 0),
    delivery: v.optional(v.number(), 0),
  }),
  v.transform((w) => ({
    thinking: w.thinking,
    tool: w.tool,
    say: w.say,
    answer: w.answer,
    instruction: w.instruction,
    delivery: w.delivery,
  })),
);

/** The three counter sets a Run result states about what it spent: its own agent, its children,
 *  and the two added up. A detail the record left out is empty, never invented. */
const runUsage = v.pipe(
  v.object({
    usage_details: v.optional(v.record(v.string(), v.number()), {}),
    child_usage_details: v.optional(v.record(v.string(), v.number()), {}),
    inclusive_usage_details: v.optional(v.record(v.string(), v.number()), {}),
  }),
  v.transform((w) => ({
    usageDetails: w.usage_details,
    childUsageDetails: w.child_usage_details,
    inclusiveUsageDetails: w.inclusive_usage_details,
  })),
);

/** One agent's share of the whole Run, as the overview's ledger row states it. */
const runAgentRollup = v.pipe(
  v.object({
    agent: v.string(),
    /** What the roster calls it: the Run's own agent is `main`, a child is its stored objective. */
    label: v.string(),
    order: v.number(),
    steps: v.number(),
    tool_calls: v.number(),
    marks: v.optional(runMarkCounts),
    tool_time_ms: v.optional(v.number(), 0),
    /** The earliest and latest instant the agent acts, ISO-8601 UTC; null for an agent with no steps. */
    first_at: v.nullable(v.string()),
    last_at: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    agent: w.agent,
    label: w.label,
    order: w.order,
    steps: w.steps,
    toolCalls: w.tool_calls,
    marks: w.marks ?? {failed: 0, degraded: 0, repeated: 0, redirect: 0},
    toolTimeMs: w.tool_time_ms,
    firstAt: w.first_at,
    lastAt: w.last_at,
  })),
);
export type RunAgentRollup = v.InferOutput<typeof runAgentRollup>;

/** One step the Run overview's attention list names: a marked step, or a delivery step. */
const runFinding = v.pipe(
  v.object({
    agent: v.string(),
    sequence: v.number(),
    kind: v.picklist(TIMELINE_KINDS),
    at: v.nullable(v.string()),
    /** The tool's own name and bounded subject; null for a step that states none. */
    name: v.nullable(v.string()),
    subject: v.nullable(v.string()),
    outcome: v.nullable(v.string()),
    marks: v.array(decisionMark),
    /** The first line of the step's masked content, bounded at the server edge. */
    excerpt: v.string(),
  }),
  v.transform((w) => ({
    agent: w.agent,
    sequence: w.sequence,
    kind: w.kind,
    at: w.at,
    name: w.name,
    subject: w.subject,
    outcome: w.outcome,
    marks: w.marks,
    excerpt: w.excerpt,
  })),
);
export type RunFinding = v.InferOutput<typeof runFinding>;

/** One Run's whole-Run facts: its totals, its agents in roster order, and its capped findings. Every
 *  number is the whole Run's — the panel never derives one from a page of steps. */
const runOverview = v.pipe(
  v.object({
    run_id: v.string(),
    totals: v.object({
      steps: v.number(),
      kinds: v.optional(runKindCounts),
      decisions: v.optional(v.number(), 0),
      marks: v.optional(runMarkCounts),
      tool_time_ms: v.optional(v.number(), 0),
    }),
    agents: v.array(runAgentRollup),
    findings: v.array(runFinding),
    /** How many findings the whole Run states; `findings` is capped below it. */
    findings_total: v.optional(v.number(), 0),
    /** The Run result's own usage record; null when the record states none. */
    usage: v.nullable(runUsage),
  }),
  v.transform((w) => ({
    runId: w.run_id,
    totals: {
      steps: w.totals.steps,
      kinds: w.totals.kinds ?? {thinking: 0, tool: 0, say: 0, answer: 0, instruction: 0, delivery: 0},
      decisions: w.totals.decisions,
      marks: w.totals.marks ?? {failed: 0, degraded: 0, repeated: 0, redirect: 0},
      toolTimeMs: w.totals.tool_time_ms,
    },
    agents: w.agents,
    findings: w.findings,
    findingsTotal: w.findings_total,
    usage: w.usage,
  })),
);
export type RunOverview = v.InferOutput<typeof runOverview>;

/** One step of a Run-wide step search: the message's actionable step, with its one-line excerpt and
 *  the whole Run's decisions about it. Thinking steps never appear. */
const runStep = v.pipe(
  v.object({
    agent: v.string(),
    agent_label: v.nullable(v.string()),
    at: v.nullable(v.string()),
    kind: v.picklist(TIMELINE_KINDS),
    /** The step's place in its own Session: what names the transcript row it came from. */
    sequence: v.number(),
    name: v.nullable(v.string()),
    subject: v.nullable(v.string()),
    outcome: v.nullable(v.string()),
    duration_ms: v.nullable(v.number()),
    excerpt: v.string(),
    /** The decisions the whole Run states about this step; the row never judges one itself. */
    marks: v.array(decisionMark),
  }),
  v.transform((w) => ({
    agent: w.agent,
    agentLabel: w.agent_label,
    at: w.at,
    sequence: w.sequence,
    kind: w.kind,
    name: w.name,
    subject: w.subject,
    outcome: w.outcome,
    durationMs: w.duration_ms,
    excerpt: w.excerpt,
    marks: w.marks,
  })),
);
export type RunStep = v.InferOutput<typeof runStep>;

/** One page of the Run's steps whose messages hold a search's words: a page of the Run's matches,
 *  oldest first within the page, and the cursor for the page before it. */
const stepSearchPage = v.pipe(
  v.object({
    run_id: v.string(),
    steps: v.array(runStep),
    next_before: v.nullable(v.string()),
    /** How many of the Run's searchable steps hold the words; a Run count, not the page's. */
    matched: v.number(),
    /** The words the search read, which a client marks. */
    terms: v.optional(v.array(v.string()), []),
    /** Whether the words as written answered, or the looser reading of them did. */
    match: v.optional(v.string(), 'all'),
  }),
  v.transform((w) => ({
    runId: w.run_id,
    steps: w.steps,
    /** The oldest instant on this page; ask again before it for the matches before these. */
    nextBefore: w.next_before,
    matched: w.matched,
    terms: w.terms,
    match: w.match,
  })),
);
export type StepSearchPage = v.InferOutput<typeof stepSearchPage>;

const childControlRecord = v.pipe(
  v.object({
    control_sequence: v.number(),
    kind: v.string(),
    content: v.string(),
    origin: v.string(),
    consumed: v.boolean(),
    consumed_at: v.optional(v.nullable(v.string())),
    created_at: v.optional(v.nullable(v.string())),
    operation_id: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    controlSequence: w.control_sequence,
    kind: w.kind,
    content: w.content,
    origin: w.origin,
    consumed: w.consumed,
    consumedAt: w.consumed_at ?? null,
    createdAt: w.created_at ?? null,
    operationId: w.operation_id ?? null,
  })),
);
export type ChildControlRecord = v.InferOutput<typeof childControlRecord>;

const childQuestion = v.pipe(
  v.object({
    request_id: v.string(),
    question: v.string(),
    status: v.string(),
    reply: v.optional(v.nullable(v.string())),
    reply_origin: v.optional(v.nullable(v.string())),
    expires_at: v.optional(v.nullable(v.string())),
    created_at: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({
    requestId: w.request_id,
    question: w.question,
    status: w.status,
    reply: w.reply ?? null,
    replyOrigin: w.reply_origin ?? null,
    expiresAt: w.expires_at ?? null,
    createdAt: w.created_at ?? null,
  })),
);
export type ChildQuestion = v.InferOutput<typeof childQuestion>;

const childResultLineage = v.pipe(
  v.nullable(v.object({
    status: v.optional(v.string()),
    summary: v.optional(v.string()),
    handles: v.optional(v.array(v.string())),
    operation_id: v.optional(v.nullable(v.string())),
  })),
  v.transform((w) => w === null ? null : {
    status: w.status ?? '',
    summary: w.summary ?? '',
    handles: w.handles ?? [],
    operationId: w.operation_id ?? null,
  }),
);

const childObservation = v.pipe(
  v.object({
    run_id: v.string(),
    child: agentChildStatus,
    controls: v.array(childControlRecord),
    questions: v.array(childQuestion),
    result: v.optional(childResultLineage),
  }),
  v.transform((w) => ({
    runId: w.run_id,
    child: w.child,
    controls: w.controls,
    questions: w.questions,
    result: w.result ?? null,
  })),
);
export type ChildObservation = v.InferOutput<typeof childObservation>;

const conversationTurn = v.pipe(
  v.object({
    turn_id: v.string(),
    turn_number: v.number(),
    answer_run_id: v.string(),
    submission_id: v.string(),
    status: v.picklist(['queued', 'running', 'succeeded', 'failed', 'cancelled']),
    cancel_requested: v.boolean(),
    user_text: v.string(),
    assistant_text: v.string(),
    user_attachments: v.array(attachmentReference),
    steering_messages: v.array(v.string()),
    presentation: v.nullable(answerPresentation),
    usage: v.record(v.string(), v.unknown()),
    error_kind: v.nullable(v.string()),
    error_message: v.nullable(v.string()),
    child_count: v.optional(v.number(), 0),
    answer_sequence: v.optional(v.nullable(v.number())),
    feedback: v.optional(v.nullable(answerJudgement)),
    created_at: v.string(),
  }),
  v.transform((w) => ({
    turnId: w.turn_id,
    turnNumber: w.turn_number,
    answerRunId: w.answer_run_id,
    submissionId: w.submission_id,
    status: w.status,
    cancelRequested: w.cancel_requested,
    userText: w.user_text,
    assistantText: w.assistant_text,
    userAttachments: w.user_attachments,
    steeringMessages: w.steering_messages,
    presentation: w.presentation,
    usage: w.usage,
    errorKind: w.error_kind,
    errorMessage: w.error_message,
    /** How many Child Sessions the turn's Run started. */
    childCount: w.child_count,
    /** The `sequence` of the Run's final answer message in its main Session, which a judgement names; null
     *  while the Run has not answered. */
    answerSequence: w.answer_sequence ?? null,
    /** The reader's judgement of that message, null when they have given none. */
    feedback: w.feedback ?? null,
    createdAt: w.created_at,
  })),
);
export type ConversationTurn = v.InferOutput<typeof conversationTurn>;

/** The structured result a command carries, when the server wrote one this client can render. Any
 *  payload that is not an object reads as "no structured result": the canonical English `result` is
 *  then the fallback the command node shows. */
function readCommandResultData(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/** One command event the server executed and kept for a conversation: the reader's slash line, its
 *  name and words, and the result. Its `sequence` shares one ordering space with a turn's
 *  `turn_number`, which is what lets a reader interleave the two in the order they happened. */
const conversationCommand = v.pipe(
  v.object({
    command_id: v.string(),
    sequence: v.number(),
    name: v.string(),
    input: v.string(),
    /** The server's canonical English result, shown only when `result_data` is absent or unreadable. */
    result: v.string(),
    /** The structured result the client renders in the reader's language, or null for a command that
     *  states only its English result. */
    result_data: v.nullable(v.record(v.string(), v.unknown())),
    tone: v.picklist(['success', 'error']),
    created_at: v.string(),
  }),
  v.transform((w) => ({
    commandId: w.command_id,
    sequence: w.sequence,
    name: w.name,
    input: w.input,
    result: w.result,
    resultData: readCommandResultData(w.result_data),
    tone: w.tone,
    createdAt: w.created_at,
  })),
);
export type ConversationCommand = v.InferOutput<typeof conversationCommand>;

export const acceptedAnswer = v.object({
  kind: v.literal('accepted'),
  conversation: conversationSummary,
  turn: conversationTurn,
});
export type AcceptedAnswer = v.InferOutput<typeof acceptedAnswer>;

/** One message waiting behind the conversation's live Run, in the order it will be delivered. */
const queuedMessage = v.pipe(
  v.object({
    message_id: v.string(),
    content: v.string(),
    position: v.number(),
    created_at: v.string(),
  }),
  v.transform((w) => ({
    messageId: w.message_id,
    content: w.content,
    position: w.position,
    createdAt: w.created_at,
  })),
);
export type QueuedMessage = v.InferOutput<typeof queuedMessage>;

const conversationQueue = v.pipe(
  v.object({items: v.array(queuedMessage)}),
  v.transform((w) => ({items: w.items})),
);

/** A send the server delivered to the live Run instead of starting a turn. `queue` names the held
 *  message; `steer` names the Run it was injected into and the control record that says so. */
const queuedDelivery = v.pipe(
  v.object({
    kind: v.literal('queue'),
    message: queuedMessage,
  }),
  v.transform((w) => ({kind: 'queue' as const, message: w.message})),
);
const steeredDelivery = v.pipe(
  v.object({
    kind: v.literal('steer'),
    run_id: v.string(),
    control_sequence: v.number(),
  }),
  v.transform((w) => ({
    kind: 'steer' as const,
    runId: w.run_id,
    controlSequence: w.control_sequence,
  })),
);

/** What one send produced: an accepted turn, or delivery to the Run already live. */
export const answerSubmissionResponse = v.variant('kind', [
  acceptedAnswer,
  queuedDelivery,
  steeredDelivery,
]);
export type AnswerSubmissionResponse = v.InferOutput<typeof answerSubmissionResponse>;

const conversationPage = v.pipe(
  v.object({
    items: v.array(conversationSummary),
    next_cursor: v.optional(v.nullable(v.string())),
  }),
  v.transform((w) => ({items: w.items, nextCursor: w.next_cursor ?? null})),
);
export type ConversationPage = v.InferOutput<typeof conversationPage>;

const conversationHistory = v.pipe(
  v.object({
    conversation: conversationSummary,
    turns: v.array(conversationTurn),
    commands: v.array(conversationCommand),
    next_cursor: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    conversation: w.conversation,
    turns: w.turns,
    commands: w.commands,
    nextCursor: w.next_cursor,
  })),
);
export type ConversationHistory = v.InferOutput<typeof conversationHistory>;

const agentChildRosterPage = v.pipe(
  v.object({
    children: v.array(agentChildStatus),
    next_cursor: v.nullable(v.string()),
    // The Run's own row, stated the way each row below is: where it stands and the
    // clock its agent row draws.
    run_status: v.string(),
    run_started_at: v.nullable(v.string()),
    run_finished_at: v.nullable(v.string()),
  }),
  v.transform((w) => ({
    children: w.children,
    nextCursor: w.next_cursor,
    run: {
      status: w.run_status,
      startedAt: w.run_started_at,
      finishedAt: w.run_finished_at,
    },
  })),
);
export type AgentChildRosterPage = v.InferOutput<typeof agentChildRosterPage>;

/** The Run's own row as the roster page states it: the same facts a child's row
 *  carries, so the main agent is drawn from one source and never invented. */
export interface AgentRunStatus {
  readonly status: string;
  readonly startedAt: string | null;
  readonly finishedAt: string | null;
}

/** Create one empty conversation. The composer uses it when a command has no conversation yet. */
export async function createConversation(signal?: AbortSignal): Promise<ConversationSummary> {
  const response = await fetch('/web/api/conversations', {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: '{}',
    signal,
  });
  return parseWire(response, conversationSummary);
}

export async function listConversations(
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<ConversationPage> {
  const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`;
  const response = await fetch(`/web/api/conversations${query}`, {signal});
  return parseWire(response, conversationPage);
}

export async function getConversationHistory(
  conversationId: string,
  cursor: string | null = null,
  limit?: number,
  signal?: AbortSignal,
): Promise<ConversationHistory> {
  const id = encodeURIComponent(conversationId);
  const query = new URLSearchParams();
  if (cursor !== null) query.set('cursor', cursor);
  if (limit !== undefined) query.set('limit', String(limit));
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`/web/api/conversations/${id}/history${suffix}`, {signal});
  return parseWire(response, conversationHistory);
}

/** The command the server executed and kept, as the command route states it. */
const acceptedConversationCommand = v.pipe(
  v.object({command: conversationCommand}),
  v.transform((w) => w.command),
);

/** Run one Web command in this conversation on the server, which owns the execution and keeps the
 *  command's result in the conversation's history. The returned command is the persisted one. */
export async function postConversationCommand(
  conversationId: string,
  command: {name: string; input: string; workspaces?: readonly string[]},
  signal?: AbortSignal,
): Promise<ConversationCommand> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}/commands`, {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({
      name: command.name,
      input: command.input,
      ...(command.workspaces === undefined ? {} : {workspaces: [...command.workspaces]}),
    }),
    signal,
  });
  return parseWire(response, acceptedConversationCommand);
}

export async function renameConversation(
  conversationId: string,
  title: string,
  signal?: AbortSignal,
): Promise<ConversationSummary> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}`, {
    method: 'PATCH',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({title}),
    signal,
  });
  return parseWire(response, conversationSummary);
}

export async function deleteConversation(
  conversationId: string,
  signal?: AbortSignal,
): Promise<void> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

export async function deleteAllConversations(signal?: AbortSignal): Promise<void> {
  const response = await fetch('/web/api/conversations', {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

/** The route behind a server-provided artifact URL, which must be this page's own origin. */
function artifactRoute(url: string): string {
  const target = new URL(url, window.location.origin);
  if (target.origin !== window.location.origin) {
    throw new Error('Untrusted artifact URL');
  }
  return target.pathname + target.search;
}

/** Parse the presentation served at a server-provided artifact URL. The URL
 *  stays server-owned; the response still crosses the validated edge. */
export async function getArtifactPresentationAt(
  presentationUrl: string,
  signal?: AbortSignal,
): Promise<AnswerPresentation> {
  const response = await fetch(artifactRoute(presentationUrl), {signal});
  return parseWire(response, answerPresentation);
}

/** The text of the artifact served at a server-provided URL: its first `firstBytes` bytes when a limit is
 *  given, which is how a text preview avoids reading a file of any size. A refusal is an `ApiError`. */
export async function getArtifactTextAt(
  dataUrl: string,
  options: {firstBytes?: number} = {},
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(artifactRoute(dataUrl), {
    headers: options.firstBytes === undefined
      ? undefined
      : {Range: `bytes=0-${options.firstBytes - 1}`},
    signal,
  });
  if (!response.ok) throw await apiError(response);
  return await response.text();
}

export async function getRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/runs/${id}`, {signal});
  return parseWire(response, conversationTurn);
}

const agentspaceHead = v.pipe(
  v.object({
    lane_id: v.string(),
    head_run_id: v.string(),
    manifest_digest: v.string(),
    revision: v.number(),
    committed_at: v.string(),
  }),
  v.transform((w) => ({
    laneId: w.lane_id,
    headRunId: w.head_run_id,
    manifestDigest: w.manifest_digest,
    revision: w.revision,
    committedAt: w.committed_at,
  })),
);

/** Which way one saved copy moved a path: it appeared, its bytes changed, or it came through. */
const agentspaceChange = v.picklist(['added', 'changed', 'unchanged']);
export type AgentspaceChange = v.InferOutput<typeof agentspaceChange>;

/** What one saved copy changed against the copy it was built on. A copy whose parent is no longer
 *  retained is read as built on nothing, so every file in it counts as added and none as removed. */
const agentspaceChangeCounts = v.pipe(
  v.object({
    added: v.number(),
    changed: v.number(),
    removed: v.number(),
  }),
  v.transform((w) => ({added: w.added, changed: w.changed, removed: w.removed})),
);
export type AgentspaceChangeCounts = v.InferOutput<typeof agentspaceChangeCounts>;

const agentspaceSnapshot = v.pipe(
  v.object({
    run_id: v.string(),
    parent_run_id: v.nullable(v.string()),
    manifest_digest: v.string(),
    file_count: v.number(),
    created_at: v.string(),
    changes: agentspaceChangeCounts,
  }),
  v.transform((w) => ({
    runId: w.run_id,
    parentRunId: w.parent_run_id,
    manifestDigest: w.manifest_digest,
    fileCount: w.file_count,
    createdAt: w.created_at,
    changes: w.changes,
  })),
);
export type AgentspaceSnapshot = v.InferOutput<typeof agentspaceSnapshot>;

const agentspaceFile = v.pipe(
  v.object({
    relative_path: v.string(),
    content_digest: v.string(),
    size_bytes: v.number(),
    change: agentspaceChange,
  }),
  v.transform((w) => ({
    relativePath: w.relative_path,
    contentDigest: w.content_digest,
    sizeBytes: w.size_bytes,
    change: w.change,
  })),
);
export type AgentspaceFile = v.InferOutput<typeof agentspaceFile>;

const agentspaceFileVersion = v.pipe(
  v.object({
    run_id: v.string(),
    content_digest: v.string(),
    size_bytes: v.number(),
    created_at: v.string(),
    change: agentspaceChange,
  }),
  v.transform((w) => ({
    runId: w.run_id,
    contentDigest: w.content_digest,
    sizeBytes: w.size_bytes,
    createdAt: w.created_at,
    change: w.change,
  })),
);
export type AgentspaceFileVersion = v.InferOutput<typeof agentspaceFileVersion>;

const conversationAgentspace = v.pipe(
  v.object({heads: v.array(agentspaceHead), snapshots: v.array(agentspaceSnapshot)}),
  v.transform((w) => ({heads: w.heads, snapshots: w.snapshots})),
);
export type ConversationAgentspace = v.InferOutput<typeof conversationAgentspace>;

const agentspaceSnapshotDetail = v.pipe(
  v.object({snapshot: agentspaceSnapshot, files: v.array(agentspaceFile)}),
  v.transform((w) => ({snapshot: w.snapshot, files: w.files})),
);
export type AgentspaceSnapshotDetail = v.InferOutput<typeof agentspaceSnapshotDetail>;

const agentspaceFileHistory = v.pipe(
  v.object({
    relative_path: v.string(),
    versions: v.array(agentspaceFileVersion),
  }),
  v.transform((w) => ({relativePath: w.relative_path, versions: w.versions})),
);
export type AgentspaceFileHistory = v.InferOutput<typeof agentspaceFileHistory>;

/** The conversation's branch heads and restorable snapshots, for the Inspector. */
export async function readConversationAgentspace(
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationAgentspace> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}/workspace`, {signal});
  return parseWire(response, conversationAgentspace);
}

/** One snapshot's files. */
export async function readAgentspaceSnapshot(
  conversationId: string,
  runId: string,
  signal?: AbortSignal,
): Promise<AgentspaceSnapshotDetail> {
  const id = encodeURIComponent(conversationId);
  const run = encodeURIComponent(runId);
  const response = await fetch(
    `/web/api/conversations/${id}/workspace/snapshots/${run}`,
    {signal},
  );
  return parseWire(response, agentspaceSnapshotDetail);
}

/** Which settled Run changed one Agentspace path, oldest first. */
export async function readAgentspaceFileHistory(
  conversationId: string,
  relativePath: string,
  signal?: AbortSignal,
): Promise<AgentspaceFileHistory> {
  const id = encodeURIComponent(conversationId);
  const params = new URLSearchParams({path: relativePath});
  const response = await fetch(
    `/web/api/conversations/${id}/workspace/files?${params}`,
    {signal},
  );
  return parseWire(response, agentspaceFileHistory);
}

/** Forget the conversation's Agentspace branches and snapshots; refused while a Run is live. */
export async function resetConversationAgentspace(
  conversationId: string,
  signal?: AbortSignal,
): Promise<void> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}/workspace/reset`, {
    method: 'POST',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

/** The messages this reader queued behind the conversation's live Run, oldest first. */
export async function readConversationQueue(
  conversationId: string,
  signal?: AbortSignal,
): Promise<readonly QueuedMessage[]> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}/queue`, {signal});
  return (await parseWire(response, conversationQueue)).items;
}

/** Rewrite one held message; the words are what will be delivered. */
export async function editQueuedMessage(
  conversationId: string,
  messageId: string,
  content: string,
  signal?: AbortSignal,
): Promise<QueuedMessage> {
  const id = encodeURIComponent(conversationId);
  const message = encodeURIComponent(messageId);
  const response = await fetch(`/web/api/conversations/${id}/queue/${message}`, {
    method: 'PATCH',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({content}),
    signal,
  });
  const rewritten = await parseWire(response, v.object({message: queuedMessage}));
  return rewritten.message;
}

/** Drop one held message before it is delivered. */
export async function removeQueuedMessage(
  conversationId: string,
  messageId: string,
  signal?: AbortSignal,
): Promise<void> {
  const id = encodeURIComponent(conversationId);
  const message = encodeURIComponent(messageId);
  const response = await fetch(`/web/api/conversations/${id}/queue/${message}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  if (!response.ok) throw await apiError(response);
}

/** Rewrite the whole order; every held message exactly once. */
export async function reorderConversationQueue(
  conversationId: string,
  messageIds: readonly string[],
  signal?: AbortSignal,
): Promise<readonly QueuedMessage[]> {
  const id = encodeURIComponent(conversationId);
  const response = await fetch(`/web/api/conversations/${id}/queue/order`, {
    method: 'PUT',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({message_ids: [...messageIds]}),
    signal,
  });
  return (await parseWire(response, conversationQueue)).items;
}

export async function getAnswerRunChildrenPage(
  runId: string,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<AgentChildRosterPage> {
  const id = encodeURIComponent(runId);
  const query = new URLSearchParams();
  if (cursor !== null) query.set('cursor', cursor);
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`/web/api/answer/${id}/children${suffix}`, {signal});
  return parseWire(response, agentChildRosterPage);
}

export async function getAnswerRunChild(
  runId: string,
  childSessionId: string,
  signal?: AbortSignal,
): Promise<ChildObservation> {
  const run = encodeURIComponent(runId);
  const child = encodeURIComponent(childSessionId);
  const response = await fetch(`/web/api/answer/${run}/children/${child}`, {signal});
  return parseWire(response, childObservation);
}

/** One Run's whole-Run overview: its totals, its agents in roster order, and its capped findings.
 *  The server revalidates it by the derived index's version, so a follow read is cheap; every number
 *  is the whole Run's, never a page's. */
export async function getAnswerRunOverview(
  runId: string,
  signal?: AbortSignal,
): Promise<RunOverview> {
  const run = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${run}/overview`, {signal});
  return parseWire(response, runOverview);
}

/** One page of the Run's steps whose messages hold `text`, newest page first; `before` is the cursor
 *  the previous page stated. `agent` names the step's own agent, and thinking steps never appear. */
export async function getAnswerStepSearch(
  runId: string,
  text: string,
  before: string | null = null,
  signal?: AbortSignal,
): Promise<StepSearchPage> {
  const run = encodeURIComponent(runId);
  const query = new URLSearchParams({q: text});
  if (before !== null) query.set('before', before);
  const response = await fetch(`/web/api/answer/${run}/steps/search?${query.toString()}`, {signal});
  return parseWire(response, stepSearchPage);
}

/** One page of a Run's main agent transcript, or of one of its children's: the newest page without a
 *  cursor, then each older one by the cursor its predecessor gave. */
export async function getAnswerActivityPage(
  runId: string,
  childSessionId: string | null,
  cursor: string | null = null,
  signal?: AbortSignal,
): Promise<ActivityPage> {
  const query = new URLSearchParams();
  if (childSessionId !== null) query.set('child', childSessionId);
  if (cursor !== null) query.set('before', cursor);
  return readActivityPage(runId, query, signal);
}

/** The messages this agent wrote after a sequence, oldest first: what a reader holding a transcript
 *  asks for to append the ones that landed since, instead of reading the newest page again. */
export async function getAnswerActivitySince(
  runId: string,
  childSessionId: string | null,
  sequence: number,
  signal?: AbortSignal,
): Promise<ActivityPage> {
  const query = new URLSearchParams();
  if (childSessionId !== null) query.set('child', childSessionId);
  query.set('since', String(sequence));
  return readActivityPage(runId, query, signal);
}

async function readActivityPage(
  runId: string,
  query: URLSearchParams,
  signal?: AbortSignal,
): Promise<ActivityPage> {
  const suffix = query.size > 0 ? `?${query.toString()}` : '';
  const response = await fetch(`/web/api/answer/${encodeURIComponent(runId)}/transcript${suffix}`, {signal});
  const page = await parseWire(response, activityPage);
  return {...page, messages: readAgents(page.messages)};
}

/** The messages of one agent's transcript that hold the reader's words, oldest first. */
const traceSearch = v.pipe(
  v.object({
    messages: v.array(activityMessage),
    terms: v.optional(v.array(v.string())),
    total: v.optional(v.number()),
    match: v.optional(v.string()),
  }),
  v.transform((w) => ({messages: w.messages, terms: w.terms ?? [], total: w.total ?? w.messages.length, match: w.match ?? 'all'})),
);
export type TraceSearch = Omit<v.InferOutput<typeof traceSearch>, 'messages'> & {
  messages: ActivityMessage[];
};

/** Search one agent's transcript: the Run's main agent when `childSessionId` is null. */
export async function getAnswerTraceSearch(
  runId: string,
  text: string,
  signal?: AbortSignal,
): Promise<TraceSearch> {
  const query = new URLSearchParams({q: text});
  const response = await fetch(
    `/web/api/answer/${encodeURIComponent(runId)}/transcript/search?${query.toString()}`,
    {signal},
  );
  const answer = await parseWire(response, traceSearch);
  return {...answer, messages: readAgents(answer.messages)};
}

export async function forkAnswerRun(
  runId: string,
  content: string,
  submissionId: string,
  signal?: AbortSignal,
): Promise<AcceptedAnswer> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/answer/${id}/fork`, {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({content, submission_id: submissionId}),
    signal,
  });
  if (!response.ok) throw await webCommandError(response);
  return parseWire(response, acceptedAnswer);
}

/** Ask the server to stop a run; disconnecting never does this on its own. */
export async function cancelRun(
  runId: string,
  signal?: AbortSignal,
): Promise<ConversationTurn> {
  const id = encodeURIComponent(runId);
  const response = await fetch(`/web/api/runs/${id}`, {
    method: 'DELETE',
    headers: csrfHeaders(),
    signal,
  });
  return parseWire(response, conversationTurn);
}

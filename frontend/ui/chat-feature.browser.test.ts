// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The chat feature in the page: the history it shows, the Runs it follows, and what it hands the rest of the app. */

import {expect} from '@esm-bundle/chai';
import type {AcceptedAnswer, AnswerArtifact, AnswerPresentation, ConversationAttachmentReference, ConversationTurn, PresentationImage} from '../api/conversations.ts';
import type {ChatTurnView, ProcessEntry} from '../lib/chat-views.ts';
import type {MemoryOperationEvent} from '../api/memory.ts';
import {ANSWER_PHASE_LABELS, answerPhaseLabel} from '../lib/turn-projection.ts';
import {applyToolEvent, toolStatusText} from '../lib/tool-events.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import {toolDisplay} from '../lib/tool-display.ts';
import {attachmentStore, continuationDescriptor, conversationStore, conversationWire, originalRevokeObjectURL, policy, presentation, restoreChatFeatureFixtures, settle, storedTurn, submissions} from '../testing/chat-feature.ts';
import {chooseFiles, waitFor} from '../testing/dom.ts';
import {followConversationView} from '../testing/conversation-view.ts';
import './chat-composer.ts';
import './chat-feature.ts';
import type {ChatRunActivityDetail, DlChatFeature} from './chat-feature.ts';
import type {DlAnswerFeedback} from './answer-feedback.ts';
import type {DlChatComposer} from './chat-composer.ts';
import {ANSWER_RECONNECT_COPY, answerReconnectState, type ChatRunActionDetail, type DlChatMessageList, storedTurnView} from './chat-message-list.ts';

restoreChatFeatureFixtures();

function attachmentWire(attachment: ConversationAttachmentReference): Record<string, unknown> {
  return {
    attachment_id: attachment.attachmentId,
    ordinal: attachment.ordinal,
    kind: attachment.kind,
    filename: attachment.filename,
    mime_type: attachment.mimeType,
    byte_size: attachment.byteSize,
    url: attachment.url,
    thumbnail_url: attachment.thumbnailUrl,
    label: attachment.label,
  };
}

function presentationImageWire(image: PresentationImage): Record<string, unknown> {
  return {
    id: image.id,
    chunk_id: image.chunkId,
    source_ref: image.sourceRef,
    url: image.url,
    thumbnail_url: image.thumbnailUrl,
    label: image.label,
    answer_image_sent: image.answerImageSent,
  };
}

function artifactWire(artifact: AnswerArtifact): Record<string, unknown> {
  return {
    resource_id: artifact.resourceId,
    media_type: artifact.mediaType,
    label: artifact.label,
    filename: artifact.filename,
    byte_size: artifact.byteSize,
    digest: artifact.digest,
    presentation: artifact.presentation,
    status: artifact.status,
    uri: artifact.uri,
    width: artifact.width,
    height: artifact.height,
    data_url: artifact.dataUrl,
    download_url: artifact.downloadUrl,
    presentation_url: artifact.presentationUrl,
    issue: artifact.issue === null ? null : {
      kind: artifact.issue.kind,
      description: artifact.issue.description,
      resource_id: artifact.issue.resourceId,
    },
  };
}

function presentationWire(presentation: AnswerPresentation): Record<string, unknown> {
  return {
    answer_text: presentation.answerText,
    parts: presentation.parts.map((part) => ({
      type: part.type,
      text: part.text,
      html: part.html,
      artifact: part.artifact === null ? null : artifactWire(part.artifact),
      evidence_image: part.evidenceImage === null ? null : presentationImageWire(part.evidenceImage),
      inline: part.inline,
    })),
    sources: presentation.sources.map((source) => ({
      id: source.id,
      title: source.title,
      source_url: source.sourceUrl,
      download_url: source.downloadUrl,
      chunks: source.chunks.map((chunk) => ({
        chunk_idx: chunk.chunkIdx,
        page_number: chunk.pageNumber,
        content_html: chunk.contentHtml,
        image_url: chunk.imageUrl,
        thumbnail_url: chunk.thumbnailUrl,
      })),
    })),
    evidence_images: presentation.evidenceImages.map(presentationImageWire),
    artifacts: presentation.artifacts.map(artifactWire),
    artifact_outcome: {
      status: presentation.artifactOutcome.status,
      issues: presentation.artifactOutcome.issues.map((issue) => ({
        kind: issue.kind,
        description: issue.description,
        resource_id: issue.resourceId,
      })),
    },
  };
}

function turnWire(turn: ConversationTurn): Record<string, unknown> {
  return {
    turn_id: turn.turnId,
    turn_number: turn.turnNumber,
    answer_run_id: turn.answerRunId,
    submission_id: turn.submissionId,
    status: turn.status,
    cancel_requested: turn.cancelRequested,
    user_text: turn.userText,
    assistant_text: turn.assistantText,
    user_attachments: turn.userAttachments.map(attachmentWire),
    steering_messages: turn.steeringMessages,
    presentation: turn.presentation === null ? null : presentationWire(turn.presentation),
    usage: turn.usage,
    error_kind: turn.errorKind,
    error_message: turn.errorMessage,
    child_count: turn.childCount,
    created_at: turn.createdAt,
  };
}

function acceptedWire(accepted: AcceptedAnswer): Record<string, unknown> {
  return {
    kind: 'accepted',
    conversation: conversationWire(accepted.conversation),
    turn: turnWire(accepted.turn),
  };
}

it('composes stored history through public properties and AnswerPresentation properties', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.attachmentPolicy = policy;
  feature.attachmentAccept = 'image/*,.md,.pdf';
  feature.view = {
    kind: 'ready',
    conversationId: 'conversation-1',
    history: [storedTurn()],
    lineage: 'Original thread',
  };
  document.body.appendChild(feature);
  await settle(feature);

  const answer = feature.querySelector('dl-answer-presentation');
  expect(answer?.presentation).to.equal(presentation);
  expect(answer?.textContent).to.contain('A stored answer.');
  expect(feature.querySelector('[role="log"]')?.getAttribute('aria-label')).to.equal(
    'Conversation messages',
  );
  expect(feature.textContent).to.contain('Forked from Original thread');

  let action: ChatRunActionDetail | null = null;
  feature.addEventListener('dl-chat-run-action', (event) => {
    action = (event as CustomEvent<ChatRunActionDetail>).detail;
  });
  feature.querySelector<HTMLButtonElement>('button')?.focus();
  const fork = Array.from(feature.querySelectorAll('button')).find(
    (button) => button.textContent?.trim() === 'Fork',
  );
  fork?.click();
  expect(action).to.deep.equal({action: 'fork', runId: 'run-1'});
});

it('offers Fork and Agent traces on every settled turn, and no per-turn Follow-Up control', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'fork-per-turn',
    lineage: null,
    history: [
      storedTurn(),
      {
        ...storedTurn(),
        turnId: 'turn-2',
        turnNumber: 2,
        answerRunId: 'run-2',
        submissionId: 'submission-2',
        childCount: 2,
      },
    ],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const labels = (turnId: string): (string | undefined)[] => Array.from(
    feature.querySelectorAll<HTMLButtonElement>(`[data-turn-id="${turnId}"] button`),
  ).map((button) => button.textContent?.trim());

  // Fork branches from the turn it names, so every settled turn offers it. A
  // Follow-Up appends to the Lane tip, which the composer already owns, so the
  // Web offers no per-turn control for it.
  // Agent traces holds the main agent beside any children, so every turn with a Run has it, first.
  expect(labels('turn-1')).to.deep.equal(['Agent traces', 'Fork']);
  expect(labels('turn-2')).to.deep.equal(['Agent traces', 'Fork']);

  const actions: ChatRunActionDetail[] = [];
  feature.addEventListener('dl-chat-run-action', (event) => {
    actions.push((event as CustomEvent<ChatRunActionDetail>).detail);
  });
  const turn2 = (name: string) => Array.from(
    feature.querySelectorAll<HTMLButtonElement>('[data-turn-id="turn-2"] button'),
  ).find((button) => button.textContent?.trim() === name);
  turn2('Fork')?.click();
  turn2('Agent traces')?.click();
  expect(actions).to.deep.equal([
    {action: 'fork', runId: 'run-2'},
    {action: 'traces', runId: 'run-2'},
  ]);
  // A settled Run's way in is words alone: the live dot is the only glyph this
  // control ever draws, and only while the Run is still going.
  expect(turn2('Agent traces')?.querySelectorAll('svg')).to.have.length(0);
});

it('offers the thumbs after Fork on a settled answer the server has named, when the reader is offered feedback', async () => {
  const judged = {value: 'down' as const, reasons: ['off_topic' as const], note: null};
  const turn = (number: number, extra: Partial<ConversationTurn> = {}): ConversationTurn => ({
    ...storedTurn(),
    turnId: `turn-${number}`,
    turnNumber: number,
    answerRunId: `run-${number}`,
    submissionId: `submission-${number}`,
    ...extra,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.answerFeedbackEnabled = true;
  feature.view = {
    kind: 'ready',
    conversationId: 'feedback-thumbs',
    lineage: null,
    history: [
      turn(1, {answerSequence: 7, feedback: judged, usage: {usage_details: {total_tokens: 1234}}}),
      // The Run has not been read since it finished: the server has not said which message is the answer.
      turn(2),
      turn(3, {status: 'failed', presentation: null, errorKind: 'internal', errorMessage: 'It failed.'}),
    ],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const labels = (turnId: string): (string | undefined)[] => Array.from(
    feature.querySelectorAll<HTMLButtonElement>(`[data-turn-id="${turnId}"] button`),
  ).map((button) => button.getAttribute('aria-label') ?? button.textContent?.trim());
  // Fork branches the Run, the thumbs judge its answer, and the tokens say what it spent.
  expect(labels('turn-1')).to.deep.equal(['Agent traces', 'Fork', 'Good answer', 'Bad answer']);
  expect(labels('turn-2')).to.deep.equal(['Agent traces', 'Fork']);
  expect(labels('turn-3')).to.deep.equal(['Retry this message', 'Agent traces', 'Fork']);
  const element = feature.querySelector<DlAnswerFeedback>('[data-turn-id="turn-1"] dl-answer-feedback')!;
  expect([element.runId, element.agent, element.sequence, element.layout, element.judgement])
    .to.deep.equal(['run-1', 'main', 7, 'row', judged]);
  expect(element.querySelector('[data-thumb="down"]')?.getAttribute('aria-pressed')).to.equal('true');
  const usage = feature.querySelector('[data-turn-id="turn-1"] [class*="runUsage"]')!;
  expect(element.compareDocumentPosition(usage) & Node.DOCUMENT_POSITION_FOLLOWING).to.be.greaterThan(0);

  // A reader the operator has switched feedback off for is offered no thumbs, however the page drew the turn.
  feature.answerFeedbackEnabled = false;
  await settle(feature);
  expect(labels('turn-1')).to.deep.equal(['Agent traces', 'Fork']);
});

it('redraws a settled turn\'s Run actions and token total in the language the reader switches to', async () => {
  const turn: ConversationTurn = {
    ...storedTurn(),
    presentation: {
      ...presentation,
      sources: [{id: '1', title: 'Source', sourceUrl: null, downloadUrl: null, chunks: []}],
    },
    usage: {usage_details: {total_tokens: 1234}},
    answerSequence: 7,
  };
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.answerFeedbackEnabled = true;
  feature.view = {
    kind: 'ready',
    conversationId: 'locale-turn-actions',
    lineage: null,
    history: [turn],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const turnText = (): string => feature.querySelector('[data-turn-id="turn-1"]')?.textContent ?? '';
  const usage = (): string => feature
    .querySelector('[data-turn-id="turn-1"] .runUsage')?.textContent?.trim() ?? '';
  const thumbs = (): (string | null)[] => Array.from(
    feature.querySelectorAll('[data-turn-id="turn-1"] [data-thumb]'),
  ).map((button) => button.getAttribute('aria-label'));
  const reasons = (): (string | undefined)[] => Array.from(
    feature.querySelectorAll('[data-turn-id="turn-1"] [data-reason]'),
  ).map((button) => button.textContent?.trim());

  expect(turnText()).to.contain('Agent traces');
  expect(turnText()).to.contain('Fork');
  expect(usage()).to.equal('1.2K tokens');
  expect(thumbs()).to.deep.equal(['Good answer', 'Bad answer']);
  // The reasons are open when the language changes: they are drawn by the element and not by the cached turn.
  feature.querySelector<HTMLButtonElement>('[data-turn-id="turn-1"] [data-thumb="down"]')!.click();
  await settle(feature);
  await feature.querySelector('dl-answer-feedback')?.updateComplete;
  expect(reasons()).to.deep.equal(
    ['Unsupported claims', 'Factual error', 'Too verbose', 'Off topic', 'Irrelevant citations'],
  );
  try {
    await setLanguagePreference('zh');
    await feature.updateComplete;
    await feature.querySelector('dl-chat-message-list')?.updateComplete;
    await feature.querySelector('dl-answer-feedback')?.updateComplete;
    expect(turnText()).to.contain('代理轨迹');
    expect(turnText()).to.contain('分叉');
    expect(usage()).to.equal('1234 个 token');
    expect(thumbs()).to.deep.equal(['回答不错', '回答不佳']);
    expect(reasons()).to.deep.equal(['没有依据', '事实错误', '太啰嗦', '答非所问', '引用无关']);
  } finally {
    await setLanguagePreference('en');
  }
});

it('redraws a running turn\'s status line in the language the reader switches to', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {kind: 'ready', conversationId: 'locale-activity', history: [], lineage: null};
  list.turns = [{
    id: 'turn-activity',
    userText: 'Question',
    userAttachments: [],
    runId: 'run-activity',
    state: 'streaming',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: {kind: 'phase', phase: 'searching'},
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  }];
  document.body.appendChild(list);
  await list.updateComplete;

  const phase = (): string => list.querySelector('[data-phase]')?.getAttribute('data-phase') ?? '';
  const announced = (): string => list.querySelector(
    '[data-turn-id="turn-activity"] .dl-sr-only[role="status"]',
  )?.textContent ?? '';
  expect(phase()).to.equal('Searching knowledge base...');
  try {
    await setLanguagePreference('zh');
    await list.updateComplete;
    // The view holds the phase id and the component applies the words, so both the
    // drawn line and its announcement follow the switch without a new event.
    expect(phase()).to.equal('正在搜索知识库…');
    expect(announced()).to.equal('正在搜索知识库…');
  } finally {
    await setLanguagePreference('en');
  }
});

it('keeps the words of a streaming answer out of the live log, and lets the status say how the turn ended', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  list.view = {kind: 'ready', conversationId: 'quiet-stream', history: [], lineage: null};
  const streaming: ChatTurnView = {
    id: 'turn-quiet',
    userText: 'Question',
    userAttachments: [],
    runId: 'run-quiet',
    state: 'streaming',
    streamText: 'The answer so far',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: {kind: 'phase', phase: 'generating'},
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  };
  list.turns = [streaming];
  document.body.appendChild(list);
  await list.updateComplete;

  // The messages are a log, which a screen reader reads out as it grows.
  const log = list.querySelector('[role="log"]')!;
  const status = (): Element | null => list.querySelector('[data-turn-id="turn-quiet"] .dl-sr-only[role="status"]');
  // A busy log could hold back the status that says the turn is over, so it is never marked busy.
  expect(log.hasAttribute('aria-busy')).to.equal(false);
  // Words that grow on every frame, and the dots drawn beside them, are not read out as they change.
  expect(list.querySelector('.stream-tail')?.textContent).to.equal('The answer so far');
  expect(list.querySelector('.stream-tail')?.getAttribute('aria-live')).to.equal('off');
  expect(list.querySelector('[data-phase]')?.getAttribute('aria-live')).to.equal('off');
  expect(list.querySelector('.stream-tail')?.closest('[role="log"]') === log, 'they sit in the log').to.equal(true);

  list.turns = [{...streaming, streamText: 'The answer so far, and a little more'}];
  await list.updateComplete;
  expect(list.querySelector('.stream-tail')?.textContent).to.equal('The answer so far, and a little more');
  expect(list.querySelector('.stream-tail')?.getAttribute('aria-live'), 'still quiet as it grows').to.equal('off');

  // The status is the one live part of a turn: it says what the turn is doing, and then how it ended.
  expect(status()?.getAttribute('aria-live')).to.equal('polite');
  expect(status()?.textContent).to.equal('Generating answer...');
  list.turns = [{...streaming, state: 'cancelled', streamText: 'The answer so far', activity: null}];
  await list.updateComplete;
  expect(status()?.textContent).to.equal('Answer stopped');
  expect(status()?.getAttribute('aria-live')).to.equal('polite');

  // A turn the server has not yet answered draws its dots without a tail, and they are quiet as well.
  list.turns = [{...streaming, state: 'pending', streamText: '', activity: null}];
  await list.updateComplete;
  expect(list.querySelectorAll('.stream-tail')).to.have.length(0);
  expect(list.querySelector('[data-phase]')?.getAttribute('data-phase')).to.equal('Answer in progress...');
  expect(list.querySelector('[data-phase]')?.getAttribute('aria-live')).to.equal('off');
});

it('raises background intent without treating interactive message controls as background', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'background-intent',
    history: [storedTurn()],
    lineage: null,
  };
  let backgroundIntents = 0;
  feature.addEventListener('dl-chat-background-click', () => { backgroundIntents += 1; });
  document.body.appendChild(feature);
  await settle(feature);

  feature.querySelector<HTMLElement>('main[aria-label="Chat"]')?.click();
  expect(backgroundIntents).to.equal(1);
  const fork = Array.from(feature.querySelectorAll<HTMLButtonElement>('button')).find(
    (button) => button.textContent?.trim() === 'Fork',
  );
  fork?.click();
  expect(backgroundIntents).to.equal(1);
});

it('renders cancelled history without a simultaneous stopping phase', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'cancelled-history',
    lineage: null,
    history: [{
      ...storedTurn(),
      status: 'cancelled',
      cancelRequested: true,
      presentation: null,
    }],
  };
  document.body.appendChild(feature);
  await settle(feature);

  expect(feature.textContent).to.contain('Stopped');
  expect(feature.turns[0].activity).to.equal(null);
});

it('maps every server answer phase and tool name to qualitative deterministic copy', () => {
  expect(ANSWER_PHASE_LABELS).to.deep.equal({
    routing: 'Routing answer...',
    planning: 'Planning answer...',
    searching: 'Searching knowledge base...',
    researching: 'Researching sources...',
    generating: 'Generating answer...',
  });
  for (const [phase, label] of Object.entries(ANSWER_PHASE_LABELS)) {
    expect(answerPhaseLabel(phase)).to.equal(label);
  }
  for (const unknown of ['provider-secret-phase', 'toString', '__proto__']) {
    expect(answerPhaseLabel(unknown)).to.equal(null);
  }
  const known = toolDisplay('load_skill');
  expect(known.known).to.equal(true);
  expect(known.verbId).to.equal('chatFeature.tool.load_skill');
  const unknown = toolDisplay('acme_custom_tool');
  expect(unknown.known).to.equal(false);
  expect(unknown.verb).to.equal('Acme Custom Tool');
  expect(unknown.verbId).to.equal(null);
  const rows = applyToolEvent([], 'tool_start', {tool_name: 'load_skill', call_id: 'c1'}, 1000);
  expect(toolStatusText(applyToolEvent(rows, 'tool_progress', {
    call_id: 'c1', object_label: 'code-review',
  }, 1000))).to.contain('code-review');
  expect(toolDisplay('mcp_connection_deadbeef')).to.deep.equal({
    verb: 'Calling an MCP tool',
    verbId: 'chatFeature.tool.mcp',
    known: false,
  });
});

it('maps and renders every reconnect state with one visible status and action', async () => {
  expect(answerReconnectState(false)).to.equal('running');
  expect(answerReconnectState(true)).to.equal('stopping');
  expect(ANSWER_RECONNECT_COPY).to.deep.equal({
    running: {
      status: 'Connection lost while this answer is running.',
      action: 'Reconnect',
    },
    stopping: {
      status: 'Connection lost while this answer is stopping.',
      action: 'Reconnect',
    },
  });

  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const retryable = (cancelRequested: boolean): ChatTurnView => ({
    id: `turn-${cancelRequested ? 'stopping' : 'running'}`,
    userText: 'Question',
    userAttachments: [],
    runId: `run-${cancelRequested ? 'stopping' : 'running'}`,
    state: 'retryable',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: [],
  });
  list.turns = [retryable(false), retryable(true)];
  document.body.appendChild(list);
  await list.updateComplete;

  for (const state of ['running', 'stopping'] as const) {
    const notice = list.querySelector<HTMLElement>(`[data-reconnect-state="${state}"]`)!;
    expect(notice.querySelector('[role="status"]')?.textContent?.trim())
      .to.equal(ANSWER_RECONNECT_COPY[state].status);
    expect(notice.querySelector('button')?.textContent?.trim())
      .to.equal(ANSWER_RECONNECT_COPY[state].action);
    expect(notice.querySelector('button')?.getAttribute('aria-label'))
      .to.equal('Reconnect to this answer');
  }
});

it('hands accepted work to RunController without announcing a false idle gap', async () => {
  const conversationId = 'submission-handoff';
  const accepted = continuationDescriptor(conversationId);
  conversationStore.adoptCreatedConversation(accepted.conversation);
  let eventsRequested = false;
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === '/web/api/answer' && init?.method === 'POST') {
      return Promise.resolve(new Response(JSON.stringify(acceptedWire(accepted)), {
        status: 202,
        headers: {'Content-Type': 'application/json'},
      }));
    }
    if (url.endsWith('/events')) {
      eventsRequested = true;
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [],
  };
  const activeChanges: boolean[] = [];
  feature.addEventListener('dl-chat-running-change', (event) => {
    activeChanges.push((event as CustomEvent<{active: boolean}>).detail.active);
  });
  document.body.appendChild(feature);
  await settle(feature);

  const composer = feature.querySelector('dl-chat-composer') as DlChatComposer;
  composer.attachmentPolicy = policy;
  chooseFiles(composer, [new File(['notes'], 'handoff.md', {type: 'text/markdown'})]);
  const leasedUrl = attachmentStore.list()[0].objectUrl;
  let authoritativeAtRevoke = false;
  URL.revokeObjectURL = (url) => {
    if (url === leasedUrl) {
      const current = feature.turns.at(-1);
      authoritativeAtRevoke = current?.runId === accepted.turn.answerRunId
        && current.userAttachments.every((attachment) => attachment.url !== leasedUrl);
    }
    originalRevokeObjectURL.call(URL, url);
  };
  const input = feature.querySelector<HTMLTextAreaElement>('[aria-label="Message"]')!;
  input.value = 'Start without flicker';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  await feature.querySelector('dl-chat-composer')?.updateComplete;
  feature.querySelector<HTMLButtonElement>('[aria-label="Send"]')?.click();
  await waitFor(() => eventsRequested);

  expect(activeChanges).to.deep.equal([true]);
  expect(authoritativeAtRevoke).to.equal(true);
  expect(submissions.list()).to.have.length(0);
  feature.detachRun();
});

it('shows cancel-aware reconnect state and preserves stopping when reconnecting', async () => {
  const conversationId = 'conversation-stopping';
  const runId = 'run-stopping';
  const stoppingTurn: ConversationTurn = {
    ...storedTurn(),
    turnId: 'turn-stopping',
    answerRunId: runId,
    status: 'running',
    cancelRequested: true,
    presentation: null,
  };
  let eventRequests = 0;
  let finishFirstEvents!: (response: Response) => void;
  const firstEvents = new Promise<Response>((resolve) => { finishFirstEvents = resolve; });
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/events')) {
      eventRequests += 1;
      if (eventRequests === 1) return firstEvents;
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        signal?.addEventListener(
          'abort',
          () => reject(new DOMException('Aborted', 'AbortError')),
          {once: true},
        );
      });
    }
    if (url === `/web/api/runs/${runId}`) {
      return Promise.resolve(new Response(JSON.stringify(turnWire(stoppingTurn)), {
        status: 200,
        headers: {'Content-Type': 'application/json'},
      }));
    }
    throw new Error(`unexpected fetch: ${url}`);
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId: conversationId,
    title: 'Stopping',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answerRunId: runId,
      turnId: 'turn-stopping',
      status: 'running',
      cancelRequested: true,
      presentation: null,
    }],
  };
  document.body.appendChild(feature);

  await waitFor(() => eventRequests === 1 && feature.turns.length === 1);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    activity: {kind: 'phase', phase: 'planning'},
  }));
  finishFirstEvents(new Response('', {status: 410}));

  await waitFor(() => feature.textContent?.includes(
    'Connection lost while this answer is stopping.',
  ) === true);
  expect(feature.turns[0].activity).to.equal(null);
  expect(feature.querySelectorAll('[data-reconnect-state="stopping"] [role="status"]'))
    .to.have.length(1);
  feature.querySelector<HTMLButtonElement>('[aria-label="Reconnect to this answer"]')?.click();
  await waitFor(() => eventRequests === 2);

  expect(feature.querySelector('dl-chat-composer')?.stopping).to.equal(true);
  expect(feature.turns[0].activity).to.deep.equal({
    kind: 'copy', source: 'Stopping...', id: 'chatFeature.stopping',
  });
});

it('frame-batches 2,000 streamed tokens into bounded Chat and Message List updates', async () => {
  const originalRequestFrame = window.requestAnimationFrame;
  const originalCancelFrame = window.cancelAnimationFrame;
  let nextFrame = 1;
  const frames = new Map<number, FrameRequestCallback>();
  window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    const id = nextFrame;
    nextFrame += 1;
    frames.set(id, callback);
    return id;
  };
  window.cancelAnimationFrame = (id: number): void => {
    frames.delete(id);
  };
  const runFrames = (): void => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => { callback(performance.now()); });
  };

  const conversationId = 'conversation-frame-batching';
  const runId = 'run-frame-batching';
  const expected = Array.from(
    {length: 2_000},
    (_, index) => String.fromCharCode(97 + (index % 26)),
  ).join('');
  const chunks: string[] = [];
  let sequence = 1;
  chunks.push(`id: ${sequence}\nevent: progress\ndata: {"phase":"planning"}\n\n`);
  sequence += 1;
  for (let index = 0; index < expected.length; index += 1) {
    chunks.push(
      `id: ${sequence}\nevent: token\ndata: ${JSON.stringify(expected[index])}\n\n`,
    );
    sequence += 1;
    if (index === 499 || index === 1_499) {
      chunks.push(
        `id: ${sequence}\nevent: memory_operation_settled\n`
        + `data: {"operation":"remember","outcome":"changed",`
        + `"intent_id":"memory-${index}","change_id":"change-${index}"}\n\n`,
      );
      sequence += 1;
    }
  }
  chunks.push(
    `id: ${sequence}\nevent: done\n`
    + 'data: {"status":"cancelled","presentation":null}\n\n',
  );
  const encoder = new TextEncoder();
  let chunkIndex = 0;
  let releaseEvents!: (response: Response) => void;
  const eventResponse = new Promise<Response>((resolve) => { releaseEvents = resolve; });
  window.fetch = ((input: RequestInfo | URL) => {
    if (String(input).endsWith('/events')) return eventResponse;
    return Promise.resolve(new Response('{}', {status: 503}));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId: conversationId,
    title: 'Frame batching',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  try {
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    feature.view = {
      kind: 'ready',
      conversationId,
      lineage: null,
      history: [{
        ...storedTurn(),
        answerRunId: runId,
        turnId: 'turn-frame-batching',
        status: 'running',
        presentation: null,
      }],
    };
    const memoryOperations: string[] = [];
    feature.addEventListener('dl-chat-memory-operation', (event) => {
      const detail = (event as CustomEvent<MemoryOperationEvent>).detail;
      memoryOperations.push(`${detail.intentId}/${detail.changeId}`);
    });
    document.body.appendChild(feature);
    await waitFor(() => feature.querySelector('dl-chat-message-list') !== null);
    await settle(feature);
    runFrames();

    const turnsProperty = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(feature) as object,
      'turns',
    );
    if (!turnsProperty?.get || !turnsProperty.set) throw new Error('turns accessor unavailable');
    let turnAssignments = 0;
    Object.defineProperty(feature, 'turns', {
      configurable: true,
      get: () => turnsProperty.get!.call(feature) as readonly ChatTurnView[],
      set: (value: readonly ChatTurnView[]) => {
        turnAssignments += 1;
        turnsProperty.set!.call(feature, value);
      },
    });
    const list = feature.querySelector('dl-chat-message-list') as DlChatMessageList;
    const instrumentedList = list as unknown as {
      updated: (changed: Map<PropertyKey, unknown>) => void;
    };
    const originalUpdated = instrumentedList.updated.bind(list);
    let listTurnsUpdates = 0;
    instrumentedList.updated = (changed) => {
      if (changed.has('turns')) listTurnsUpdates += 1;
      originalUpdated(changed);
    };

    releaseEvents(new Response(new ReadableStream<Uint8Array>({
      pull(controller): void {
        if (chunkIndex === chunks.length) {
          controller.close();
          return;
        }
        if (chunkIndex > 0 && chunkIndex % 500 === 0) runFrames();
        controller.enqueue(encoder.encode(chunks[chunkIndex]));
        chunkIndex += 1;
      },
    }), {status: 200, headers: {'Content-Type': 'text/event-stream'}}));

    await waitFor(() => feature.turns[0]?.state === 'cancelled');
    await settle(feature);
    runFrames();

    expect(feature.turns[0].streamText).to.equal(expected);
    expect(feature.querySelector(
      '[data-turn-id="turn-frame-batching"] .dl-sr-only[role="status"]',
    )?.textContent).to.equal('Answer stopped');
    expect(memoryOperations).to.deep.equal(['memory-499/change-499', 'memory-1499/change-1499']);
    expect(turnAssignments).to.equal(6);
    expect(listTurnsUpdates).to.equal(5);
    // The turn has one status of its own; the older-page statuses and the composer's count of skills are others'.
    expect(feature.querySelectorAll(
      '[role="status"]:not([data-load-older-status]):not([data-skill-status])',
    )).to.have.length(1);
    expect(feature.textContent).to.contain('Stopped');
  } finally {
    window.requestAnimationFrame = originalRequestFrame;
    window.cancelAnimationFrame = originalCancelFrame;
  }
});

it('shows the answer a live done frame carries when no refresh can supply it', async () => {
  const conversationId = 'conversation-done-frame';
  const runId = 'run-done-frame';
  const done = {status: 'succeeded', presentation: presentationWire(presentation), usage: {}};
  // Only the stream answers: the history read that follows the run and every other
  // request fail, so the frame alone has to supply the answer.
  window.fetch = ((input: RequestInfo | URL) => {
    if (!String(input).endsWith('/events')) return Promise.resolve(new Response('{}', {status: 503}));
    return Promise.resolve(new Response(
      'id: 1\nevent: token\ndata: "A stored"\n\n'
      + `id: 2\nevent: done\ndata: ${JSON.stringify(done)}\n\n`,
      {status: 200, headers: {'Content-Type': 'text/event-stream'}},
    ));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Done frame',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answerRunId: runId,
      turnId: 'turn-done-frame',
      status: 'running',
      presentation: null,
    }],
  };
  document.body.appendChild(feature);

  await waitFor(() => feature.turns[0]?.state === 'succeeded');
  await settle(feature);

  expect(feature.turns[0].state).to.equal('succeeded');
  expect(feature.turns[0].presentation).to.deep.equal(presentation);
  expect(feature.querySelector('dl-answer-presentation')?.textContent).to.contain('A stored answer.');
});

it('rolls the reasoning folds and the tool rows into one growing live process trace', async () => {
  const conversationId = 'conversation-process-trace';
  const runId = 'run-process-trace';
  const encoder = new TextEncoder();
  let send!: (chunk: string) => void;
  let close!: () => void;
  const events = new ReadableStream<Uint8Array>({
    start(controller) {
      send = (chunk) => controller.enqueue(encoder.encode(chunk));
      close = () => controller.close();
    },
  });
  let sequence = 1;
  const sse = (eventType: string, data: unknown): string => {
    const frame = `id: ${sequence}\nevent: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
    sequence += 1;
    return frame;
  };
  window.fetch = ((input: RequestInfo | URL) => {
    if (!String(input).endsWith('/events')) return Promise.resolve(new Response('{}', {status: 503}));
    return Promise.resolve(new Response(events, {
      status: 200, headers: {'Content-Type': 'text/event-stream'},
    }));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Process trace',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answerRunId: runId,
      turnId: 'turn-process-trace',
      status: 'running',
      presentation: null,
    }],
  };
  document.body.appendChild(feature);
  await waitFor(() => feature.turns.length === 1);
  const entries = (): readonly ProcessEntry[] => feature.turns[0]?.processEntries ?? [];
  const open = (index: number): boolean | null => {
    const entry = entries()[index];
    return entry?.kind === 'thinking' ? entry.open : null;
  };

  send(sse('reasoning', 'Weigh the filing.'));
  await waitFor(() => entries().length === 1);
  expect(entries().map((entry) => entry.kind)).to.deep.equal(['thinking']);
  expect(open(0), 'the fold being written is open').to.equal(true);

  send(sse('tool_start', {tool_name: 'search_web', call_id: 'c1'}));
  await waitFor(() => entries().length === 2);
  expect(entries().map((entry) => entry.kind)).to.deep.equal(['thinking', 'tool']);
  expect(open(0), 'the call shuts the fold that chose it').to.equal(false);

  send(sse('reasoning', 'Now read the cap.'));
  await waitFor(() => entries().length === 3);
  send(sse('tool_start', {tool_name: 'read', call_id: 'c2'}));
  await waitFor(() => entries().length === 4);
  send(sse('reasoning', 'One more thought.'));
  await waitFor(() => entries().length === 5);

  // The sixth step joins the history: the block's own scroll reads it, and no step is rolled out.
  send(sse('tool_start', {tool_name: 'read', call_id: 'c3'}));
  await waitFor(() => entries().length === 6 && entries().at(-1)?.kind === 'tool');
  expect(entries().map((entry) => entry.kind))
    .to.deep.equal(['thinking', 'tool', 'thinking', 'tool', 'thinking', 'tool']);
  expect(
    entries().flatMap((entry) => (entry.kind === 'thinking' ? [entry.text] : [])),
    'every thought the Run showed stays in the trace',
  ).to.deep.equal(['Weigh the filing.', 'Now read the cap.', 'One more thought.']);

  send(sse('done', {status: 'cancelled', presentation: null}));
  close();
  await waitFor(() => feature.turns[0]?.state === 'cancelled');
  expect(entries(), 'a settled turn keeps no live trace').to.deep.equal([]);
});

it('keeps the first answer of a new conversation when the history refresh after it fails', async () => {
  const conversationId = 'conversation-refresh-fails';
  const runId = 'run-refresh-fails';
  const done = {status: 'succeeded', presentation: presentationWire(presentation), usage: {}};
  let historyReads = 0;
  window.fetch = ((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes('/history')) historyReads += 1;
    if (!url.endsWith('/events')) return Promise.resolve(new Response('{}', {status: 503}));
    return Promise.resolve(new Response(
      'id: 1\nevent: token\ndata: "A stored"\n\n'
      + `id: 2\nevent: done\ndata: ${JSON.stringify(done)}\n\n`,
      {status: 200, headers: {'Content-Type': 'text/event-stream'}},
    ));
  }) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  followConversationView(feature);
  await feature.updateComplete;
  // The first accepted answer: the store adopts the conversation without loaded
  // history, and the live answer owns the viewport.
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Refresh fails',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answerRunId: runId,
      turnId: 'turn-refresh-fails',
      status: 'running',
      presentation: null,
    }],
  };

  // The history read is the refresh that follows the finished run.
  await waitFor(() => historyReads > 0);
  // The failed read reaches the store a task later; let it travel to the Feature.
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await settle(feature);

  expect(historyReads).to.equal(1);
  expect(feature.textContent).to.not.contain('Conversation history is unavailable.');
  expect(feature.querySelector('[role="alert"]')).to.equal(null);
  expect(feature.querySelector('dl-answer-presentation')?.textContent).to.contain('A stored answer.');
  expect(conversationStore.canAnswer).to.equal(true);
  expect(conversationStore.answerConversationId).to.equal(conversationId);
});

it('hands an open dock every batch of its Run\'s events, and its end', async () => {
  const originalRequestFrame = window.requestAnimationFrame;
  const originalCancelFrame = window.cancelAnimationFrame;
  let nextFrame = 1;
  const frames = new Map<number, FrameRequestCallback>();
  window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    const id = nextFrame;
    nextFrame += 1;
    frames.set(id, callback);
    return id;
  };
  window.cancelAnimationFrame = (id: number): void => { frames.delete(id); };
  const runFrames = (): void => {
    const pending = [...frames.values()];
    frames.clear();
    pending.forEach((callback) => { callback(performance.now()); });
  };
  const conversationId = 'conversation-child-activity';
  const runId = 'run-child-activity';
  const chunks = [
    'id: 1\nevent: tool_start\ndata: {"tool_name":"search_corpus","call_id":"search-0"}\n\n',
    'id: 2\nevent: tool_start\ndata: {"tool_name":"spawn_agent","call_id":"child-1"}\n\n',
    'id: 3\nevent: token\ndata: "Deleg"\n\n',
    'id: 4\nevent: token\ndata: "ated"\n\n',
    'id: 5\nevent: progress\ndata: {"phase":"searching"}\n\n',
    'id: 6\nevent: tool_start\ndata: {"tool_name":"search_corpus","call_id":"search-1"}\n\n',
    'id: 7\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n',
  ];
  const encoder = new TextEncoder();
  let chunkIndex = 0;
  window.fetch = ((input: RequestInfo | URL) => {
    if (!String(input).endsWith('/events')) return Promise.resolve(new Response('{}', {status: 503}));
    return Promise.resolve(new Response(new ReadableStream<Uint8Array>({
      pull(controller): void {
        // Each chunk lands in its own frame batch.
        runFrames();
        if (chunkIndex === chunks.length) {
          controller.close();
          return;
        }
        controller.enqueue(encoder.encode(chunks[chunkIndex]));
        chunkIndex += 1;
      },
    }), {status: 200, headers: {'Content-Type': 'text/event-stream'}}));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Child activity',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  try {
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    const activity: {runId: string; kinds: string[]}[] = [];
    feature.addEventListener('dl-run-activity', (event) => {
      const detail = (event as CustomEvent<ChatRunActivityDetail>).detail;
      activity.push({runId: detail.runId, kinds: detail.events.map((one) => one.kind)});
    });
    feature.view = {
      kind: 'ready',
      conversationId,
      lineage: null,
      history: [{
        ...storedTurn(),
        answerRunId: runId,
        turnId: 'turn-child-activity',
        status: 'running',
        presentation: null,
      }],
    };
    document.body.appendChild(feature);

    await waitFor(() => feature.turns[0]?.state === 'cancelled');
    runFrames();

    expect(feature.turns[0].streamText).to.equal('Delegated');
    // Every batch is the dock's to draw, tokens included: a live answer is a token stream, and the
    // controller has already folded each frame's tokens into one batch.
    expect(activity.map((one) => one.runId)).to.deep.equal(activity.map(() => runId));
    expect(activity.some((one) => one.kinds.includes('token')), 'tokens are the chat\'s, not the dock\'s').to.equal(false);
    expect(activity.some((one) => one.kinds.includes('tool')), 'tool events reach the dock').to.equal(true);
    expect(activity.at(-1)!.kinds.includes('done'), "the Run's end reaches the dock").to.equal(true);
  } finally {
    window.requestAnimationFrame = originalRequestFrame;
    window.cancelAnimationFrame = originalCancelFrame;
  }
});

it('announces the end of a run with children that its stored row reports', async () => {
  const originalSetTimeout = window.setTimeout;
  // The reconnect delay after the stream closes runs at once.
  window.setTimeout = ((handler: TimerHandler) => originalSetTimeout(handler, 0)) as typeof window.setTimeout;
  const conversationId = 'conversation-child-stored-end';
  const runId = 'run-child-stored-end';
  const turnId = 'turn-child-stored-end';
  let eventRequests = 0;
  window.fetch = ((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith('/events')) {
      eventRequests += 1;
      if (eventRequests > 1) return Promise.resolve(new Response('', {status: 410}));
      // The stream shows a child, then closes without a done event.
      return Promise.resolve(new Response(
        'id: 1\nevent: tool_start\ndata: {"tool_name":"spawn_agent","call_id":"child-1"}\n\n',
        {status: 200, headers: {'Content-Type': 'text/event-stream'}},
      ));
    }
    if (url === `/web/api/runs/${runId}`) {
      return Promise.resolve(new Response(JSON.stringify(turnWire({
        ...storedTurn(),
        answerRunId: runId,
        turnId,
      })), {status: 200, headers: {'Content-Type': 'application/json'}}));
    }
    return Promise.resolve(new Response('{}', {status: 503}));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Child stored end',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });

  try {
    const feature = document.createElement('dl-chat-feature') as DlChatFeature;
    const activity: string[] = [];
    feature.addEventListener('dl-run-activity', (event) => {
      activity.push((event as CustomEvent<{runId: string}>).detail.runId);
    });
    feature.view = {
      kind: 'ready',
      conversationId,
      lineage: null,
      history: [{...storedTurn(), answerRunId: runId, turnId, status: 'running', presentation: null}],
    };
    document.body.appendChild(feature);

    await waitFor(() => feature.turns[0]?.state === 'succeeded');

    expect(eventRequests).to.equal(2);
    // spawn_agent, then the settled row: the only word of the Run's end.
    expect(activity).to.deep.equal([runId, runId]);
  } finally {
    window.setTimeout = originalSetTimeout;
  }
});

it('Chat Feature deduplicates terminal live turns by authoritative run identity', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'same', history: [], lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = [{
    ...storedTurnView(storedTurn()),
    id: 'local-terminal',
  }];

  feature.view = {
    kind: 'ready', conversationId: 'same', history: [storedTurn()], lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.id).to.equal('turn-1');
  expect(feature.querySelectorAll('[data-run-id="run-1"]')).to.have.length(1);
});

it('Chat Feature lets an authoritative terminal stored turn replace local retryable state', async () => {
  const running = {...storedTurn(), status: 'running' as const, presentation: null};
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'terminal-wins', history: [running], lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    state: 'retryable' as const,
    streamText: 'Partial local answer',
    processEntries: [],
    error: {source: 'Reconnect', id: null},
  }));

  feature.view = {
    kind: 'ready', conversationId: 'terminal-wins', history: [storedTurn()], lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.id).to.equal('turn-1');
  expect(feature.turns[0]?.state).to.equal('succeeded');
  expect(feature.turns[0]?.streamText).to.equal('');
  expect(feature.turns[0]?.presentation).to.deep.equal(presentation);
  expect(feature.querySelectorAll('[data-run-id="run-1"]')).to.have.length(1);
});

it('Chat Feature preserves a non-terminal live projection across history republish', async () => {
  const running = {...storedTurn(), status: 'running' as const, presentation: null};
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'same-running', history: [running], lineage: null,
    olderMessages: {state: 'idle', starting: false, hasOlder: true, outcome: null},
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = feature.turns.map((turn) => ({
    ...turn,
    state: 'streaming' as const,
    streamText: 'Live answer',
    processEntries: [],
  }));

  feature.view = {
    ...feature.view,
    olderMessages: {state: 'loading', starting: false, hasOlder: true, outcome: null},
  };
  await settle(feature);

  expect(feature.turns).to.have.length(1);
  expect(feature.turns[0]?.state).to.equal('streaming');
  expect(feature.turns[0]?.streamText).to.equal('Live answer');
});

it('re-reads the conversation when the goal bar reports a resume', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'goal-resumed',
    lineage: null,
    history: [storedTurn()],
  };
  document.body.appendChild(feature);
  await settle(feature);

  const conversations = feature.handles.conversations;
  const original = conversations.refreshActive;
  let refreshed = 0;
  conversations.refreshActive = async () => {
    refreshed += 1;
    return 'stale';
  };
  try {
    feature
      .querySelector('dl-goal-bar')
      ?.dispatchEvent(new CustomEvent('dl-goal-resumed', {bubbles: true}));
    await settle(feature);
  } finally {
    conversations.refreshActive = original;
  }
  expect(refreshed).to.equal(1);
});

it('shows the goal the Agent creates mid-run without a reload', async () => {
  const conversationId = 'conversation-goal-midrun';
  const runId = 'run-goal-midrun';
  const goal = {
    conversation_id: conversationId,
    goal_id: 'goal-midrun',
    revision: 1,
    objective: 'finish the sealed model',
    phase: 'active',
    max_rounds: 8,
    rounds_started: 1,
    blocked_reason: null,
  };
  let written = false;
  let goalReads = 0;
  const encoder = new TextEncoder();
  const chunks = [
    'id: 1\nevent: tool_start\ndata: {"tool_name":"create_goal","call_id":"c-goal"}\n\n',
    'id: 2\nevent: tool_end\n'
    + 'data: {"tool_name":"create_goal","call_id":"c-goal","outcome":"succeeded"}\n\n',
  ];
  let nextChunk = 0;
  let releaseEvents!: (response: Response) => void;
  const eventsResponse = new Promise<Response>((resolve) => { releaseEvents = resolve; });
  window.fetch = ((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith(`/conversations/${conversationId}/goal`)) {
      goalReads += 1;
      return Promise.resolve(Response.json(written ? goal : null));
    }
    if (url.endsWith('/events')) return eventsResponse;
    return Promise.resolve(new Response('{}', {status: 503}));
  }) as typeof fetch;
  conversationStore.adoptCreatedConversation({
    conversationId,
    title: 'Mid-run goal',
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    forkedFromConversationId: null,
    forkedFromTitle: null,
  });
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId,
    lineage: null,
    history: [{
      ...storedTurn(),
      answerRunId: runId,
      turnId: 'turn-goal-midrun',
      status: 'running',
      presentation: null,
    }],
  };
  document.body.appendChild(feature);

  const barText = (): string => feature.querySelector('dl-goal-bar')?.textContent ?? '';
  await waitFor(() => goalReads >= 1);
  await settle(feature);
  expect(barText()).to.not.contain('Ongoing goal', 'no Goal before the Tool settles');
  const readsBefore = goalReads;

  // The Run is still going: the bar must show the Goal from the settled Tool alone.
  const events = new ReadableStream<Uint8Array>({
    pull(controller): void | Promise<void> {
      if (nextChunk >= chunks.length) return new Promise<void>(() => {});
      const chunk = chunks[nextChunk]!;
      if (chunk.includes('tool_end')) written = true;
      controller.enqueue(encoder.encode(chunk));
      nextChunk += 1;
    },
  });
  releaseEvents(new Response(events, {
    status: 200,
    headers: {'Content-Type': 'text/event-stream'},
  }));

  await waitFor(() => barText().includes('Ongoing goal'));

  expect(goalReads).to.be.greaterThan(readsBefore, 'the bar re-read the Goal at the Tool settle');
  expect(['pending', 'streaming']).to.include(
    feature.turns[0]?.state,
    'the Run has not settled: the Goal appeared mid-Run',
  );
  feature.detachRun();
});

it('Chat Feature drops stale terminal ranges while preserving a pending optimistic turn', async () => {
  const oldHistory = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turnId: `turn-${index + 1}`,
    turnNumber: index + 1,
    answerRunId: `run-${index + 1}`,
    submissionId: `submission-${index + 1}`,
  }));
  const recentHistory = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turnId: `turn-${index + 51}`,
    turnNumber: index + 51,
    answerRunId: `run-${index + 51}`,
    submissionId: `submission-${index + 51}`,
  }));
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready', conversationId: 'gap-replaced', history: oldHistory, lineage: null,
  };
  document.body.appendChild(feature);
  await settle(feature);
  feature.turns = [...feature.turns, {
    ...storedTurnView(storedTurn()),
    id: 'local-pending',
    runId: '',
    state: 'pending',
    userText: 'Pending optimistic question',
    streamText: '',
    processEntries: [],
    presentation: null,
  }];

  feature.view = {
    kind: 'ready', conversationId: 'gap-replaced', history: recentHistory, lineage: null,
  };
  await settle(feature);

  expect(feature.turns).to.have.length(41);
  expect(feature.turns.some((turn) => turn.id === 'turn-1')).to.equal(false);
  expect(feature.turns.filter((turn) => turn.runId).map((turn) => turn.id)).to.deep.equal(
    Array.from({length: 40}, (_, index) => `turn-${index + 51}`),
  );
  expect(feature.turns.at(-1)?.id).to.equal('local-pending');
  expect(feature.turns.at(-1)?.state).to.equal('pending');
});

it('Chat Feature renders every explicitly loaded history page without a product cap', async () => {
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  feature.view = {
    kind: 'ready',
    conversationId: 'bounded-history',
    lineage: null,
    history: Array.from({length: 101}, (_, index) => ({
      ...storedTurn(),
      turnId: `turn-${index}`,
      answerRunId: `run-${index}`,
      turnNumber: index + 1,
      userText: `Question ${index}`,
    })),
  };
  document.body.appendChild(feature);
  await settle(feature);

  expect(feature.turns).to.have.length(101);
  expect(feature.querySelectorAll('[data-turn-slot]').length).to.equal(101);
  const mounted = feature.querySelector('[role="log"]')?.querySelectorAll('article').length ?? 0;
  expect(mounted).to.be.greaterThan(0);
  expect(mounted).to.be.lessThan(101);
});

it('pages more than 40 turns through the wired store, the route view, and Load older control', async () => {
  const recent = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turnId: `turn-${index + 41}`,
    turnNumber: index + 41,
    answerRunId: `run-${index + 41}`,
    submissionId: `submission-${index + 41}`,
    userText: `Question ${index + 41}`,
  }));
  const older = Array.from({length: 40}, (_, index) => ({
    ...storedTurn(),
    turnId: `turn-${index + 1}`,
    turnNumber: index + 1,
    answerRunId: `run-${index + 1}`,
    submissionId: `submission-${index + 1}`,
    userText: `Question ${index + 1}`,
  }));
  window.fetch = (async (input) => {
    const url = new URL(String(input), window.location.origin);
    if (url.pathname !== '/web/api/conversations/wired/history') {
      throw new Error(`unexpected fetch: ${url}`);
    }
    const isOlder = url.searchParams.get('cursor') === 'before-41';
    return new Response(JSON.stringify({
      conversation: {
        conversation_id: 'wired', title: 'Wired',
        created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
      },
      turns: (isOlder ? older : recent).map(turnWire),
      commands: [],
      next_cursor: isOlder ? null : 'before-41',
    }), {status: 200, headers: {'Content-Type': 'application/json'}});
  }) as typeof fetch;
  const feature = document.createElement('dl-chat-feature') as DlChatFeature;
  document.body.append(feature);
  followConversationView(feature);

  await conversationStore.open('wired');
  await settle(feature);
  feature.querySelector<DlChatMessageList>('dl-chat-message-list')
    ?.querySelector<HTMLButtonElement>('[data-load-older="messages"]')?.click();
  await waitFor(() => conversationStore.history?.turns.length === 80);
  await settle(feature);

  expect(feature.querySelectorAll('[data-turn-slot]')).to.have.length(80);
  expect(feature.querySelector('[data-turn-id="turn-1"]')).not.to.equal(null);
  expect(feature.textContent).to.contain('Question 80');
  expect(feature.querySelector('[data-load-older="messages"]')).to.equal(null);
});

it('renders a named, timed tool trace and ticks only while a row is running', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const rows: ChatTurnView['toolRows'] = [
    {
      callId: 'c1',
      label: 'Personal tools · search_issues',
      name: 'mcp_connection_deadbeef',
      verb: 'Calling an MCP tool',
      verbId: 'chatFeature.tool.mcp',
      object: '',
      state: 'running',
      startedAt: performance.now() - 400,
      durationMs: null,
    },
    {
      callId: 'c2',
      label: null,
      name: 'read',
      verb: 'Reading a document',
      verbId: 'chatFeature.tool.read',
      object: 'report.pdf',
      state: 'done',
      startedAt: null,
      durationMs: 2400,
    },
    {
      callId: 'c3',
      label: null,
      name: 'bash',
      verb: 'Running a command',
      verbId: 'chatFeature.tool.bash',
      object: '',
      state: 'failed',
      startedAt: null,
      durationMs: 30,
    },
  ];
  const entries = (held: ChatTurnView['toolRows']): readonly ProcessEntry[] => (
    held.map((row) => ({kind: 'tool', row}))
  );
  list.turns = [{
    id: 'turn-trace',
    userText: 'Question',
    userAttachments: [],
    runId: 'run-trace',
    state: 'streaming',
    streamText: '',
    processEntries: entries(rows),
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null, feedback: null,
    toolRows: rows,
  }];
  document.body.append(list);
  await list.updateComplete;

  const text = (): string => list.textContent ?? '';
  expect(text()).to.contain('Personal tools · search_issues');
  expect(text()).to.not.contain('Deadbeef');
  expect(text()).to.contain('Reading a document — report.pdf');
  expect(text()).to.contain('2.4s');
  expect(text()).to.contain('30ms');
  expect(text()).to.not.contain('0.4s', 'a running row stays quiet in its first second');
  const ticking = (): string[] => [...list.querySelectorAll('[aria-hidden="true"]')]
    .map((element) => element.textContent ?? '')
    .filter((label) => /^\d+(\.\d+)?(ms|s)$/.test(label));
  expect(ticking()).to.deep.equal([], 'nothing ticks until a second has passed');

  const deadline = performance.now() + 3000;
  while (performance.now() < deadline && ticking().length === 0) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  expect(text()).to.match(/\b1\.\ds\b/, 'the running row counts up on its own');
  expect(ticking()).to.have.lengthOf(1, 'only the running counter ticks');
  expect(ticking()[0]).to.match(/^1\.\ds$/, 'the ticking counter stays out of the live region');

  const settled = rows.map((row) => row.state === 'running'
    ? {...row, state: 'done' as const, startedAt: null, durationMs: 8000}
    : row);
  list.turns = list.turns.map((turn) => ({
    ...turn,
    toolRows: settled,
    processEntries: entries(settled),
  }));
  await list.updateComplete;
  expect(text()).to.contain('8.0s', 'server truth replaces the viewer clock');
  list.remove();
});

it('marks a browsing Run with a host the reader can open the live browser from', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const browsing = (overrides: Partial<ChatTurnView>): ChatTurnView => ({
    id: 'turn-live',
    userText: 'Question',
    userAttachments: [],
    runId: 'run-live',
    state: 'streaming',
    streamText: '',
    processEntries: [],
    presentation: null,
    usage: {},
    error: null,
    activity: null,
    cancelRequested: false,
    steeringMessages: [],
    answerSequence: null,
    feedback: null,
    toolRows: [{
      callId: 'c1',
      label: null,
      name: 'browser',
      verb: 'Browsing the web',
      verbId: 'chatFeature.tool.browser',
      object: 'example.com/pricing',
      state: 'running',
      startedAt: performance.now(),
      durationMs: null,
    }],
    ...overrides,
  });
  list.turns = [browsing({})];
  document.body.append(list);
  await list.updateComplete;

  const indicator = (): HTMLButtonElement | null => list.querySelector<HTMLButtonElement>('.liveButton');
  expect(indicator()).to.not.equal(null, 'a browsing Run offers its live browser');
  expect(indicator()!.textContent).to.contain('example.com');
  expect(indicator()!.getAttribute('aria-label')).to.contain('example.com');

  const actions: ChatRunActionDetail[] = [];
  list.addEventListener('dl-chat-run-action', (event) => {
    actions.push((event as CustomEvent<ChatRunActionDetail>).detail);
  });
  indicator()!.click();
  expect(actions).to.deep.equal([{action: 'live', runId: 'run-live'}]);

  // Nothing opens the dock by itself: the turn draws the way in, and no event at all is
  // raised until the reader takes it.
  expect(actions).to.have.length(1);

  // A Run that is not browsing has nothing to offer, and a settled one has released its
  // browser: the indicator is only ever the live turn's.
  list.turns = [browsing({id: 'turn-other', runId: 'run-other', toolRows: []})];
  await list.updateComplete;
  expect(indicator()).to.equal(null, 'no browser row, no indicator');

  list.turns = [browsing({id: 'turn-settled', runId: 'run-settled', state: 'succeeded'})];
  await list.updateComplete;
  expect(indicator()).to.equal(null, 'a settled Run has given its browser back');

  list.remove();
});

it('says what a settled Run spent, and never repeats the source count', async () => {
  const list = document.createElement('dl-chat-message-list') as DlChatMessageList;
  const turn = storedTurn();
  turn.presentation = {...presentation, sources: [
    {id: '1', title: 'One source', sourceUrl: 'https://example.com/one', downloadUrl: null, chunks: []},
    {id: '2', title: 'Another source', sourceUrl: 'https://example.com/two', downloadUrl: null, chunks: []},
  ]};
  turn.usage = {usage_details: {total_tokens: 1500}};
  list.turns = [storedTurnView(turn)];
  document.body.appendChild(list);
  await list.updateComplete;
  // The row says what the Run spent; References carries how many sources the
  // answer cites, so the count is not said twice.
  expect(list.querySelector('.runUsage')!.textContent?.trim()).to.equal('1.5K tokens');
  expect(list.querySelector('[data-turn-id="turn-1"]')?.textContent).to.not.contain('sources');
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Builds the `/web/api/answer` request body for one submission. With no attachments
// it posts the compact JSON envelope; with attachments it posts one multipart
// form carrying the same envelope fields plus repeated `attachments` file parts
// in user order. The server contract (parse_web_answer_request) reads exactly
// these names: query, workspaces, optional conversation_id, submission_id, attachments.

export type AnswerMode = 'auto' | 'fast' | 'research';

export interface AnswerEnvelope {
    query: string;
    workspaces: string[];
    conversationId: string | null;
    submissionId: string;
    mode?: AnswerMode;
}

export interface AnswerRequestInit {
    body: BodyInit;
    headers?: Record<string, string>;
}

export function buildAnswerRequest(
    envelope: AnswerEnvelope,
    attachments: readonly File[],
): AnswerRequestInit {
    if (attachments.length === 0) {
        return {
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: envelope.query,
                workspaces: envelope.workspaces,
                conversation_id: envelope.conversationId,
                submission_id: envelope.submissionId,
                ...(envelope.mode ? {mode: envelope.mode} : {}),
            }),
        };
    }
    const form = new FormData();
    form.append('query', envelope.query);
    form.append('workspaces', JSON.stringify(envelope.workspaces));
    if (envelope.conversationId) form.append('conversation_id', envelope.conversationId);
    form.append('submission_id', envelope.submissionId);
    if (envelope.mode) form.append('mode', envelope.mode);
    for (const file of attachments) form.append('attachments', file, file.name);
    return {body: form};
}

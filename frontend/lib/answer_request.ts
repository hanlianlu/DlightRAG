// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Builds the `/web/answer` request body for one submission. With no attachments
// it posts the compact JSON envelope; with attachments it posts one multipart
// form carrying the same envelope fields plus repeated `attachments` file parts
// in user order. The server contract (parse_web_answer_request) reads exactly
// these names: query, workspaces, conversation_id, submission_id, attachments.

export interface AnswerEnvelope {
    query: string;
    workspaces: string[];
    conversationId: string;
    submissionId: string;
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
            }),
        };
    }
    const form = new FormData();
    form.append('query', envelope.query);
    form.append('workspaces', JSON.stringify(envelope.workspaces));
    form.append('conversation_id', envelope.conversationId);
    form.append('submission_id', envelope.submissionId);
    for (const file of attachments) form.append('attachments', file, file.name);
    return {body: form};
}

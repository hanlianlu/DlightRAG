// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {buildAnswerRequest} from './answer-request.ts';

function file(name: string, type: string): File {
  return new File([name], name, {type});
}

const envelope = {
  query: 'what is this?',
  workspaces: ['alpha', 'beta'],
  conversationId: 'conv-1',
  submissionId: 'sub-1',
};

test('no attachments emits the JSON envelope with the Web form fields only', () => {
  const {body, headers} = buildAnswerRequest(envelope, []);

  assert.equal(headers?.['Content-Type'], 'application/json');
  assert.equal(typeof body, 'string');
  const parsed = JSON.parse(body as string) as Record<string, unknown>;
  assert.deepEqual(parsed, {
    query: 'what is this?',
    workspaces: ['alpha', 'beta'],
    conversation_id: 'conv-1',
    submission_id: 'sub-1',
  });
  assert.ok(!('images' in parsed));
  assert.ok(!('documents' in parsed));
});

test('a first JSON submission explicitly carries no conversation', () => {
  const {body} = buildAnswerRequest({...envelope, conversationId: null}, []);

  const parsed = JSON.parse(body as string) as Record<string, unknown>;
  assert.equal(parsed.conversation_id, null);
  assert.equal(parsed.submission_id, 'sub-1');
});

test('a requested skill rides both the JSON and multipart envelopes', () => {
  const withSkill = {...envelope, requestedSkill: 'review'};

  const jsonBody = buildAnswerRequest(withSkill, []).body as string;
  const parsed = JSON.parse(jsonBody) as Record<string, unknown>;
  assert.equal(parsed.requested_skill, 'review');

  const multipartBody = buildAnswerRequest(withSkill, [file('a.png', 'image/png')]).body;
  assert.ok(multipartBody instanceof FormData);
  assert.equal((multipartBody as FormData).get('requested_skill'), 'review');
});

test('a submission without a requested skill never emits the field', () => {
  const jsonBody = buildAnswerRequest(envelope, []).body as string;
  const parsed = JSON.parse(jsonBody) as Record<string, unknown>;
  assert.ok(!('requested_skill' in parsed));
});

test('mixed attachments emit one multipart with the envelope plus repeated attachments in order', () => {
  const attachments = [
    file('a.png', 'image/png'),
    file('b.pdf', 'application/pdf'),
    file('c.png', 'image/png'),
  ];
  const {body, headers} = buildAnswerRequest(envelope, attachments);

  assert.equal(headers, undefined);
  assert.ok(body instanceof FormData);
  const form = body as FormData;

  assert.equal(form.get('query'), 'what is this?');
  assert.equal(form.get('workspaces'), JSON.stringify(['alpha', 'beta']));
  assert.equal(form.get('conversation_id'), 'conv-1');
  assert.equal(form.get('submission_id'), 'sub-1');

  assert.equal(form.getAll('images').length, 0);
  assert.equal(form.getAll('documents').length, 0);

  const parts = form.getAll('attachments');
  assert.equal(parts.length, 3);
  assert.deepEqual(
    parts.map((part) => (part as File).name),
    ['a.png', 'b.pdf', 'c.png'],
  );
});

test('a first multipart submission omits the optional conversation field', () => {
  const {body} = buildAnswerRequest(
    {...envelope, conversationId: null},
    [file('report.pdf', 'application/pdf')],
  );

  const form = body as FormData;
  assert.equal(form.has('conversation_id'), false);
  assert.equal(form.get('submission_id'), 'sub-1');
});

test('a rebuilt request from the same attachments never duplicates the file parts', () => {
  const attachments = [file('a.png', 'image/png'), file('b.pdf', 'application/pdf')];
  const first = buildAnswerRequest(envelope, attachments).body as FormData;
  const second = buildAnswerRequest(envelope, attachments).body as FormData;

  assert.equal(first.getAll('attachments').length, 2);
  assert.equal(second.getAll('attachments').length, 2);
});

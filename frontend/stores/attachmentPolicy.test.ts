// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  acceptsDocumentUpload,
  classifyAttachmentFile,
  getDocumentAdmissionPolicy,
} from '../ui/attachment_policy.ts';

function file(name: string, type: string, size: number): File {
  return {name, type, size} as File;
}

test('mixed attachment classification separates images and documents', () => {
  assert.equal(classifyAttachmentFile(file('chart.png', 'image/png', 10)), 'image');
  assert.equal(classifyAttachmentFile(file('report.pdf', 'application/pdf', 10)), 'document');
  assert.equal(classifyAttachmentFile(file('notes.md', '', 10)), 'document');
  assert.equal(classifyAttachmentFile(file('archive.zip', 'application/zip', 10)), 'unsupported');
  assert.equal(classifyAttachmentFile(file('noext', '', 10)), 'unsupported');
});

test('document admission enforces independent count and byte limits', () => {
  const policy = {currentUploadLimit: 3, maxUploadBytes: 100 * 1024 * 1024};

  assert.equal(
    acceptsDocumentUpload(file('report.pdf', 'application/pdf', policy.maxUploadBytes), 0, policy),
    true,
  );
  assert.equal(
    acceptsDocumentUpload(file('report.pdf', 'application/pdf', policy.maxUploadBytes + 1), 0, policy),
    false,
  );
  assert.equal(acceptsDocumentUpload(file('report.pdf', 'application/pdf', 1), 3, policy), false);
  assert.equal(acceptsDocumentUpload(file('chart.png', 'image/png', 1), 0, policy), false);
});

test('document policy reads server projected data attributes', () => {
  const root = {
    dataset: {
      documentCurrentUploadLimit: '3',
      documentMaxUploadBytes: '104857600',
    },
  } as unknown as HTMLElement;

  assert.deepEqual(getDocumentAdmissionPolicy(root), {
    currentUploadLimit: 3,
    maxUploadBytes: 104857600,
  });
});

test('document policy fails closed for missing or malformed attributes', () => {
  assert.equal(getDocumentAdmissionPolicy(null), null);
  assert.equal(
    getDocumentAdmissionPolicy({dataset: {documentCurrentUploadLimit: '3'}} as unknown as HTMLElement),
    null,
  );
  assert.equal(
    getDocumentAdmissionPolicy(
      {dataset: {documentCurrentUploadLimit: 'three', documentMaxUploadBytes: '100'}} as unknown as HTMLElement,
    ),
    null,
  );
  assert.equal(
    getDocumentAdmissionPolicy(
      {dataset: {documentCurrentUploadLimit: '3', documentMaxUploadBytes: '0'}} as unknown as HTMLElement,
    ),
    null,
  );
});

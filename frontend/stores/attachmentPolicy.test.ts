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
  const extensions = new Set(['pdf', 'md']);

  assert.equal(classifyAttachmentFile(file('chart.png', 'image/png', 10), extensions), 'image');
  assert.equal(classifyAttachmentFile(file('report.pdf', 'application/pdf', 10), extensions), 'document');
  assert.equal(classifyAttachmentFile(file('notes.md', '', 10), extensions), 'document');
  assert.equal(classifyAttachmentFile(file('notes.markdown', '', 10), extensions), 'unsupported');
  assert.equal(classifyAttachmentFile(file('archive.zip', 'application/zip', 10), extensions), 'unsupported');
  assert.equal(classifyAttachmentFile(file('noext', '', 10), extensions), 'unsupported');
});

test('document admission enforces independent count and byte limits', () => {
  const policy = {
    currentUploadLimit: 3,
    maxUploadBytes: 100 * 1024 * 1024,
    extensions: new Set(['pdf']),
  };

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
  assert.equal(acceptsDocumentUpload(file('notes.md', 'text/markdown', 1), 0, policy), false);
});

test('document policy reads server projected data attributes', () => {
  const root = {
    dataset: {
      documentCurrentUploadLimit: '3',
      documentMaxUploadBytes: '104857600',
      documentExtensions: '["md","pdf"]',
    },
  } as unknown as HTMLElement;

  const policy = getDocumentAdmissionPolicy(root);

  assert.equal(policy?.currentUploadLimit, 3);
  assert.equal(policy?.maxUploadBytes, 104857600);
  assert.deepEqual(policy?.extensions, new Set(['md', 'pdf']));
});

test('document policy fails closed for missing or malformed attributes', () => {
  assert.equal(getDocumentAdmissionPolicy(null), null);
  assert.equal(
    getDocumentAdmissionPolicy(
      {
        dataset: {
          documentCurrentUploadLimit: '3',
          documentMaxUploadBytes: '100',
        },
      } as unknown as HTMLElement,
    ),
    null,
  );
  assert.equal(
    getDocumentAdmissionPolicy(
      {
        dataset: {
          documentCurrentUploadLimit: 'three',
          documentMaxUploadBytes: '100',
          documentExtensions: '["pdf"]',
        },
      } as unknown as HTMLElement,
    ),
    null,
  );
  assert.equal(
    getDocumentAdmissionPolicy(
      {
        dataset: {
          documentCurrentUploadLimit: '3',
          documentMaxUploadBytes: '0',
          documentExtensions: '["pdf"]',
        },
      } as unknown as HTMLElement,
    ),
    null,
  );
  assert.equal(
    getDocumentAdmissionPolicy(
      {
        dataset: {
          documentCurrentUploadLimit: '3',
          documentMaxUploadBytes: '100',
          documentExtensions: '["pdf",42]',
        },
      } as unknown as HTMLElement,
    ),
    null,
  );
});

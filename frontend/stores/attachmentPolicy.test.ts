// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  acceptsAttachmentUpload,
  attachmentsEnabled,
  classifyAttachmentFile,
  getAttachmentPolicy,
} from '../ui/attachment_policy.ts';

function file(name: string, type: string, size: number): File {
  return {name, type, size} as File;
}

const policy = {
  countLimit: 6,
  imageMaxBytes: 15 * 1024 * 1024,
  documentMaxBytes: 100 * 1024 * 1024,
  extensions: new Set(['pdf', 'md']),
  imageCapability: 'supported' as const,
  imageLimit: 3,
};

test('one collection classifies images and documents by MIME then extension', () => {
  const extensions = new Set(['pdf', 'md']);
  assert.equal(classifyAttachmentFile(file('chart.png', 'image/png', 10), extensions), 'image');
  assert.equal(classifyAttachmentFile(file('report.pdf', 'application/pdf', 10), extensions), 'document');
  assert.equal(classifyAttachmentFile(file('notes.md', '', 10), extensions), 'document');
  assert.equal(classifyAttachmentFile(file('notes.markdown', '', 10), extensions), 'unsupported');
  assert.equal(classifyAttachmentFile(file('archive.zip', 'application/zip', 10), extensions), 'unsupported');
  assert.equal(classifyAttachmentFile(file('noext', '', 10), extensions), 'unsupported');
});

test('admission shares one count limit across images and documents', () => {
  assert.equal(
    acceptsAttachmentUpload(file('report.pdf', 'application/pdf', 10), {total: 5, images: 2}, policy),
    true,
  );
  assert.equal(
    acceptsAttachmentUpload(file('report.pdf', 'application/pdf', 10), {total: 6, images: 2}, policy),
    false,
  );
});

test('images honor capability, image sub-limit, and per-item bytes', () => {
  assert.equal(
    acceptsAttachmentUpload(file('a.png', 'image/png', policy.imageMaxBytes), {total: 0, images: 0}, policy),
    true,
  );
  assert.equal(
    acceptsAttachmentUpload(file('a.png', 'image/png', policy.imageMaxBytes + 1), {total: 0, images: 0}, policy),
    false,
  );
  assert.equal(
    acceptsAttachmentUpload(file('a.png', 'image/png', 1), {total: 3, images: 3}, policy),
    false,
  );
  assert.equal(
    acceptsAttachmentUpload(
      file('a.png', 'image/png', 1),
      {total: 0, images: 0},
      {...policy, imageCapability: 'unsupported'},
    ),
    false,
  );
});

test('documents honor per-item bytes and supported extensions independent of image capability', () => {
  const noImages = {...policy, imageCapability: 'unsupported' as const, imageLimit: 0};
  assert.equal(
    acceptsAttachmentUpload(file('report.pdf', 'application/pdf', policy.documentMaxBytes), {total: 0, images: 0}, noImages),
    true,
  );
  assert.equal(
    acceptsAttachmentUpload(file('report.pdf', 'application/pdf', policy.documentMaxBytes + 1), {total: 0, images: 0}, noImages),
    false,
  );
  assert.equal(
    acceptsAttachmentUpload(file('archive.zip', 'application/zip', 1), {total: 0, images: 0}, policy),
    false,
  );
});

test('attachmentsEnabled reflects whether any attachment can be added', () => {
  assert.equal(attachmentsEnabled(policy), true);
  assert.equal(attachmentsEnabled({...policy, countLimit: 0}), false);
  assert.equal(
    attachmentsEnabled({...policy, imageCapability: 'unsupported', imageLimit: 0}),
    true,
  );
  assert.equal(
    attachmentsEnabled({
      ...policy,
      extensions: new Set<string>(),
      imageCapability: 'unsupported',
      imageLimit: 0,
    }),
    false,
  );
});

test('policy reads the unified server-projected data attributes', () => {
  const root = {
    dataset: {
      attachmentCountLimit: '6',
      attachmentImageMaxBytes: '15728640',
      attachmentDocumentMaxBytes: '104857600',
      attachmentExtensions: '["md","pdf"]',
      attachmentImageCapability: 'supported',
      attachmentImageLimit: '3',
    },
  } as unknown as HTMLElement;

  const parsed = getAttachmentPolicy(root);

  assert.equal(parsed?.countLimit, 6);
  assert.equal(parsed?.imageMaxBytes, 15728640);
  assert.equal(parsed?.documentMaxBytes, 104857600);
  assert.deepEqual(parsed?.extensions, new Set(['md', 'pdf']));
  assert.equal(parsed?.imageCapability, 'supported');
  assert.equal(parsed?.imageLimit, 3);
});

test('policy fails closed for missing, malformed, or unsafe attributes', () => {
  assert.equal(getAttachmentPolicy(null), null);

  const base = {
    attachmentCountLimit: '6',
    attachmentImageMaxBytes: '100',
    attachmentDocumentMaxBytes: '100',
    attachmentExtensions: '["pdf"]',
    attachmentImageCapability: 'supported',
    attachmentImageLimit: '3',
  };
  const root = (overrides: Record<string, string | undefined>): HTMLElement =>
    ({dataset: {...base, ...overrides}}) as unknown as HTMLElement;

  assert.equal(getAttachmentPolicy(root({attachmentCountLimit: undefined})), null);
  assert.equal(getAttachmentPolicy(root({attachmentCountLimit: 'six'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentCountLimit: '0'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentImageMaxBytes: '0'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentDocumentMaxBytes: '0'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentExtensions: '[]'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentExtensions: '["pdf",42]'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentImageCapability: 'bogus'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentImageLimit: '-1'})), null);
  assert.equal(getAttachmentPolicy(root({attachmentImageMaxBytes: '9007199254740992'})), null);
});

test('zero image limit stays valid but admits no images', () => {
  const zeroImages = getAttachmentPolicy(
    {
      dataset: {
        attachmentCountLimit: '6',
        attachmentImageMaxBytes: '100',
        attachmentDocumentMaxBytes: '100',
        attachmentExtensions: '["pdf"]',
        attachmentImageCapability: 'supported',
        attachmentImageLimit: '0',
      },
    } as unknown as HTMLElement,
  );

  assert.equal(zeroImages?.imageLimit, 0);
  assert.equal(
    acceptsAttachmentUpload(file('a.png', 'image/png', 1), {total: 0, images: 0}, zeroImages!),
    false,
  );
  assert.equal(
    acceptsAttachmentUpload(file('report.pdf', 'application/pdf', 1), {total: 0, images: 0}, zeroImages!),
    true,
  );
});

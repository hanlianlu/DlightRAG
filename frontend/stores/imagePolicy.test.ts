// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {test} from 'node:test';

import {
  acceptsImageUpload,
  canAddImage,
  getImageAdmissionPolicy,
  ImageReadAdmissionController,
} from '../ui/image_policy.ts';

const policy = {
  effectiveCurrentUploadLimit: 3,
  maxUploadBytes: 15 * 1024 * 1024,
  capabilityStatus: 'supported' as const,
};

test('browser image admission accepts the exact configured byte limit', () => {
  assert.equal(
    acceptsImageUpload(
      {type: 'image/png', size: policy.maxUploadBytes},
      0,
      policy,
    ),
    true,
  );
});

test('browser image admission rejects bytes or count above configured limits', () => {
  assert.equal(
    acceptsImageUpload(
      {type: 'image/png', size: policy.maxUploadBytes + 1},
      0,
      policy,
    ),
    false,
  );
  assert.equal(
    acceptsImageUpload({type: 'image/png', size: 1}, policy.effectiveCurrentUploadLimit, policy),
    false,
  );
});

test('canAddImage blocks non-supported capability at the single entry', () => {
  assert.equal(canAddImage({status: 'unsupported', limit: 3, current: 0}), false);
  assert.equal(canAddImage({status: 'unknown', limit: 3, current: 0}), false);
});

test('canAddImage allows supported uploads up to the effective limit', () => {
  assert.equal(canAddImage({status: 'supported', limit: 2, current: 1}), true);
  assert.equal(canAddImage({status: 'supported', limit: 2, current: 2}), false);
});

function policyRoot(dataset: Record<string, string>): HTMLElement {
  return {dataset} as unknown as HTMLElement;
}

test('SSR image policy fails closed for missing and malformed attributes', () => {
  assert.equal(getImageAdmissionPolicy(null), null);
  assert.equal(
    getImageAdmissionPolicy(policyRoot({effectiveCurrentUploadLimit: '3'})),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({effectiveCurrentUploadLimit: '3', maxUploadBytes: '100'}),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({maxUploadBytes: '100', answerImageCapability: 'supported'}),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: 'three',
        maxUploadBytes: '100',
        answerImageCapability: 'supported',
      }),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: '-1',
        maxUploadBytes: '100',
        answerImageCapability: 'supported',
      }),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: '3',
        maxUploadBytes: '0',
        answerImageCapability: 'supported',
      }),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: '9007199254740992',
        maxUploadBytes: '100',
        answerImageCapability: 'supported',
      }),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: '3',
        maxUploadBytes: '9007199254740992',
        answerImageCapability: 'supported',
      }),
    ),
    null,
  );
  assert.equal(
    getImageAdmissionPolicy(
      policyRoot({
        effectiveCurrentUploadLimit: '3',
        maxUploadBytes: '100',
        answerImageCapability: 'bogus',
      }),
    ),
    null,
  );
});

test('SSR zero-limit policy is valid but admits no images', () => {
  const zeroPolicy = getImageAdmissionPolicy(
    policyRoot({
      effectiveCurrentUploadLimit: '0',
      maxUploadBytes: '100',
      answerImageCapability: 'supported',
    }),
  );

  assert.deepEqual(zeroPolicy, {
    effectiveCurrentUploadLimit: 0,
    maxUploadBytes: 100,
    capabilityStatus: 'supported',
  });
  assert.equal(acceptsImageUpload({type: 'image/png', size: 1}, 0, zeroPolicy!), false);
});

class ControlledReader {
  onload: ((event: {target: {result: string | null} | null}) => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  abortCount = 0;

  readAsDataURL(_file: File): void {}

  succeed(dataUrl: string): void {
    this.onload?.({target: {result: dataUrl}});
  }

  fail(): void {
    this.onerror?.();
  }

  abort(): void {
    this.abortCount += 1;
    this.onabort?.();
  }
}

function imageFile(name: string): File {
  return {name, type: 'image/png', size: 10} as File;
}

test('file input, drop, and paste share synchronous in-flight count reservations', () => {
  const readers: ControlledReader[] = [];
  const committed: string[] = [];
  const controller = new ImageReadAdmissionController({
    getPolicy: () => policy,
    getCommittedCount: () => committed.length,
    onReady: (_file, dataUrl) => committed.push(dataUrl),
    createReader: () => {
      const reader = new ControlledReader();
      readers.push(reader);
      return reader as unknown as FileReader;
    },
  });

  const sources = ['file-input-1', 'file-input-2', 'drop', 'paste'];
  const admitted = sources.map((source) => controller.admit(imageFile(source)));

  assert.deepEqual(admitted, [true, true, true, false]);
  assert.equal(controller.inFlightCount, policy.effectiveCurrentUploadLimit);
  assert.equal(readers.length, policy.effectiveCurrentUploadLimit);

  readers[0].succeed('data:image/png;base64,one');
  assert.equal(controller.inFlightCount, 2);
  assert.equal(committed.length, 1);
  assert.equal(controller.admit(imageFile('paste-after-load')), false);

  readers[1].fail();
  assert.equal(controller.inFlightCount, 1);
  assert.equal(controller.admit(imageFile('drop-after-error')), true);
});

test('clearing in-flight reads releases slots and ignores late callbacks', () => {
  const readers: ControlledReader[] = [];
  const committed: string[] = [];
  const controller = new ImageReadAdmissionController({
    getPolicy: () => policy,
    getCommittedCount: () => committed.length,
    onReady: (_file, dataUrl) => committed.push(dataUrl),
    createReader: () => {
      const reader = new ControlledReader();
      readers.push(reader);
      return reader as unknown as FileReader;
    },
  });
  controller.admit(imageFile('drop'));
  controller.admit(imageFile('paste'));

  controller.clear();
  readers[0].succeed('data:image/png;base64,late');

  assert.equal(controller.inFlightCount, 0);
  assert.equal(readers[0].abortCount, 1);
  assert.equal(readers[1].abortCount, 1);
  assert.deepEqual(committed, []);
  assert.equal(controller.admit(imageFile('file-input-after-clear')), true);
});

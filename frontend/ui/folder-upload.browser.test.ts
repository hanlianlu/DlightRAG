// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {detectDropItems} from './folder-upload.ts';

/** What a browser hands over for a dropped file: an entry whose File may or may not be readable. */
function fileEntry(name: string, readable = true) {
  return {
    isFile: true as const,
    isDirectory: false as const,
    name,
    file(ok: (file: File) => void, fail?: (error: unknown) => void): void {
      if (readable) ok(new File(['x'], name, {type: 'text/markdown'}));
      else fail?.(new DOMException('gone', 'NotFoundError'));
    },
  };
}

/** A dropped folder whose reader answers each read with the next batch, or fails on `null`. */
function folderEntry(name: string, ...batches: unknown[][]) {
  const reads = [...batches];
  return {
    isFile: false as const,
    isDirectory: true as const,
    name,
    createReader: () => ({
      readEntries(ok: (entries: unknown[]) => void, fail?: (error: unknown) => void): void {
        const next = reads.shift();
        if (next === undefined) fail?.(new DOMException('failed', 'NotReadableError'));
        else ok(next);
      },
    }),
  };
}

function dropOf(...entries: unknown[]): DataTransferItemList {
  return entries.map((entry) => ({kind: 'file', webkitGetAsEntry: () => entry, getAsFile: () => null})) as unknown as DataTransferItemList;
}

it('keeps a dropped folder\'s structure: its name once, then its subfolders', async () => {
  const drop = dropOf(folderEntry('docs', [fileEntry('a.md'), folderEntry('sub', [fileEntry('b.md')], [])], []));

  const result = await detectDropItems(drop);

  expect(result.folderName).to.equal('docs');
  expect(result.files.map((file) => (file as {_relativePath?: string})._relativePath))
    .to.deep.equal(['docs/a.md', 'docs/sub/b.md']);
});

it('takes the files it can read from a drop, and finishes when one cannot be read', async () => {
  const drop = dropOf(fileEntry('a.md'), fileEntry('gone.md', false), fileEntry('c.md'));

  const result = await detectDropItems(drop);

  expect(result.files.map((file) => file.name)).to.deep.equal(['a.md', 'c.md']);
});

it('keeps what it read of a folder that has an unreadable file or stops answering', async () => {
  const drop = dropOf(folderEntry('docs', [fileEntry('a.md'), fileEntry('gone.md', false)]));

  const result = await detectDropItems(drop);

  expect(result.files.map((file) => (file as {_relativePath?: string})._relativePath)).to.deep.equal(['docs/a.md']);
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the localization catalogs.

 * Every statically declared `id:` on a msg() call in ui/ and lib/ must exist
 * in the zh catalog, so a renamed or newly added message cannot silently fall
 * back to English under a non-English locale. Catalogued tool verbs
 * (`chatFeature.tool.${name}` via toolDisplay) are locked the same way even
 * though their ids are interpolated. In the other direction every catalog
 * entry needs a declaring id (`id:` or `verbId:`, static or interpolated), and
 * each interpolated family must hold exactly the members its source map names,
 * so copy for removed UI cannot accumulate.
 */

import {readdirSync, readFileSync, statSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const FRONTEND_DIR = join(dirname(fileURLToPath(import.meta.url)), '..');

function sourceFiles(): string[] {
  const files: string[] = [];
  for (const relative of ['ui', 'lib']) {
    const dir = join(FRONTEND_DIR, relative);
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isFile() && name.endsWith('.ts') && !name.endsWith('.test.ts')) {
        files.push(path);
      }
    }
  }
  return files;
}

function declaredIds(): Set<string> {
  const ids = new Set<string>();
  for (const file of sourceFiles()) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(/\b(?:id|verbId):\s*'([A-Za-z][\w.]*)'/g)) {
      ids.add(match[1]);
    }
  }
  return ids;
}

/** Interpolated ids, as patterns where each `${...}` names one dotted segment. */
function declaredFamilies(): RegExp[] {
  const families: RegExp[] = [];
  for (const file of sourceFiles()) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(/\b(?:id|verbId):\s*`([A-Za-z][^`]*)`/g)) {
      const pattern = match[1]
        .split(/\$\{[^}]*\}/)
        .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('[^.]+');
      families.push(new RegExp(`^${pattern}$`));
    }
  }
  return families;
}

/** The keys of one source-file map literal, e.g. `const MODE_LABELS = {...}`. */
function mapKeys(relative: string, name: string, keyPattern = /^\s*([A-Za-z][\w]*):/gm): string[] {
  const source = readFileSync(join(FRONTEND_DIR, relative), 'utf8');
  const block = source.match(new RegExp(`${name}[^{]*\\{([\\s\\S]*?)\\n\\}`))?.[1];
  assert.ok(block, `${name} map missing in ${relative}`);
  return [...block.matchAll(keyPattern)].map((match) => match[1]);
}

/** The catalog entries of one interpolated family must be exactly its members. */
function assertClosedFamily(prefix: string, members: string[], suffix = ''): void {
  const actual = [...catalogKeys()]
    .filter((key) => key.startsWith(prefix) && key.endsWith(suffix))
    .filter((key) => !key.slice(prefix.length, key.length - suffix.length).includes('.'))
    .sort();
  assert.deepEqual(actual, members.map((member) => `${prefix}${member}${suffix}`).sort());
}

function catalogKeys(): Set<string> {
  const catalog = readFileSync(
    join(FRONTEND_DIR, 'i18n', 'locales', 'zh.ts'),
    'utf8',
  );
  return new Set(
    [...catalog.matchAll(/^\s*'([A-Za-z][\w.]*)':/gm)].map((match) => match[1]),
  );
}

test('every declared msg id exists in the zh catalog', () => {
  const missing = [...declaredIds()].filter((id) => !catalogKeys().has(id));
  assert.deepEqual(missing, []);
});

test('every zh catalog entry has a declaring msg id', () => {
  const declared = declaredIds();
  const families = declaredFamilies();
  const orphans = [...catalogKeys()].filter(
    (key) => !declared.has(key) && !families.some((family) => family.test(key)),
  );
  assert.deepEqual(orphans, []);
});

test('localized run error kinds and their zh entries match one to one', () => {
  const errors = readFileSync(join(FRONTEND_DIR, 'lib', 'run-errors.ts'), 'utf8');
  const block = errors.match(/RUN_ERROR_KIND_COPY[^{]+\{([\s\S]*?)\n\};/)?.[1];
  assert.ok(block, 'RUN_ERROR_KIND_COPY catalog missing');
  const kinds = [...block.matchAll(/^\s*(\w+):/gm)].map((match) => `errors.kind.${match[1]}`);
  const entries = [...catalogKeys()].filter((key) => key.startsWith('errors.kind.'));
  assert.deepEqual(entries.sort(), kinds.sort());
});

test('every offered agent effort has a zh label and aria entry', () => {
  const effort = readFileSync(join(FRONTEND_DIR, 'lib', 'agent-effort.ts'), 'utf8');
  const labels = effort.match(/EFFORT_LABELS[^{]+\{([^}]+)\}/)?.[1];
  assert.ok(labels, 'EFFORT_LABELS catalog missing');
  const levels = [...labels.matchAll(/^\s*([a-z]+):/gm)].map((match) => match[1]);
  assert.deepEqual(levels, ['low', 'high', 'max']);
  assertClosedFamily('chatComposer.effort.', levels);
  assertClosedFamily('chatComposer.effortAria.', levels);
});

test('mode, phase, and reconnect families hold exactly their mapped members', () => {
  const modes = mapKeys('ui/chat-composer.ts', 'const MODE_LABELS');
  assert.deepEqual(modes, ['auto', 'fast', 'research']);
  assertClosedFamily('chatComposer.mode.', modes);
  assertClosedFamily('chatComposer.modeAria.', modes);
  assertClosedFamily('chatFeature.phase.', mapKeys('lib/turn-projection.ts', 'ANSWER_PHASE_LABELS'));
  const states = mapKeys('ui/chat-message-list.ts', 'ANSWER_RECONNECT_COPY', /^ {2}(\w+): \{/gm);
  assert.deepEqual(states, ['running', 'stopping']);
  assertClosedFamily('chatMessageList.reconnect.', states, '.status');
  assertClosedFamily('chatMessageList.reconnect.', states, '.action');
});

test('every catalogued tool verb has a zh entry', () => {
  const display = readFileSync(join(FRONTEND_DIR, 'lib', 'tool-display.ts'), 'utf8');
  const block = display.match(/const TOOL_VERBS[^{]+\{([^}]+)\}/)?.[1];
  assert.ok(block, 'TOOL_VERBS catalog missing');
  const names = [...block.matchAll(/^\s*([a-z][a-z0-9_]*):/gm)].map((match) => match[1]);
  assert.ok(names.length > 0, 'TOOL_VERBS catalog is empty');
  // `mcp` is the static verbId for tools outside the catalogued set.
  assertClosedFamily('chatFeature.tool.', [...names, 'mcp']);
});

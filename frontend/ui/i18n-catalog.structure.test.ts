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
 *
 * The last two tests lock the one rule that keeps a language change honest: a view
 * holds machine facts, and the component applies the words. Copy the client owns
 * travels as `LocalizedText` (its English source plus the catalog id), so nothing
 * under lib/ resolves a language that a later switch would leave behind.
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
    // Three ways to declare copy the client owns: a msg() option, a ToolRow verb,
    // and the source-plus-id pair a view carries (`catalogText`).
    for (const match of source.matchAll(/\b(?:id|verbId):\s*'([A-Za-z][\w.]*)'/g)) {
      ids.add(match[1]);
    }
    for (const match of source.matchAll(/\bcatalogText\([\s\S]*?,\s*'([A-Za-z][\w.]*)',?\s*\)/g)) {
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
    const patterns: string[] = [];
    for (const match of source.matchAll(/\b(?:id|verbId):\s*`([A-Za-z][^`]*)`/g)) {
      patterns.push(match[1]);
    }
    for (const match of source.matchAll(/\bcatalogText\([\s\S]*?,\s*`([A-Za-z][^`]*)`/g)) {
      patterns.push(match[1]);
    }
    for (const raw of patterns) {
      const pattern = raw
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

test('the declaration-cost estimate keeps its one/many pair and its tools-less form', () => {
  // The estimate states a token cost, then the catalogue's tool count when there is one: one id
  // for the singular, one for the plural, and one for a cost whose tool count is unknown. The
  // closed family keeps a rename from silently leaving one of the three in English.
  assertClosedFamily('connections.declarationCost', ['Tokens', 'One', 'Many']);
});

test('the Inspector switcher, its Run overview band, and its step search declare their copy, and the zh catalog states it', () => {
  // The one-line switcher, the Run panel's band and states, and the step search that replaces the
  // panel while a reader looks for words: their copy is pinned by name, so a rename on one side of
  // the feature cannot leave the other reading in English.
  const expected = [
    'tracesPanel.switcherListAria',
    'tracesPanel.switcherEmpty',
    'tracesPanel.switcherNoMatch',
    'tracesPanel.filterAgents',
    'tracesPanel.asksParent',
    'tracesPanel.asksParentMany',
    'tracesPanel.childNumber',
    'tracesPanel.overview',
    'tracesPanel.overviewLoading',
    'tracesPanel.overviewFailed',
    'tracesPanel.overviewOneStep',
    'tracesPanel.overviewSteps',
    'tracesPanel.overviewOneAgent',
    'tracesPanel.overviewAgents',
    'tracesPanel.searchResults',
    'tracesPanel.searchLoading',
    'tracesPanel.searchFailed',
    'tracesPanel.searchNoMatch',
    'tracesPanel.searchOneMatch',
    'tracesPanel.searchMatches',
    'tracesPanel.searchLoadOlder',
    'tracesPanel.searchSaid',
    'tracesPanel.searchAnswered',
    'tracesPanel.searchInstruction',
  ];
  const declared = declaredIds();
  assert.deepEqual(expected.filter((id) => !declared.has(id)), [], 'the Inspector declares its copy');
  assert.deepEqual(expected.filter((id) => !catalogKeys().has(id)), [], 'and the zh catalog states it');
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

test('every reason a thumb down is given has a label, and a zh entry', () => {
  const wire = readFileSync(join(FRONTEND_DIR, 'api', 'feedback.ts'), 'utf8');
  const block = wire.match(/FEEDBACK_REASONS = \[([^\]]+)\]/)?.[1];
  assert.ok(block, 'FEEDBACK_REASONS missing');
  const reasons = [...block.matchAll(/'([a-z_]+)'/g)].map((match) => match[1]);
  assert.equal(reasons.length, 5);
  // The labels are offered in the order the server keeps the reasons.
  assert.deepEqual(mapKeys('ui/answer-feedback.ts', 'const REASON_LABELS'), reasons);
  assertClosedFamily('answerFeedback.reason.', reasons);
});

test('every reason an Artifact attestation can give has a zh entry', () => {
  const badge = readFileSync(join(FRONTEND_DIR, 'ui', 'artifact-attestation.ts'), 'utf8');
  const block = badge.match(/const REASON_TEXT[^{]+\{([\s\S]*?)\n\};/)?.[1];
  assert.ok(block, 'REASON_TEXT catalog missing');
  const reasons = [...block.matchAll(/^\s*([a-z][a-z0-9_]*):/gm)].map((match) => match[1]);
  // The server's closed vocabulary, in the order `provenance.py` publishes it.
  assert.deepEqual(reasons, [
    'no_skill_loaded',
    'skill_loaded_without_verifier',
    'skill_has_no_verifier_contract',
    'contract_unavailable',
    'verifier_reported_failures',
    'records_pruned',
    'skill_name_unavailable',
    'arguments_truncated',
  ]);
  assertClosedFamily('artifactAttestation.reason.', reasons);
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

test('the turn view holds machine facts, never resolved copy', () => {
  const source = readFileSync(join(FRONTEND_DIR, 'lib', 'chat-views.ts'), 'utf8');
  const body = source.match(/export interface ChatTurnView \{([\s\S]*?)\n\}/)?.[1];
  assert.ok(body, 'ChatTurnView missing');
  // The only plain strings are words the server wrote or an identifier.
  const strings = [...body.matchAll(/^\s*(\w+):\s*string;$/gm)].map((match) => match[1]).sort();
  assert.deepEqual(strings, ['id', 'runId', 'streamText', 'userText']);
  // What the client owns is carried as its source, for a component to resolve.
  assert.match(body, /error: LocalizedText \| null;/);
  assert.match(body, /activity: TurnActivity \| null;/);
});

test('no view builder resolves words', () => {
  for (const relative of ['lib/chat-views.ts', 'lib/turn-projection.ts', 'lib/tool-events.ts']) {
    const source = readFileSync(join(FRONTEND_DIR, relative), 'utf8');
    assert.ok(
      !source.includes("from '@lit/localize'"),
      `${relative} must leave the words to the component that draws them`,
    );
  }
});

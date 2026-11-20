// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the Web's Agentspace/Workspace split.
 *
 * ADR 0075 gave the product two words: Workspace is the corpus scope, Agentspace is the agent's own
 * file space. The wire keeps its snake_case `workspace` paths and fields (ADR 0076), so only the
 * TypeScript language layer is renamed; this test keeps each vocabulary inside the modules it owns.
 * The names are read from the source text with line-anchored regexes instead of an AST: a declaration
 * is the form a name takes here, which makes the scan deterministic and needs no parser dependency.
 */

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';

const FRONTEND = dirname(fileURLToPath(import.meta.url));

/** The modules that speak for the agent's own file space: they say Agentspace and never Workspace. */
const AGENTSPACE_MODULES = ['api/conversations.ts', 'ui/inspector-agentspace.ts'];

/** The modules that speak for the corpus scope: they say Workspace and never Agentspace. */
const CORPUS_MODULES = [
  'api/workspaces.ts',
  'stores/workspace-store.ts',
  'ui/workspace-scope.ts',
  'ui/workspace-create.ts',
];

/** A declaration at the start of a source line, exported or module-local: the TypeScript keyword
 *  (`type`, `interface`, `const`, `let`, `function`, `class`, or a `declare`d/`async` variant)
 *  followed by the name it binds. */
const DECLARATION = /^[ \t]*(?:export\s+)?(?:declare\s+)?(?:async\s+)?(?:type|interface|const|let|function|class)\s+([A-Za-z_$][\w$]*)/gm;

/** The same shape, exported only: a name one wire module may declare alone. */
const EXPORTED = /^[ \t]*export\s+(?:declare\s+)?(?:async\s+)?(?:type|interface|const|let|function|class)\s+([A-Za-z_$][\w$]*)/gm;

function namesIn(source: string, pattern: RegExp): string[] {
  return [...source.matchAll(pattern)].map((match) => match[1]!);
}

function read(relativePath: string): string {
  return readFileSync(join(FRONTEND, relativePath), 'utf8');
}

test('the agentspace modules declare no name beginning with Workspace', () => {
  const offenders = AGENTSPACE_MODULES.flatMap((module) => namesIn(read(module), DECLARATION)
    .filter((name) => name.startsWith('Workspace'))
    .map((name) => `${module}: ${name}`));
  assert.deepEqual(offenders, []);
});

test('the corpus modules declare no name beginning with Agentspace', () => {
  const offenders = CORPUS_MODULES.flatMap((module) => namesIn(read(module), DECLARATION)
    .filter((name) => name.startsWith('Agentspace'))
    .map((name) => `${module}: ${name}`));
  assert.deepEqual(offenders, []);
});

/** Every wire module under `api/`, less the tests that read them. */
function apiModules(): string[] {
  return readdirSync(join(FRONTEND, 'api'))
    .filter((name) => name.endsWith('.ts') && !name.endsWith('.test.ts') && !name.endsWith('.d.ts'))
    .sort();
}

test('one exported name belongs to one module under api/', () => {
  const owners = new Map<string, string[]>();
  for (const module of apiModules()) {
    for (const name of namesIn(read(join('api', module)), EXPORTED)) {
      owners.set(name, [...(owners.get(name) ?? []), module]);
    }
  }
  const duplicates = [...owners]
    .filter(([, modules]) => modules.length > 1)
    .map(([name, modules]) => `${name}: ${modules.join(', ')}`);
  assert.deepEqual(duplicates, [], 'one name has two meanings on the wire; give each its own');
  assert.ok(owners.size > 100, 'the wire modules were not found');
});

test('the scan reads the declarations it is meant to judge', () => {
  // A check that can never fail is not a check: the scan must see the declaration forms this
  // codebase names its wire types with, including the exported `v.InferOutput` type lines.
  const sample = [
    'export type WorkspaceChange = (typeof CHANGES)[number];',
    'const agentspaceHead = v.pipe(v.object({}));',
    'export type AgentspaceSnapshot = v.InferOutput<typeof agentspaceSnapshot>;',
    'export async function readWorkspaceSnapshot(): Promise<void> {}',
  ].join('\n');
  assert.deepEqual(namesIn(sample, DECLARATION), [
    'WorkspaceChange',
    'agentspaceHead',
    'AgentspaceSnapshot',
    'readWorkspaceSnapshot',
  ]);
  assert.deepEqual(namesIn(sample, EXPORTED), [
    'WorkspaceChange',
    'AgentspaceSnapshot',
    'readWorkspaceSnapshot',
  ]);
});

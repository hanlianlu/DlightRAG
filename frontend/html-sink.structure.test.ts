// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Structural lock on the one place model-authored HTML becomes part of the page.
 *
 * Answers, sources, and previews carry HTML that a model or a document wrote. `lib/safe-html.ts` is the only
 * module that turns such a string into nodes, after DOMPurify has judged it, and `ui/rich-rendering.ts` is
 * the only one that calls it, in the same stages for every surface: sanitize, harden the links, typeset math,
 * draw diagrams. A second place that writes HTML is a second place to get that wrong; a surface that called
 * a stage on its own would skip the others.
 */

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';

const FRONTEND = dirname(fileURLToPath(import.meta.url));

/** Where the sink lives, and the one module that may call it. */
const SINK = 'lib/safe-html.ts';
const PIPELINE = 'ui/rich-rendering.ts';

/** Directories that hold no production module: the packages, the support code of tests, and build tooling. */
const NOT_PRODUCTION = new Set(['node_modules', 'testing', 'scripts', 'tokens']);

function sourcesUnder(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return NOT_PRODUCTION.has(entry.name) ? [] : sourcesUnder(path);
    const production = entry.name.endsWith('.ts')
      && !entry.name.endsWith('.test.ts')
      && !entry.name.endsWith('.d.ts')
      && entry.name !== 'vite.config.ts';
    return production ? [path] : [];
  });
}

/** Every production module with its comments taken out, so a sentence about a sink is not a use of one. */
const sources = new Map(sourcesUnder(FRONTEND).map((path) => [
  relative(FRONTEND, path),
  readFileSync(path, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1'),
]));

/** A template literal with nothing interpolated: markup written in this repository, the same every time. */
const CONSTANT_TEMPLATE = /`(?:[^`\\$]|\\[\s\S]|\$(?!\{))*`/y;

/** The ways to hand a string to the parser other than the sink. Assigning `innerHTML` is allowed for
 *  markup that is written here (a primitive's template), and no other way is. */
const OTHER_SINKS = /\b(?:insertAdjacentHTML|setHTMLUnsafe|createContextualFragment)\s*\(|\bunsafe(?:HTML|SVG)\b|\bdocument\.write(?:ln)?\s*\(/;

test('only lib/safe-html.ts turns a string of unknown markup into nodes', () => {
  const offenders: string[] = [];
  for (const [path, source] of sources) {
    if (path === SINK) continue;
    if (OTHER_SINKS.test(source)) offenders.push(`${path}: parses a string by a way other than the sink`);
    for (const assignment of source.matchAll(/\.(?:inner|outer)HTML\s*\+?=(?!=)\s*/g)) {
      CONSTANT_TEMPLATE.lastIndex = (assignment.index ?? 0) + assignment[0].length;
      if (!CONSTANT_TEMPLATE.test(source)) offenders.push(`${path}: assigns ${assignment[0].trim()} a value that is not constant`);
    }
  }
  assert.deepEqual(offenders, []);

  // A check that never fails is not a check: the scan sees the sink, and the constant templates it allows.
  assert.match(sources.get(SINK) ?? '', /\.innerHTML\s*=/);
  const primitive = sources.get('design-system/elements/menu.ts') ?? '';
  assert.match(primitive, /\.innerHTML\s*=\s*`/);
});

/** What only the pipeline calls, and the module that defines it. */
const PIPELINE_ONLY: ReadonlyArray<{name: string; home: string}> = [
  {name: 'setSanitizedLlmHtml', home: SINK},
  {name: 'renderMath', home: 'lib/math.ts'},
  {name: 'renderDiagrams', home: 'ui/mermaid.ts'},
];

test('the stages of the rich-content pipeline are used by the pipeline alone', () => {
  const offenders: string[] = [];
  for (const {name, home} of PIPELINE_ONLY) {
    const mentions = [...sources].filter(([, source]) => new RegExp(`\\b${name}\\b`).test(source)).map(([path]) => path);
    assert.ok(mentions.includes(PIPELINE), `${PIPELINE} no longer runs ${name}`);
    for (const path of mentions) if (path !== home && path !== PIPELINE) offenders.push(`${path}: ${name}`);
  }
  assert.deepEqual(offenders, [], 'a surface that shows rich content goes through mountRichHtml and typesetRichContent');
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Generate a deterministic DTCG-style manifest from runtime-authoritative CSS. */

import {readFile, readdir, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const foundations = resolve(frontend, 'design-system/foundations');
const outputPath = resolve(foundations, 'tokens.generated.json');
const check = process.argv.includes('--check');
const files = (await readdir(foundations))
  .filter((name) => name.endsWith('.css') && name !== 'index.css')
  .sort();

function declarations(source) {
  return [...source.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)]
    .map((match) => [match[1].slice(2), match[2].trim()]);
}

function tokenType(name, value) {
  if (name.startsWith('color-') || /^#|^rgb|^hsl/.test(value)) return 'color';
  if (name.startsWith('shadow-')) return 'shadow';
  if (name.startsWith('duration-')) return 'duration';
  if (/^(radius|size|space|layout|inset|font-size|step)-/.test(name)) return 'dimension';
  if (name.startsWith('font-')) return 'fontFamily';
  if (name.startsWith('easing-')) return 'cubicBezier';
  return 'string';
}

function dtcgValue(value) {
  const reference = value.match(/^var\(--([a-z0-9-]+)\)$/);
  return reference ? `{${reference[1]}}` : value;
}

const defaults = new Map();
const light = new Map();
for (const file of files) {
  const source = await readFile(resolve(foundations, file), 'utf8');
  const marker = ":root[data-color-mode='light']";
  const markerIndex = source.indexOf(marker);
  const defaultSource = markerIndex >= 0 ? source.slice(0, markerIndex) : source;
  const lightSource = markerIndex >= 0 ? source.slice(markerIndex) : '';
  for (const [name, value] of declarations(defaultSource)) defaults.set(name, value);
  for (const [name, value] of declarations(lightSource)) light.set(name, value);
}

const manifest = {
  $schema: 'https://design-tokens.github.io/community-group/format/',
  $description: 'Generated from DlightRAG runtime CSS. Do not edit by hand.',
};
for (const name of [...defaults.keys()].sort()) {
  const value = defaults.get(name);
  const token = {
    $type: tokenType(name, value),
    $value: dtcgValue(value),
  };
  if (light.has(name) && light.get(name) !== value) {
    token.$extensions = {
      'org.dlightrag.modes': {light: dtcgValue(light.get(name))},
    };
  }
  manifest[name] = token;
}
const output = `${JSON.stringify(manifest, null, 2)}\n`;

if (check) {
  const current = await readFile(outputPath, 'utf8').catch(() => '');
  if (current !== output) {
    throw new Error('Generated token manifest is stale; run npm run generate:tokens');
  }
} else {
  await writeFile(outputPath, output);
}

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Deterministically project the approved icon selection into a tiny runtime registry. */

import {readFile, writeFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const frontend = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const selectionPath = resolve(frontend, 'design-system/icons/selection.json');
const sourcesPath = resolve(frontend, 'design-system/icons/sources.json');
const nodesPath = resolve(frontend, 'node_modules/lucide-static/icon-nodes.json');
const packagePath = resolve(frontend, 'node_modules/lucide-static/package.json');
const outputPath = resolve(frontend, 'design-system/icons/registry.generated.ts');
const check = process.argv.includes('--check');

const selection = JSON.parse(await readFile(selectionPath, 'utf8'));
const sources = JSON.parse(await readFile(sourcesPath, 'utf8'));
const lucideNodes = JSON.parse(await readFile(nodesPath, 'utf8'));
const lucidePackage = JSON.parse(await readFile(packagePath, 'utf8'));
if (sources.lucide.version !== lucidePackage.version) {
  throw new Error(`Lucide source manifest expects ${sources.lucide.version}, installed ${lucidePackage.version}`);
}
const entries = {};
for (const semanticName of Object.keys(selection).sort()) {
  const selected = selection[semanticName];
  if (!sources[selected.source]) throw new Error(`Missing source metadata for ${selected.source}`);
  // A source's own glyph is the package's drawing, and the package's own paint goes with it. Geometry or
  // paint authored here is this product's, however much of the source's shape it keeps, and an entry that
  // claims otherwise is mislabelled — which is how two thumbs were poured full under a lucide label.
  if (selected.source === 'lucide' && selected.nodes !== undefined) {
    throw new Error(`${semanticName} carries authored geometry under a lucide label; declare it dlightrag`);
  }
  if (selected.source === 'lucide' && (selected.fill !== undefined || selected.stroke !== undefined)) {
    throw new Error(`${semanticName} carries authored paint under a lucide label; declare it dlightrag`);
  }
  const derivedFrom = selected.derivedFrom ?? null;
  if (derivedFrom !== null) {
    const parts = derivedFrom.split('/');
    if (parts.length !== 2 || !sources[parts[0]] || parts[1] === '') {
      throw new Error(`Malformed derivedFrom for ${semanticName}: ${derivedFrom}`);
    }
  }
  const nodes = selected.nodes
    ?? (selected.source === 'lucide' ? lucideNodes[selected.name] : undefined);
  if (!Array.isArray(nodes)) {
    throw new Error(`Missing icon geometry for ${semanticName}: ${selected.source}/${selected.name}`);
  }
  entries[semanticName] = {
    source: selected.source,
    sourceName: selected.name,
    derivedFrom,
    viewBox: selected.viewBox ?? '0 0 24 24',
    opticalScale: selected.opticalScale ?? 1,
    opticalX: selected.opticalX ?? 0,
    opticalY: selected.opticalY ?? 0,
    fill: selected.fill ?? 'none',
    stroke: selected.stroke ?? 'currentColor',
    nodes,
  };
}

// A derived entry names a glyph that exists — in the package it names, or in this selection — and it still
// draws that glyph's principal shape. So the provenance a catalog prints is a fact about this file rather
// than a string that happens to look like one, and a ladder cannot quietly become unrelated glyphs.
const pathsOf = (nodes) => nodes
  .filter(([tag]) => tag === 'path')
  .map(([, attributes]) => String(attributes.d ?? ''))
  .sort((left, right) => right.length - left.length);
// The shape a base is known by: its longest path, or, for a glyph drawn without paths, its first node.
const principalOf = (nodes) => {
  const [path] = pathsOf(nodes);
  return path !== undefined ? {path} : {node: nodes[0]};
};
const draws = (nodes, principal) => {
  if (principal.path !== undefined) return pathsOf(nodes).some((d) => d.includes(principal.path));
  const [tag, attributes] = principal.node ?? [];
  const wanted = JSON.stringify(Object.entries(attributes ?? {}).sort());
  return nodes.some(([eachTag, each]) => eachTag === tag
    && JSON.stringify(Object.entries(each).sort()) === wanted);
};
for (const [semanticName, entry] of Object.entries(entries)) {
  if (entry.derivedFrom === null) continue;
  const [source, name] = entry.derivedFrom.split('/');
  const base = source === 'lucide'
    ? lucideNodes[name]
    : Object.values(entries).find((each) => each.source === source && each.sourceName === name)?.nodes;
  if (!Array.isArray(base)) {
    throw new Error(`${semanticName} derives from ${entry.derivedFrom}, which the source does not hold`);
  }
  if (!draws(entry.nodes, principalOf(base))) {
    throw new Error(`${semanticName} no longer draws the ${entry.derivedFrom} shape it derives from`);
  }
}

const serialized = JSON.stringify(entries, null, 2)
  .replaceAll('"source":', 'source:')
  .replaceAll('"sourceName":', 'sourceName:')
  .replaceAll('"derivedFrom":', 'derivedFrom:')
  .replaceAll('"viewBox":', 'viewBox:')
  .replaceAll('"opticalScale":', 'opticalScale:')
  .replaceAll('"opticalX":', 'opticalX:')
  .replaceAll('"opticalY":', 'opticalY:')
  .replaceAll('"fill":', 'fill:')
  .replaceAll('"stroke":', 'stroke:')
  .replaceAll('"nodes":', 'nodes:');
const output = `// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** GENERATED by scripts/generate-icons.mjs from lucide-static ${lucidePackage.version}. */

type IconSource = 'lucide' | 'dlightrag';

export type IconNode = readonly [tag: string, attributes: Readonly<Record<string, string>>];
export interface IconDefinition {
  readonly source: IconSource;
  readonly sourceName: string;
  /** The source and name of the glyph whose shape this entry keeps, or null when it ships its own. */
  readonly derivedFrom: string | null;
  readonly viewBox: string;
  readonly opticalScale: number;
  readonly opticalX: number;
  readonly opticalY: number;
  readonly fill: string;
  readonly stroke: string;
  readonly nodes: readonly IconNode[];
}

export const ICON_REGISTRY = ${serialized} as const satisfies Record<string, IconDefinition>;
export type IconName = keyof typeof ICON_REGISTRY;
`;

if (check) {
  const current = await readFile(outputPath, 'utf8').catch(() => '');
  if (current !== output) {
    throw new Error('Generated icon registry is stale; run npm run generate:icons');
  }
} else {
  await writeFile(outputPath, output);
}

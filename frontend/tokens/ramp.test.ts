// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const css = readFileSync(
    fileURLToPath(new URL('../design-system/foundations/color.css', import.meta.url)),
    'utf8',
);
const semantics = css.slice(css.indexOf('  --color-text-primary'));

/** Declared ramp steps, keyed by hex: #7e6c37 -> gold-400. */
const ramps = new Map(
    [...css.matchAll(/--color-((?:stone|gold|red)-\d+):\s*(#[0-9a-f]{6});/g)]
        .map((match) => [match[2], match[1]]),
);

/** The light overrides live in their own block; everything before it is dark. */
function themeBlock(theme: 'dark' | 'light'): string {
    const start = semantics.indexOf(":root[data-color-mode='light']");
    assert.ok(start > 0, 'light theme block not found');
    return theme === 'dark' ? semantics.slice(0, start) : semantics.slice(start);
}

function surfaceLuminance(theme: 'dark' | 'light', name: string): {step: string; value: number} {
    const reference = new RegExp(`--${name}:\\s*var\\(--color-([\\w-]+)\\)`).exec(themeBlock(theme));
    assert.ok(reference, `--${name} is not a ramp reference in the ${theme} block`);
    const step = reference[1];
    const hex = [...ramps].find(([, declared]) => declared === step)?.[0];
    assert.ok(hex, `${step} is not a declared ramp step`);
    const linear = [1, 3, 5]
        .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
        .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return {step, value: 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]};
}

// Elevation steps away from the reading canvas: lighter in dark, darker in
// light. Nothing in a step name shows this, so it has to be checked here -- the
// light ramp had silently inverted, putting the panel above the conversation.
for (const [theme, direction] of [['dark', 1], ['light', -1]] as const) {
    test(`${theme} surfaces step away from the canvas`, () => {
        const surfaces = ['color-bg-base', 'color-bg-surface', 'color-bg-elevated']
            .map((name) => surfaceLuminance(theme, name));

        for (let i = 1; i < surfaces.length; i += 1) {
            assert.ok(
                (surfaces[i].value - surfaces[i - 1].value) * direction > 0,
                `${surfaces[i].step} must sit further from the canvas than ${surfaces[i - 1].step}`,
            );
        }
    });
}

// Every drift here arrived as a literal nobody could place by eye: an inverted
// surface ramp, an accent a step off the gold scale, borders a few units off
// stone. Only white belongs to no ramp and stays spelled out.
test('colours outside the ramps are only ever white', () => {
    const triples = new Set(
        [...ramps.keys()].map((hex) => [1, 3, 5]
            .map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ')),
    );
    const strays = [
        ...[...semantics.matchAll(/#[0-9a-fA-F]{6}\b/g)].map((m) => m[0].toLowerCase()),
        ...[...semantics.matchAll(/rgba?\((\d+, \d+, \d+)/g)]
            .map((m) => m[1])
            .filter((triple) => !triples.has(triple))
            .map((triple) => `rgb(${triple})`),
    ].filter((colour) => !['#ffffff', 'rgb(255, 255, 255)', 'rgb(0, 0, 0)'].includes(colour));

    assert.deepEqual([...new Set(strays)], []);
});

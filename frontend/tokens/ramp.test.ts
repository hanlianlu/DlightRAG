// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';

const frontend = fileURLToPath(new URL('..', import.meta.url));
const css = readFileSync(
    fileURLToPath(new URL('../design-system/foundations/color.css', import.meta.url)),
    'utf8',
);
const semantics = css.slice(css.indexOf('  --color-text-primary'));

/** Declared ramp steps, keyed by name: gold-400 -> #7e6c37. Keyed by name, not
 * by value: a measured ladder step may legitimately land on a ramp value, and the
 * dark ladder's ink-80 is the light ladder's stone-600. */
const ramps = new Map(
    [...css.matchAll(/--color-([a-z]+-\d+):\s*(#[0-9a-f]{6});/g)]
        .map((match) => [match[1], match[2]]),
);

/** The light overrides live in their own block; everything before it is dark. */
function themeBlock(theme: 'dark' | 'light'): string {
    const start = semantics.indexOf(":root[data-color-mode='light']");
    assert.ok(start > 0, 'light theme block not found');
    return theme === 'dark' ? semantics.slice(0, start) : semantics.slice(start);
}

function stepLuminance(theme: 'dark' | 'light', name: string): {step: string; value: number} {
    const reference = new RegExp(`--${name}:\\s*var\\(--color-([\\w-]+)\\)`).exec(themeBlock(theme));
    assert.ok(reference, `--${name} is not a ramp reference in the ${theme} block`);
    const step = reference[1];
    const hex = ramps.get(step);
    assert.ok(hex, `${step} is not a declared ramp step`);
    const linear = [1, 3, 5]
        .map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
        .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return {step, value: 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]};
}

function contrast(first: number, second: number): number {
    const [lighter, darker] = [first, second].sort((a, b) => b - a);
    return (lighter + 0.05) / (darker + 0.05);
}

const surfaceRoles = ['color-bg-base', 'color-bg-surface', 'color-bg-elevated'];

// Elevation steps away from the reading canvas: lighter in dark, darker in
// light. Nothing in a step name shows this, so it has to be checked here -- the
// light ramp had silently inverted, putting the panel above the conversation.
for (const [theme, direction] of [['dark', 1], ['light', -1]] as const) {
    test(`${theme} surfaces step away from the canvas`, () => {
        const surfaces = surfaceRoles.map((name) => stepLuminance(theme, name));

        for (let i = 1; i < surfaces.length; i += 1) {
            assert.ok(
                (surfaces[i].value - surfaces[i - 1].value) * direction > 0,
                `${surfaces[i].step} must sit further from the canvas than ${surfaces[i - 1].step}`,
            );
        }
    });
}

// Muted is the faintest role that clears AA for small text on every surface, so
// a caption that takes it reads anywhere. Subtle, one step below, is held only to
// the non-text floor. Accent text is the gold for text, so it takes the text floor
// too. The Primary accent marks things, such as a link's underline or a check, and
// like Subtle is held to the non-text floor.
const contrastFloors = [
    ['color-text-primary', 4.5],
    ['color-text-secondary', 4.5],
    ['color-text-tertiary', 4.5],
    ['color-text-muted', 4.5],
    ['color-accent-text', 4.5],
    ['color-text-subtle', 3],
    ['color-accent-action', 3],
] as const;

for (const theme of ['dark', 'light'] as const) {
    test(`${theme} text roles clear their contrast floor on every surface`, () => {
        const surfaces = surfaceRoles.map((name) => stepLuminance(theme, name));

        for (const [role, floor] of contrastFloors) {
            const text = stepLuminance(theme, role);
            for (const surface of surfaces) {
                assert.ok(
                    contrast(text.value, surface.value) >= floor,
                    `--${role} (${text.step}) is under ${floor}:1 on ${surface.step}`,
                );
            }
        }
    });
}

// Dim, one step below Subtle, clears neither floor, and WCAG waives contrast
// only for an inactive control, so a disabled control is all it may colour.
// Placeholders, captions, chevrons, and the switch thumb had drifted onto it
// and read at 1.69:1 on an elevated surface.
test('only a disabled control is coloured Dim', () => {
    const sources = ['styles', 'design-system'].flatMap((directory) =>
        readdirSync(join(frontend, directory), {recursive: true, encoding: 'utf8'})
            .filter((path) => /\.(css|ts)$/.test(path) && !path.endsWith('.test.ts'))
            .map((path) => join(directory, path)));

    for (const path of sources) {
        const source = readFileSync(join(frontend, path), 'utf8');
        // A design-system element carries its Shadow CSS in its template's <style>.
        const rules = (path.endsWith('.ts')
            ? [...source.matchAll(/<style>([\s\S]*?)<\/style>/g)].map((match) => match[1]).join('\n')
            : source
        ).replace(/\/\*[\s\S]*?\*\//g, '');
        for (const [, selectors, declarations] of rules.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
            if (!/var\(--color-text-dim\b(?!-)/.test(declarations)) continue;
            for (const selector of selectors.split(',')) {
                assert.match(selector, /:disabled|\[disabled\]|\[aria-disabled='true'\]/,
                    `${selector.trim()} in ${path} is not a disabled control`);
            }
        }
    }
});

// Every drift here arrived as a literal nobody could place by eye: an inverted
// surface ramp, an accent a step off the gold scale, borders a few units off
// stone. Only white belongs to no ramp and stays spelled out.
test('colours outside the ramps are only ever white', () => {
    const triples = new Set(
        [...ramps.values()].map((hex) => [1, 3, 5]
            .map((i) => parseInt(hex.slice(i, i + 2), 16)).join(', ')),
    );
    const strays = [
        ...[...semantics.matchAll(/#[0-9a-fA-F]{3,8}\b/g)].map((m) => m[0].toLowerCase()),
        ...[...semantics.matchAll(/rgba?\((\d+)[ ,]+(\d+)[ ,]+(\d+)/g)]
            .map((m) => m.slice(1, 4).join(', '))
            .filter((triple) => !triples.has(triple))
            .map((triple) => `rgb(${triple})`),
    ].filter((colour) => !['#fff', '#ffffff', 'rgb(255, 255, 255)', 'rgb(0, 0, 0)'].includes(colour));

    assert.deepEqual([...new Set(strays)], []);
});

// APCA-W3 (SAPC): the polarity-aware contrast model WCAG 3 is moving to, and the
// one that can say "too much" as well as "too little" — WCAG 2's ratio is
// symmetric, so it reads white-on-black and black-on-white the same and sees
// nothing wrong with either at 18:1. Constants are the published ones.
function luminanceOf(hex: string): number {
    const channel = (value: number): number => {
        const c = value / 255;
        return c >= 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92;
    };
    const [r, g, b] = [1, 3, 5].map((i) => channel(parseInt(hex.slice(i, i + 2), 16)));
    return 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
}

function apca(text: string, background: string): number {
    // biome-ignore lint/suspicious/noApproximativeNumericConstant: 1.414 is APCA-W3's published blkClmp exponent, not the square root of two
    const clamp = (y: number): number => (y < 0.022 ? y + (0.022 - y) ** 1.414 : y);
    const yText = clamp(luminanceOf(text));
    const yBackground = clamp(luminanceOf(background));
    if (yBackground > yText) {
        const sapc = (yBackground ** 0.56 - yText ** 0.57) * 1.14;
        return sapc < 0.1 ? 0 : (sapc - 0.027) * 100;
    }
    const sapc = (yBackground ** 0.65 - yText ** 0.62) * 1.14;
    return sapc > -0.1 ? 0 : (sapc + 0.027) * 100;
}

/** `--color-<name>` as the hex it resolves to in a theme block. */
function resolved(theme: 'dark' | 'light', name: string): string {
    const reference = new RegExp(`--${name}:\\s*var\\(--color-([\\w-]+)\\)`).exec(themeBlock(theme));
    assert.ok(reference, `--${name} is not a reference in the ${theme} block`);
    const hex = ramps.get(reference[1]);
    assert.ok(hex, `${reference[1]} is not a declared step`);
    return hex;
}

const darkBase = resolved('dark', 'color-bg-base');

// Each theme's ladder names are claims: ink-75 is the step that measures |Lc| 75
// on that theme's page. The two ladders differ because the polarities do — the
// dark one is capped at 90, the light one is not — but the rule, and this test,
// are the same in both.
const readingLadders = {
    dark: [
        ['color-text-primary', 90],
        ['color-text-secondary', 82],
        ['color-text-tertiary', 75],
        ['color-text-muted', 60],
    ],
    light: [
        ['color-text-primary', 100],
        ['color-text-secondary', 92],
        ['color-text-tertiary', 87],
        ['color-text-muted', 80],
    ],
} as const;

for (const theme of ['dark', 'light'] as const) {
    test(`the ${theme} reading ladder measures to the level each name states`, () => {
        const page = resolved(theme, 'color-bg-base');
        for (const [role, level] of readingLadders[theme]) {
            const reference = new RegExp(`--${role}:\\s*var\\(--color-ink-(\\d+)\\)`).exec(
                themeBlock(theme),
            );
            assert.ok(reference, `--${role} is not on the ink ladder in ${theme}`);
            assert.equal(
                Number(reference[1]),
                level,
                `--${role} names a level it does not target`,
            );
            const measured = Math.abs(apca(resolved(theme, role), page));
            assert.ok(
                Math.abs(measured - level) <= 2,
                `--${role} (ink-${level}) measures |Lc| ${measured.toFixed(1)} on the page`,
            );
        }
    });
}

// Lc 90 is the published maximum for light-on-dark text: past it the luminance
// step at a glyph's edge is what reads as glare, which is exactly the complaint a
// brighter value invites. Nothing in the dark reading ladder may go above it, the
// roles must stay in rank order, and the quietest one must still be readable.
test('dark text stays under the dark-mode ceiling and in rank order', () => {
    const levels = readingLadders.dark.map(([role]) =>
        Math.abs(apca(resolved('dark', role), darkBase)),
    );
    assert.ok(levels[0] <= 90, `the brightest reading role is at |Lc| ${levels[0].toFixed(1)}`);
    assert.deepEqual(
        [...levels].sort((first, second) => second - first),
        levels,
        'the ladder is out of rank order',
    );
    assert.ok(levels[3] >= 60, `the quietest reading role is at |Lc| ${levels[3].toFixed(1)}`);
});

// The light theme has no ceiling, so what it owes instead is a surface ramp tight
// enough to keep the quiet end readable on the brightest panel: stone-50/200/300
// cost 21 |Lc| between the page and an elevated card and left Muted at 58.8, under
// the 60 a caption needs. The surfaces now sit two steps apart, and this is the
// assertion that holds them there.
test('light text keeps its rank and stays readable on the brightest panel', () => {
    const levels = readingLadders.light.map(([role]) =>
        Math.abs(apca(resolved('light', role), resolved('light', 'color-bg-elevated'))),
    );
    assert.deepEqual(
        [...levels].sort((first, second) => second - first),
        levels,
        'the ladder is out of rank order on an elevated surface',
    );
    assert.ok(levels[3] >= 60, `the quietest reading role is at |Lc| ${levels[3].toFixed(1)} there`);
});

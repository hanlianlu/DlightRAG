// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const css = readFileSync(fileURLToPath(new URL('./utopia.css', import.meta.url)), 'utf8');

/** The light overrides live in their own block; everything before it is dark. */
function themeBlock(theme: 'dark' | 'light'): string {
    const start = css.indexOf(":root[data-color-mode='light']");
    assert.ok(start > 0, 'light theme block not found');
    return theme === 'dark' ? css.slice(0, start) : css.slice(start);
}

function token(theme: 'dark' | 'light', name: string): string {
    const match = new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`).exec(themeBlock(theme));
    assert.ok(match, `--${name} not found as a hex in the ${theme} block`);
    return match[1];
}

function luminance(hex: string): number {
    const channels = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
    const linear = channels.map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

// Elevation steps away from the reading canvas: lighter in dark, darker in
// light. Nothing in a hex literal shows this, so it has to be checked here --
// the light ramp had silently inverted, making the panel brighter than the
// conversation it sits beside.
for (const [theme, direction] of [['dark', 1], ['light', -1]] as const) {
    test(`${theme} surfaces step away from the canvas`, () => {
        const ramp = ['color-bg-base', 'color-bg-surface', 'color-bg-elevated']
            .map((name) => ({name, luminance: luminance(token(theme, name))}));

        for (let i = 1; i < ramp.length; i += 1) {
            const delta = (ramp[i].luminance - ramp[i - 1].luminance) * direction;
            assert.ok(
                delta > 0,
                `${ramp[i].name} must sit further from the canvas than ${ramp[i - 1].name} `
                + `(luminance ${ramp[i - 1].luminance.toFixed(4)} -> ${ramp[i].luminance.toFixed(4)})`,
            );
        }
    });
}

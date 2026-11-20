// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Payload tests for the URL allowlists (lib/urls.ts) that every `href` and `src` the application
 * sets from model, document or server text passes through. They resolve against
 * `window.location.origin`, so they run in the browser suite. Each table row is an input and the
 * one answer the helper may give: a rejected URL is the empty string, never a sanitized variant. */

import {expect} from '@esm-bundle/chai';
import {safeExternalHttpHref, safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';

const origin = window.location.origin;
// What a URL with the page's own scheme and another host resolves to.
const otherHost = 'evil.example';

type Row = [name: string, input: unknown, expected: string];

// Spellings of a script scheme that the URL parser reads as that scheme.
const SCRIPT_URLS: Array<[string, string]> = [
  ['plain', 'javascript:alert(1)'],
  ['mixed case', 'JaVaScRiPt:alert(1)'],
  ['leading whitespace', '   javascript:alert(1)'],
  ['leading control character', '\u0001javascript:alert(1)'],
  ['tab inside the scheme', 'java\tscript:alert(1)'],
  ['newline inside the scheme', 'java\nscript:alert(1)'],
  ['vbscript', 'vbscript:msgbox(1)'],
  ['mixed-case vbscript', 'VbScRiPt:msgbox(1)'],
];

const OTHER_SCHEMES: Array<[string, string]> = [
  ['data html', 'data:text/html,<script>alert(1)</script>'],
  ['data html, base64', 'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=='],
  ['file', 'file:///etc/passwd'],
  ['ftp', 'ftp://example.com/x'],
  ['mailto', 'mailto:a@example.com'],
  ['tel', 'tel:+123'],
  ['blob from another origin', `blob:https://${otherHost}/0f5a`],
  ['about', 'about:blank'],
];

// Values that are not a URL at all.
const NOT_URLS: Array<[string, unknown]> = [
  ['empty', ''],
  ['spaces', '   '],
  ['undefined', undefined],
  ['null', null],
  ['number', 42],
  ['object', {href: '/x'}],
  ['array', ['/x']],
  ['unparsable authority', 'http://'],
  ['unparsable port', 'http://example.com:99999/'],
];

function rejects(helper: (value: unknown) => string, rows: ReadonlyArray<readonly [string, unknown]>): void {
  for (const [name, input] of rows) {
    it(`rejects ${name}`, () => {
      expect(helper(input)).to.equal('');
    });
  }
}

function answers(helper: (value: unknown) => string, rows: Row[]): void {
  for (const [name, input, expected] of rows) {
    it(name, () => {
      expect(helper(input)).to.equal(expected);
    });
  }
}

describe('safeSameOriginHref', () => {
  answers(safeSameOriginHref, [
    ['resolves a root-relative path against the page', '/web/files/1?x=1#top', `${origin}/web/files/1?x=1#top`],
    ['resolves a bare relative path', 'files/1', `${origin}/files/1`],
    ['keeps an absolute URL on this origin', `${origin}/a/b`, `${origin}/a/b`],
    ['keeps a fragment-only reference on the origin root', '#top', `${origin}/#top`],
    ['trims surrounding whitespace', `  /ok\t`, `${origin}/ok`],
    ['reads a scheme-only reference with the page scheme as relative', `${window.location.protocol}evil.example`, `${origin}/evil.example`],
  ]);

  describe('rejects another origin', () => {
    rejects(safeSameOriginHref, [
      ['an absolute https URL', `https://${otherHost}/x`],
      ['an absolute http URL', `http://${otherHost}/x`],
      ['a protocol-relative URL', `//${otherHost}/x`],
      ['a protocol-relative URL with no path', `//${otherHost}`],
      ['a backslash URL', `\\\\${otherHost}\\x`],
      ['a slash and a backslash', `/\\${otherHost}`],
      ['a tab between the slashes', `/\t/${otherHost}`],
      ['userinfo that names this origin', `${window.location.protocol}//${window.location.host}@${otherHost}/`],
      ['a backslash that ends the authority', `${window.location.protocol}//${otherHost}\\@${window.location.host}/`],
      ['this host on another port', `${window.location.protocol}//${window.location.hostname}:1/x`],
      ['this host on the other scheme', `${window.location.protocol === 'https:' ? 'http:' : 'https:'}//${window.location.host}/x`],
    ]);
  });

  describe('rejects a script scheme', () => {
    rejects(safeSameOriginHref, SCRIPT_URLS);
  });

  describe('rejects every other scheme', () => {
    rejects(safeSameOriginHref, OTHER_SCHEMES);
  });

  describe('rejects what is not a URL', () => {
    rejects(safeSameOriginHref, NOT_URLS);
  });
});

describe('safeExternalHttpHref', () => {
  answers(safeExternalHttpHref, [
    ['keeps an https URL with its query and fragment', 'https://example.com/a?b=1#c', 'https://example.com/a?b=1#c'],
    ['adds the root path to a bare host', 'http://example.com', 'http://example.com/'],
    ['lower-cases the scheme and host', 'HTTPS://Example.COM/A', 'https://example.com/A'],
    ['keeps a URL on this origin', `${origin}/x`, `${origin}/x`],
    ['resolves a relative path against this origin', '/x', `${origin}/x`],
    ['gives a protocol-relative URL the page scheme', `//${otherHost}/x`, `${window.location.protocol}//${otherHost}/x`],
    ['reads a backslash as a slash, as the browser navigates it', `https:\\\\example.com\\x`, 'https://example.com/x'],
    ['trims surrounding whitespace', '  https://example.com/x  ', 'https://example.com/x'],
    ['drops a tab inside the host the way the parser does', 'https://exa\tmple.com/x', 'https://example.com/x'],
  ]);

  describe('rejects a script scheme', () => {
    rejects(safeExternalHttpHref, SCRIPT_URLS);
  });

  describe('rejects every other scheme', () => {
    rejects(safeExternalHttpHref, OTHER_SCHEMES);
  });

  describe('rejects what is not a URL', () => {
    rejects(safeExternalHttpHref, NOT_URLS);
  });
});

describe('safeImageSrc', () => {
  const png = 'data:image/png;base64,iVBORw0KGgo=';

  answers(safeImageSrc, [
    ['keeps a same-origin path', '/web/images/1.png', `${origin}/web/images/1.png`],
    ['keeps a same-origin absolute URL', `${origin}/i.png?w=2`, `${origin}/i.png?w=2`],
    ['keeps a blob this page made', `blob:${origin}/0f5a-uuid`, `blob:${origin}/0f5a-uuid`],
    ['keeps an inline PNG', png, png],
    ['keeps an inline PNG after whitespace', `  ${png}  `, png],
    ['keeps an inline PNG with an upper-case scheme and subtype', 'DATA:IMAGE/PNG;BASE64,AAAA', 'data:IMAGE/PNG;BASE64,AAAA'],
    ['keeps an inline JPEG', 'data:image/jpeg;base64,/9j/4AAQ', 'data:image/jpeg;base64,/9j/4AAQ'],
    ['keeps an inline WebP', 'data:image/webp;base64,UklGRg==', 'data:image/webp;base64,UklGRg=='],
    ['keeps an inline GIF', 'data:image/gif;base64,R0lGODlh', 'data:image/gif;base64,R0lGODlh'],
    ['keeps an inline AVIF', 'data:image/avif;base64,AAAA', 'data:image/avif;base64,AAAA'],
    ['keeps an inline BMP', 'data:image/bmp;base64,Qk0=', 'data:image/bmp;base64,Qk0='],
  ]);

  describe('rejects an image on another origin', () => {
    rejects(safeImageSrc, [
      ['an absolute https URL', `https://${otherHost}/p.png`],
      ['an absolute http URL', `http://${otherHost}/p.png`],
      ['a protocol-relative URL', `//${otherHost}/p.png`],
      ['a backslash URL', `\\\\${otherHost}\\p.png`],
      ['a slash and a backslash', `/\\${otherHost}/p.png`],
      ['a blob of another origin', `blob:https://${otherHost}/0f5a`],
      ['a blob with no origin', 'blob:0f5a'],
    ]);
  });

  describe('rejects a data URL that is not a plain inline raster image', () => {
    rejects(safeImageSrc, [
      ['an SVG, which can hold script and links', 'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4='],
      ['an SVG with a plain-text body', 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>'],
      ['an HTML document', 'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=='],
      ['an HTML document dressed as an image type', 'data:text/html;image/png;base64,AAAA'],
      ['an image that is not base64', 'data:image/png,AAAA'],
      ['an image with parameters before base64', 'data:image/png;charset=utf-8;base64,AAAA'],
      ['an image whose body holds markup', 'data:image/png;base64,AAAA<script>alert(1)</script>'],
      ['an image whose body holds a newline', 'data:image/png;base64,AA\nAA'],
      ['an image with a fragment', 'data:image/png;base64,AAAA#x'],
      ['an image with an empty body', 'data:image/png;base64,'],
      ['an unknown image type', 'data:image/tiff;base64,AAAA'],
    ]);
  });

  describe('rejects a script scheme', () => {
    rejects(safeImageSrc, SCRIPT_URLS);
  });

  describe('rejects every other scheme', () => {
    rejects(safeImageSrc, OTHER_SCHEMES);
  });

  describe('rejects what is not a URL', () => {
    rejects(safeImageSrc, NOT_URLS);
  });
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {getArtifactPresentationAt, type AnswerPresentation, type LinkCard} from '../api/conversations.ts';
import type {AnswerPresentationElement} from './answer-presentation.ts';
import './answer-presentation.ts';
import cases from './fixtures/link-card-occurrences.json' with {type: 'json'};
import mixed from './fixtures/mixed-resource-presentation.json' with {type: 'json'};

const card = (url: string): LinkCard => ({url, title: 'Film <script>evil()</script>', description: 'Description', site: 'Example', image: null});
const cards = [card('https://example.com/x'), card('https://example.com/y'), card('https://example.com/x.'), card('https://example.com/x%5C%60')];

function presentation(text: string, html: string): AnswerPresentation {
  return {
    answerText: text,
    parts: [{type: 'markdown', text, html, artifact: null, evidenceImage: null, card: null, inline: false}],
    linkCards: cards, sources: [], evidenceImages: [], artifacts: [],
    artifactOutcome: {status: 'complete', issues: []},
  };
}

async function mount(value: AnswerPresentation): Promise<AnswerPresentationElement> {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = value;
  document.body.append(element);
  await element.updateComplete;
  return element;
}

afterEach(() => { document.body.replaceChildren(); });

for (const fixture of cases) {
  it(`upgrades only real link occurrences: ${fixture.name}`, async () => {
    const before = document.createElement('div');
    before.innerHTML = fixture.html;
    const code = [...before.querySelectorAll('code')].map((node) => node.textContent);
    const element = await mount(presentation(fixture.markdown, fixture.html));
    const links = [...element.querySelectorAll<HTMLAnchorElement>('[data-answer-link-card]')];
    expect(links.map((node) => node.href)).to.deep.equal(fixture.hrefs);
    expect([...element.querySelectorAll('code')].map((node) => node.textContent)).to.deep.equal(code);
    expect(element.querySelector('code [data-answer-link-card]')).to.equal(null);
    expect(element.querySelector('script, iframe')).to.equal(null);
    for (const link of links) {
      expect(link.target).to.equal('_blank');
      expect(link.rel).to.equal('noopener noreferrer');
      expect(link.textContent).to.contain('Film <script>evil()</script>');
    }
    for (const tag of ['blockquote', 'ul', 'li', 'table', 'td', 'strong']) {
      expect(element.querySelectorAll(tag).length, tag).to.equal(before.querySelectorAll(tag).length + (tag === 'strong' ? links.length : 0));
    }
  });
}

it('preserves the full grammar around placed video, images and link cards through the wire', async () => {
  const originalFetch = window.fetch;
  let value: AnswerPresentation;
  try {
    window.fetch = async () => new Response(JSON.stringify(mixed.wire), {status: 200, headers: {'Content-Type': 'application/json'}});
    value = await getArtifactPresentationAt('/presentation-fixture');
  } finally {
    window.fetch = originalFetch;
  }
  const element = await mount(value);
  expect(element.querySelectorAll('[data-answer-link-card]').length).to.equal(1);
  expect(element.querySelectorAll('[data-answer-video]').length).to.equal(1);
  expect(element.querySelectorAll('.answer-artifact-card').length).to.equal(1);
  expect(element.querySelectorAll('.answer-inline-evidence').length).to.equal(1);
  expect(element.querySelector('code')?.textContent).to.equal('before [quoted](artifact:clip.mp4) https://example.com/x after');
  expect(element.querySelector('code [data-answer-link-card]')).to.equal(null);
  expect(element.querySelector('li .answer-inline-evidence')).not.to.equal(null);
  expect(element.querySelector('.answer-evidence')).to.equal(null);
  expect(element.querySelector('blockquote [data-answer-link-card]')).to.equal(null);
  let opened = '';
  element.addEventListener('dl-artifact-open', (event) => { opened = (event as CustomEvent).detail.artifact.resourceId; });
  element.querySelector<HTMLButtonElement>('.answer-artifact-card button')?.click();
  expect(opened).to.equal('clip.mp4');
});

it('keeps a slotted video alive across References toggles and equivalent presentations', async () => {
  const originalFetch = window.fetch;
  let value: AnswerPresentation;
  try {
    window.fetch = async () => new Response(JSON.stringify(mixed.wire), {status: 200, headers: {'Content-Type': 'application/json'}});
    value = await getArtifactPresentationAt('/presentation-fixture');
  } finally {
    window.fetch = originalFetch;
  }
  value.sources = Array.from({length: 4}, (_, i) => ({id: String(i + 1), title: 'Source', sourceUrl: null, downloadUrl: null, chunks: []}));
  const element = await mount(value);
  const player = element.querySelector<HTMLVideoElement>('video')!;
  player.volume = 0.25;
  player.currentTime = 7;
  element.querySelector<HTMLButtonElement>('[aria-expanded]')!.click();
  await element.updateComplete;
  expect(element.querySelector('video')).to.equal(player);
  expect(player.volume).to.equal(0.25);
  expect(player.currentTime).to.equal(7);
  element.presentation = {...value, linkCards: []};
  await element.updateComplete;
  expect(element.querySelector('video')).to.equal(player);
  expect(player.currentTime).to.equal(7);
  expect(element.querySelector('[data-answer-link-card]')).to.equal(null);
});

it('isolates typed thumbnails and cards from prose image and link styles', async () => {
  const stylesheets: HTMLLinkElement[] = [];
  try {
    for (const path of ['../design-system/index.css', '../styles/answer-presentation.module.css', '../styles/chat.module.css']) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = new URL(path, import.meta.url).href;
      stylesheets.push(link);
      await new Promise<void>((resolve, reject) => {
        link.onload = () => resolve(); link.onerror = reject;
        document.head.append(link);
      });
    }
    const image = {id: 'shot', chunkId: '', sourceRef: '', url: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7', thumbnailUrl: '', label: 'Shot', answerImageSent: true};
    const value = presentation('Photo', '<p><span class="answer-resource-slot-0"></span> <a href="https://example.com/x">film</a></p>');
    value.parts.push({type: 'evidence_image', text: '', html: '', artifact: null, evidenceImage: image, card: null, inline: true, slot: 0});
    value.linkCards = [{...cards[0], image: 'https://example.com/cover.png'}];
    const element = await mount(value);
    const thumbnail = element.querySelector<HTMLImageElement>('[data-answer-image] img')!;
    const button = thumbnail.closest('button')!;
    const style = getComputedStyle(thumbnail);
    expect(style.marginTop).to.equal('0px');
    expect(style.marginBottom).to.equal('0px');
    expect(thumbnail.getBoundingClientRect().height).to.equal(button.clientHeight);
    const cover = element.querySelector<HTMLImageElement>('[data-answer-link-card] img')!;
    expect(getComputedStyle(cover).marginTop).to.equal('0px');
    expect(getComputedStyle(cover).maxWidth).to.equal('220px');
    const cardLink = element.querySelector<HTMLAnchorElement>('[data-answer-link-card]')!;
    const probe = document.createElement('span');
    probe.style.color = 'var(--color-text-primary)'; element.append(probe);
    expect(getComputedStyle(cardLink).color).to.equal(getComputedStyle(probe).color);
  } finally {
    stylesheets.forEach((link) => { link.remove(); });
  }
});

it('leaves source links and citation controls alone', async () => {
  const value = presentation('Source', '<p><a href="https://example.com/x">source</a> <cite class="citation-badge" data-ref="1">1</cite></p>');
  value.sources = [{id: '1', title: 'Source', sourceUrl: 'https://example.com/x', downloadUrl: null, chunks: []}];
  const element = await mount(value);
  expect(element.querySelector('[data-answer-link-card]')).to.equal(null);
  expect(element.querySelector('a')?.textContent).to.equal('source');
  expect(element.querySelector('.citation-badge')?.textContent).to.equal('1');
});

it('rebuilds from sanitized HTML without duplicate cards on updates', async () => {
  const value = presentation('Link', '<p><a href="https://example.com/x">film</a></p>');
  const element = await mount(value);
  element.presentation = {...value};
  await element.updateComplete;
  expect(element.querySelectorAll('[data-answer-link-card]').length).to.equal(1);
  element.presentation = {...value, linkCards: []};
  await element.updateComplete;
  expect(element.querySelector('[data-answer-link-card]')).to.equal(null);
  expect(element.querySelector('a')?.textContent).to.equal('film');
});

it('never turns unsafe card metadata into a same-DOM HTML sink', async () => {
  const value = presentation('Link', '<p><a href="javascript:evil()">unsafe</a><a href="https://example.com/x">safe</a></p>');
  value.linkCards = [{...cards[0], image: 'javascript:evil()'}, card('javascript:evil()')];
  const element = await mount(value);
  expect(element.querySelectorAll('[data-answer-link-card]').length).to.equal(1);
  expect(element.querySelector('img, script, iframe')).to.equal(null);
  expect(element.querySelector('a[href^="javascript:"]')).to.equal(null);
});

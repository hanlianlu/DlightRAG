// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {type AnswerArtifact, getArtifactPresentationAt} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import './answer-presentation.ts';
import './artifact-canvas.ts';

defineDesignSystemElements();

const url = 'https://www.youtube.com/watch?v=abcdefghijk';
const wire = {
  answer_text: `Watch ${url}`,
  parts: [{type: 'markdown', text: '', html: `<p><a href="${url}" target="_blank">Watch this</a></p>`, artifact: null, evidence_image: null, card: null, inline: false}],
  video_links: [{url, provider: 'YouTube', player_domains: ['www.youtube-nocookie.com', 'youtube.com']}],
  sources: [], evidence_images: [], link_cards: [], artifacts: [],
  artifact_outcome: {status: 'complete', issues: []},
};
const originalFetch = window.fetch;

// Deterministic tests exercise our DOM/intent contract, never third-party media.
// Actual provider playback is qualified separately against the deployment.
before(() => {
  const csp = document.createElement('meta');
  csp.httpEquiv = 'Content-Security-Policy';
  csp.content = "frame-src 'none'; img-src 'none'";
  document.head.append(csp);
});

async function mount() {
  const value = await getArtifactPresentationAt('/fixture');
  const element = document.createElement('dl-answer-presentation');
  element.presentation = value;
  document.body.append(element);
  await element.updateComplete;
  await element.querySelector('dl-video-playback')?.updateComplete;
  return element;
}

function until(selector: string, root: ParentNode = document): Promise<Element> {
  return new Promise((resolve, reject) => {
    const find = () => {
      const node = root.querySelector(selector);
      if (node) { observer.disconnect(); clearTimeout(timeout); resolve(node); }
    };
    const observer = new MutationObserver(find);
    const timeout = setTimeout(() => { observer.disconnect(); reject(new Error(`Missing ${selector}`)); }, 3000);
    observer.observe(root, {childList: true, subtree: true, attributes: true});
    find();
  });
}

const result = (id = 'abcdefghijk') => new Response(JSON.stringify({
  embed_url: `https://www.youtube-nocookie.com/embed/${id}?autoplay=1&playsinline=1`,
  aspect_ratio: 16 / 9,
}), {headers: {'Content-Type': 'application/json'}});

afterEach(() => { document.body.replaceChildren(); window.fetch = originalFetch; });

it('offers playback for a metadata-free prose link without requesting a player before click', async () => {
  const calls: string[] = [];
  window.fetch = async (input) => {
    calls.push(String(input));
    return new Response(JSON.stringify(wire), {headers: {'Content-Type': 'application/json'}});
  };
  const element = await mount();
  expect(element.querySelector('[data-video-play]')).not.to.equal(null);
  expect(element.querySelector('a')?.href).to.equal(url);
  expect(element.querySelector('iframe')).to.equal(null);
  expect(calls).to.deep.equal(['/fixture']);
});

it('loads one narrowly granted player only on click and Stop restores the original link', async () => {
  const calls: RequestInit[] = [];
  window.fetch = async (input, init) => {
    if (String(input) === '/fixture') return new Response(JSON.stringify(wire));
    expect(input).to.equal('/web/api/video-playback');
    calls.push(init!);
    return result();
  };
  const element = await mount();
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const frame = await until('iframe', element) as HTMLIFrameElement;
  expect(calls.length).to.equal(1);
  expect(calls[0].method).to.equal('POST');
  expect(JSON.parse(String(calls[0].body))).to.deep.equal({url});
  expect(frame.src).to.equal('https://www.youtube-nocookie.com/embed/abcdefghijk?autoplay=1&playsinline=1');
  expect(frame.referrerPolicy).to.equal('strict-origin-when-cross-origin');
  expect(frame.sandbox.contains('allow-scripts')).to.equal(true);
  expect(frame.sandbox.contains('allow-top-navigation')).to.equal(false);
  expect(frame.getAttribute('allow')).to.contain('fullscreen');
  expect(element.querySelector('a')?.href).to.equal(url);
  element.querySelector<HTMLButtonElement>('[data-video-stop]')!.click();
  expect(frame.isConnected).to.equal(false);
  await until('[data-video-play]', element);
  expect(element.querySelector('a')?.textContent).to.equal('Watch this');
});

it('selecting B across Answers cancels A and a late A resolution cannot steal playback', async () => {
  let resolveA!: (value: Response) => void;
  let signalA: AbortSignal | null | undefined;
  let requests = 0;
  window.fetch = async (input, init) => {
    if (String(input) === '/fixture') return new Response(JSON.stringify(wire));
    requests += 1;
    if (requests === 1) {
      signalA = init?.signal;
      return new Promise<Response>((resolve) => { resolveA = resolve; });
    }
    return result();
  };
  const a = await mount();
  const b = await mount();
  a.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  b.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const frameB = await until('iframe', b);
  expect(signalA?.aborted).to.equal(true);
  resolveA(result());
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  expect(a.querySelector('iframe')).to.equal(null);
  expect(document.querySelectorAll('iframe').length).to.equal(1);
  expect(b.querySelector('iframe')).to.equal(frameB);
  b.remove();
  expect(frameB.isConnected).to.equal(false);
});

it('unrelated presentation updates preserve the player and selecting another removes it immediately', async () => {
  window.fetch = async (input) => String(input) === '/fixture'
    ? new Response(JSON.stringify(wire)) : result();
  const a = await mount();
  a.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const frame = await until('iframe', a);
  a.presentation = {...a.presentation!, linkCards: []};
  await a.updateComplete;
  await a.querySelector('dl-video-playback')?.updateComplete;
  expect(a.querySelector('iframe')).to.equal(frame);
  const b = await mount();
  b.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  expect(frame.isConnected).to.equal(false);
  await until('iframe', b);
  expect(document.querySelectorAll('iframe').length).to.equal(1);
});

it('keeps published citation occurrences and code out of playback while upgrading the recommendation', async () => {
  const markup = `<p>Fact <a class="answer-citation-link" href="${url}">9</a>.
    <cite class="citation-badge" data-ref="9">9</cite>
    <code>${url}</code> Watch <a href="${url}">this</a>.</p>`;
  window.fetch = async () => new Response(JSON.stringify({
    ...wire, parts: [{...wire.parts[0], html: markup}],
    link_cards: [{url, title: 'Film', description: '', site: 'YouTube', image: null}],
  }));
  const element = await mount();
  expect(element.querySelectorAll('[data-video-play]').length).to.equal(1);
  expect(element.querySelectorAll('[data-answer-link-card]').length).to.equal(1);
  expect(element.querySelector('.answer-citation-link')?.textContent).to.equal('9');
  expect(element.querySelector('.answer-citation-link')?.closest('dl-video-playback')).to.equal(null);
  expect(element.querySelector('.citation-badge')?.closest('dl-video-playback')).to.equal(null);
  expect(element.querySelector('code')?.textContent).to.equal(url);
});

it('restores the original preview card and cover when B takes over', async () => {
  const cover = 'https://images.example/film.jpg';
  window.fetch = async (input) => String(input) === '/fixture'
    ? new Response(JSON.stringify({...wire, link_cards: [{url, title: 'Film', description: 'About film', site: 'YouTube', image: cover}]}))
    : result();
  const a = await mount();
  const b = await mount();
  a.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const frameA = await until('iframe', a);
  expect(a.querySelector('[data-answer-link-card]')).to.equal(null);
  b.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  expect(frameA.isConnected).to.equal(false);
  await until('[data-answer-link-card]', a);
  await until('iframe', b);
  expect(a.querySelector('img')?.getAttribute('src')).to.equal(cover);
  expect(a.querySelector('strong')?.textContent).to.equal('Film');
  expect(a.querySelector('[data-video-play]')).not.to.equal(null);
  expect(document.querySelectorAll('iframe').length).to.equal(1);
});

it('can retry a failed request and replay after Stop without losing the preview or player root', async () => {
  let requests = 0;
  window.fetch = async (input) => {
    if (String(input) === '/fixture') return new Response(JSON.stringify(wire));
    return ++requests === 1 ? new Response('Unavailable', {status: 503}) : result();
  };
  const element = await mount();
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  await until('[role="alert"]', element);
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const first = await until('iframe', element);
  expect(element.querySelector('[role="alert"]')).to.equal(null);
  element.querySelector<HTMLButtonElement>('[data-video-stop]')!.click();
  await until('[data-video-play]', element);
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const second = await until('iframe', element);
  expect(second).not.to.equal(first);
  expect(requests).to.equal(3);
});

it('clearing an Answer cancels a pending resolver and cannot resurrect its removed player', async () => {
  let complete!: (value: Response) => void;
  let signal: AbortSignal | null | undefined;
  window.fetch = async (input, init) => {
    if (String(input) === '/fixture') return new Response(JSON.stringify(wire));
    signal = init?.signal;
    return new Promise<Response>((resolve) => { complete = resolve; });
  };
  const element = await mount();
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  element.presentation = null;
  await element.updateComplete;
  expect(signal?.aborted).to.equal(true);
  complete(result());
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  expect(element.querySelector('dl-video-playback')).to.equal(null);
  expect(document.querySelector('iframe')).to.equal(null);
});

it('shares ownership with a Markdown Artifact Canvas and closing the Canvas tears its player down', async () => {
  window.fetch = async (input) => String(input) === '/web/api/video-playback'
    ? result() : new Response(JSON.stringify(wire));
  const answer = await mount();
  answer.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const original = await until('iframe', answer);
  const canvas = document.createElement('dl-artifact-canvas');
  document.body.append(canvas);
  const artifact: AnswerArtifact = {
    resourceId: 'report', mediaType: 'text/markdown', label: 'Report', filename: 'report.md',
    byteSize: 42, digest: 'a'.repeat(64), presentation: 'markdown', status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/report', width: null, height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/report',
    downloadUrl: '/web/api/answer/run-1/artifacts/report?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/report/presentation', issue: null,
  };
  await canvas.open(artifact);
  const play = await until('[data-video-play]', canvas) as HTMLButtonElement;
  play.click();
  const frame = await until('iframe', canvas);
  expect(original.isConnected).to.equal(false);
  expect(document.querySelectorAll('iframe').length).to.equal(1);
  canvas.close();
  await canvas.updateComplete;
  expect(frame.isConnected).to.equal(false);
  expect(document.querySelector('iframe')).to.equal(null);
  expect(answer.querySelector('[data-video-play]')).not.to.equal(null);
});

it('plays a registry publisher outside the three examples using the selected link permissions', async () => {
  const destination = 'https://www.dailymotion.com/video/fixture';
  const player = 'https://geo.dailymotion.com/player.html?video=fixture';
  window.fetch = async (input) => String(input) === '/fixture'
    ? new Response(JSON.stringify({
      ...wire,
      parts: [{...wire.parts[0], html: `<p><a href="${destination}">Film</a></p>`}],
      video_links: [{url: destination, provider: 'Dailymotion', player_domains: ['dailymotion.com']}],
    }))
    : new Response(JSON.stringify({embed_url: player, aspect_ratio: 16 / 9}));
  const element = await mount();
  expect(element.querySelector('iframe')).to.equal(null);
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  const frame = await until('iframe', element) as HTMLIFrameElement;
  expect(frame.src).to.equal(player);
  expect(element.querySelector('a')?.href).to.equal(destination);
});

it('rejects an unsafe wire destination and leaves the source link and retry available', async () => {
  window.fetch = async (input) => String(input) === '/fixture'
    ? new Response(JSON.stringify(wire))
    : new Response(JSON.stringify({embed_url: 'https://evil.example/player', aspect_ratio: 1}));
  const element = await mount();
  element.querySelector<HTMLButtonElement>('[data-video-play]')!.click();
  await until('[role="alert"]', element);
  expect(element.querySelector('iframe')).to.equal(null);
  expect(element.querySelector('a')?.href).to.equal(url);
  expect(element.querySelector('[data-video-play]')).not.to.equal(null);
});

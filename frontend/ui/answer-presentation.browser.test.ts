// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact, AnswerPresentation} from '../api/conversations.ts';
import {setLanguagePreference} from '../i18n/locale.ts';
import './answer-presentation.ts';
import './inspector-sources.ts';
import type {AnswerPresentationElement} from './answer-presentation.ts';
import type {DlImageLightbox} from './image-lightbox.ts';
import './image-lightbox.ts';
import type {DlInspectorSources} from './inspector-sources.ts';

const presentation: AnswerPresentation = {
  answerText: 'Safe [1].',
  parts: [{
    type: 'markdown',
    text: 'Safe [1].',
    html: '<p>Safe <cite class="citation-badge" data-ref="1">1</cite></p><script>x()</script>',
    artifact: null,
    evidenceImage: null,
    inline: false,
  }],
  sources: [
    {
      id: '1',
      title: '<img src=x>',
      sourceUrl: 'https://example.com/report',
      downloadUrl: 'https://evil.example.com/steal',
      chunks: [
        {
          chunkIdx: 2,
          pageNumber: 4,
          contentHtml: '<p>Evidence</p><script>x()</script>',
          imageUrl: null,
          thumbnailUrl: null,
        },
      ],
    },
  ],
  evidenceImages: [], linkCards: [],
  artifacts: [],
  artifactOutcome: {status: 'complete', issues: []},
};

afterEach(() => { document.body.replaceChildren(); });

it('sanitizes rich answer HTML while Lit escapes structured references', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = presentation;
  document.body.appendChild(element);
  await element.updateComplete;

  expect(element.querySelector('.answer-rich-content')?.textContent).to.contain('Safe 1');
  expect(element.querySelector('script')).to.equal(null);
  expect(element.querySelector('.answer-ref-title')?.textContent).to.equal('<img src=x>');
  expect(element.querySelector('.answer-ref-title img')).to.equal(null);

  let referenceId = '';
  let eventPresentation: AnswerPresentation | null = null;
  element.addEventListener('dl-answer-source-open', (event) => {
    const detail = (event as CustomEvent).detail;
    referenceId = detail.referenceId;
    eventPresentation = detail.presentation;
  });
  element.querySelector<HTMLElement>('.citation-badge')?.click();
  expect(referenceId).to.equal('1');
  expect(eventPresentation).to.equal(element.presentation);
});

it('expands and collapses long References lists locally per Answer', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = {
    ...presentation,
    sources: Array.from({length: 7}, (_, index) => ({
      ...presentation.sources[0],
      id: String(index + 1),
      title: `Reference ${index + 1}`,
    })),
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const toggle = element.querySelector<HTMLButtonElement>('.answer-references-toggle')!;
  expect(toggle.textContent?.replace(/\s+/g, ' ').trim()).to.equal('References Show all 7');
  expect(toggle.getAttribute('aria-expanded')).to.equal('false');
  // The caption itself is the control, in words: no glyph rides with it, and the
  // bar the reader lights is the whole line.
  expect(toggle.closest('h3.answer-section-title')).to.not.equal(null);
  expect(toggle.querySelector('svg')).to.equal(null);
  toggle.click();
  await element.updateComplete;

  const collapse = element.querySelector<HTMLButtonElement>('.answer-references-toggle')!;
  expect(element.querySelector('.answer-reference-list')?.classList.contains('expanded')).to.equal(true);
  expect(collapse.textContent?.replace(/\s+/g, ' ').trim()).to.equal('References Show fewer');
  expect(collapse.getAttribute('aria-expanded')).to.equal('true');
  collapse.click();
  await element.updateComplete;
  expect(element.querySelector('.answer-reference-list')?.classList.contains('expanded')).to.equal(false);
});

it('renders the References control from three sources and not from two', async () => {
  // Two sources can never be truncated (the collapsed list shows two rows), so
  // the render gate starts at three and the stylesheet decides whether it is shown.
  const render = async (count: number) => {
    const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
    element.presentation = {
      ...presentation,
      sources: Array.from({length: count}, (_, index) => ({
        ...presentation.sources[0],
        id: String(index + 1),
        title: `Reference ${index + 1}`,
      })),
    };
    document.body.appendChild(element);
    await element.updateComplete;
    const toggle = element.querySelector('.answer-references-toggle');
    element.remove();
    return toggle;
  };

  expect(await render(2)).to.equal(null);
  expect(await render(3)).to.not.equal(null);
});

const HIGHLIGHT_CLASS = 'answer-ref-item-highlighted';

const referenceCard = (element: AnswerPresentationElement, id: string): HTMLElement | null =>
  element.querySelector<HTMLElement>(`[data-reference-id="${CSS.escape(id)}"]`);

/** The ids of the rows the trace has lit, in reading order. */
const litReferences = (element: AnswerPresentationElement): string[] =>
  [...element.querySelectorAll<HTMLElement>(`.${HIGHLIGHT_CLASS}`)]
    .map((row) => row.getAttribute('data-reference-id') ?? '');

const referencesExpanded = (element: AnswerPresentationElement): boolean =>
  element.querySelector('.answer-reference-list')?.classList.contains('expanded') === true;

/** What a trace does for one source: one request, heard by every Answer. */
const askForReference = (sourceId: string, runId?: string | null): void => {
  document.dispatchEvent(new CustomEvent('dl-show-reference', {detail: {sourceId, runId}}));
};

const answerWithReferences = (ids: readonly string[]): AnswerPresentation => ({
  ...presentation,
  sources: ids.map((id) => ({...presentation.sources[0], id, title: `Reference ${id}`})),
});

it('opens the References on the source a trace names, lights that row, and ignores a foreign id', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = answerWithReferences(['1', '2', '3', '4', '5']);
  document.body.appendChild(element);
  await element.updateComplete;

  // Each row carries the id the Answer publishes, which is what the trace names.
  expect([...element.querySelectorAll<HTMLElement>('[data-reference-id]')]
    .map((row) => row.getAttribute('data-reference-id')))
    .to.deep.equal(['1', '2', '3', '4', '5']);
  expect(referencesExpanded(element), 'the list starts folded').to.equal(false);
  expect(litReferences(element)).to.deep.equal([]);

  // A source another Answer holds: nothing opens, nothing lights, nothing scrolls.
  askForReference('99');
  await element.updateComplete;
  expect(referencesExpanded(element)).to.equal(false);
  expect(litReferences(element)).to.deep.equal([]);

  // A row the folded list was hiding is revealed first, then lit, then brought into view.
  askForReference('4');
  await element.updateComplete;
  expect(referencesExpanded(element)).to.equal(true);
  expect(litReferences(element)).to.deep.equal(['4']);
  expect(referenceCard(element, '4')?.getBoundingClientRect().height).to.be.greaterThan(0);
  expect(referenceCard(element, '1')?.classList.contains(HIGHLIGHT_CLASS)).to.equal(false);

  // The next request owns the light: the row it names is lit, the one before goes out.
  askForReference('2');
  await element.updateComplete;
  expect(litReferences(element)).to.deep.equal(['2']);
  expect(referenceCard(element, '4')?.classList.contains(HIGHLIGHT_CLASS)).to.equal(false);

  // A reader who was in the trace lands on the row the trace pointed at, and a screen reader
  // hears that the answer moved them there rather than being left on the Inspector.
  expect(document.activeElement).to.equal(referenceCard(element, '2'));
  expect(element.querySelector('[data-reference-announcer]')!.textContent)
    .to.contain('Reference 2 shown');
});

it('answers a trace request only for the Answer that holds the source', async () => {
  const holder = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  holder.presentation = answerWithReferences(['1', '2', '3']);
  const other = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  other.presentation = answerWithReferences(['a', 'b']);
  document.body.append(holder, other);
  await Promise.all([holder.updateComplete, other.updateComplete]);

  askForReference('3');
  await Promise.all([holder.updateComplete, other.updateComplete]);

  expect(referencesExpanded(holder), 'the Answer holding the source opens').to.equal(true);
  expect(litReferences(holder)).to.deep.equal(['3']);
  expect(referencesExpanded(other), 'an Answer without it stays as it was').to.equal(false);
  expect(litReferences(other)).to.deep.equal([]);
});

it('stops listening when an Answer leaves the page, and listens again when it returns', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = answerWithReferences(['1', '2', '3']);
  document.body.appendChild(element);
  await element.updateComplete;

  element.remove();
  askForReference('2');
  await element.updateComplete;
  expect(litReferences(element), 'a detached Answer answers nothing').to.deep.equal([]);
  expect(referencesExpanded(element)).to.equal(false);

  // A new connect starts a new lifetime: the page's request reaches the Answer again.
  document.body.appendChild(element);
  await element.updateComplete;
  askForReference('2');
  await element.updateComplete;
  expect(litReferences(element)).to.deep.equal(['2']);
});

it('renders Artifact intent and semantic Visual Evidence in approved order', async () => {
  const artifact = {
    resourceId: 'artifact-report',
    mediaType: 'text/markdown',
    label: 'Quarterly report',
    filename: 'report.md',
    byteSize: 20,
    digest: 'a'.repeat(64),
    presentation: 'markdown' as const,
    status: 'available' as const,
    uri: 'dlightrag://answer/run-1/artifacts/artifact-report',
    width: null,
    height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-report',
    downloadUrl: '/web/api/answer/run-1/artifacts/artifact-report?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/artifact-report/presentation',
    issue: null,
    // The server derived no attestation record for these, which draws nothing.
    attestation: null,
  };
  const value: AnswerPresentation = {
    ...presentation,
    parts: [
      presentation.parts[0],
      {type: 'artifact', text: '', html: '', artifact, evidenceImage: null, inline: false},
    ],
    artifacts: [artifact],
    evidenceImages: [{
      id: 'image-1', chunkId: 'chunk-1', sourceRef: '1',
      url: '/web/api/images/default/chunk-1?size=full',
      thumbnailUrl: '/web/api/images/default/chunk-1?size=thumb',
      label: 'Chart', answerImageSent: true,
    }],
  };
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = value;
  document.body.appendChild(element);
  await element.updateComplete;

  let opened = '';
  let imageSource = '';
  element.addEventListener('dl-artifact-open', (event) => {
    opened = (event as CustomEvent).detail.artifact.resourceId;
  });
  element.addEventListener('dl-image-open', (event) => {
    imageSource = event.detail.src;
  });
  element.querySelector<HTMLButtonElement>('.answer-artifact-card .dl-btn')?.click();
  element.querySelector<HTMLButtonElement>('.answer-image-item')?.click();

  expect(opened).to.equal('artifact-report');
  expect(imageSource).to.equal(
    new URL('/web/api/images/default/chunk-1?size=full', window.location.origin).href,
  );
  expect(element.querySelector('.answer-evidence h3')?.textContent).to.equal('Visual Evidence');
  expect(element.querySelector('.answer-image-source')?.getAttribute('data-ref')).to.equal('1');
  const evidence = element.querySelector('.answer-evidence');
  const references = element.querySelector('.answer-references');
  expect(Boolean(evidence && references && (evidence.compareDocumentPosition(references) & Node.DOCUMENT_POSITION_FOLLOWING))).to.equal(true);
});

it('plays an inline video Artifact and keeps the card for a non-inline placement', async () => {
  const video = {
    resourceId: 'artifact-video',
    mediaType: 'video/mp4',
    label: 'Screen recording',
    filename: 'clip.mp4',
    byteSize: 1444,
    digest: 'a'.repeat(64),
    presentation: 'video' as const,
    status: 'available' as const,
    uri: 'dlightrag://answer/run-1/artifacts/artifact-video',
    width: null,
    height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-video',
    downloadUrl: '/web/api/answer/run-1/artifacts/artifact-video?download=1',
    presentationUrl: null,
    issue: null,
    // The server derived no attestation record for these, which draws nothing.
    attestation: null,
  };
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact: video, evidenceImage: null, inline: true}],
    artifacts: [video],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const player = element.querySelector<HTMLVideoElement>('[data-answer-video]');
  expect(player?.src).to.equal(
    new URL('/web/api/answer/run-1/artifacts/artifact-video', window.location.origin).href,
  );
  expect(player?.hasAttribute('controls')).to.equal(true);
  expect(player?.getAttribute('preload')).to.equal('metadata');
  expect(element.querySelector('.answer-artifact-card')).to.equal(null);
  // The escape hatch a browser that cannot decode the container needs.
  const captionLinks = Array.from(element.querySelectorAll<HTMLAnchorElement>('figcaption a'));
  expect(captionLinks.map((link) => link.textContent?.trim()))
    .to.deep.equal(['Open in a new tab', 'Download']);
  // Both wear the shared button face, so neither reads as a link by colour alone.
  expect(captionLinks.every((link) => link.classList.contains('dl-btn'))).to.equal(true);
  expect(captionLinks[0]?.href).to.equal(player?.src);
  expect(captionLinks[1]?.getAttribute('href')).to.equal(
    new URL('/web/api/answer/run-1/artifacts/artifact-video?download=1', window.location.origin).href,
  );
  expect(captionLinks[1]?.hasAttribute('download')).to.equal(true);

  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact: video, evidenceImage: null, inline: false}],
    artifacts: [video],
  };
  await element.updateComplete;

  expect(element.querySelector('video')).to.equal(null);
  expect(element.querySelector('.answer-artifact-card')).not.to.equal(null);
});

it('upgrades a linked page with card metadata into a card that leaves for its source', async () => {
  const card = {
    url: 'https://www.youtube.com/watch?v=abc123',
    title: 'Big Buck Bunny',
    description: 'A short film.',
    site: 'YouTube',
    image: 'https://i.ytimg.com/vi/abc123/hqdefault.jpg',
  };
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = {
    ...presentation,
    parts: [{
      type: 'markdown',
      text: `[Watch](${card.url})`,
      html: `<p><a href="${card.url}">Watch</a></p>`,
      artifact: null,
      evidenceImage: null,
      inline: false,
    }],
    linkCards: [card],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const anchor = element.querySelector<HTMLAnchorElement>('[data-answer-link-card]');
  expect(anchor?.getAttribute('href')).to.equal(card.url);
  expect(anchor?.getAttribute('target')).to.equal('_blank');
  expect(anchor?.getAttribute('rel')).to.equal('noopener noreferrer');
  expect(anchor?.querySelector('img')?.getAttribute('src')).to.equal(card.image);
  expect(anchor?.textContent).to.contain('Big Buck Bunny');
  expect(anchor?.textContent).to.contain('A short film.');
  expect(anchor?.textContent).to.contain('YouTube');
  // A card is a link out: the platform is never embedded.
  expect(element.querySelector('iframe')).to.equal(null);
});

it('includes typed Answer images in previous and next gallery navigation', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = {
    ...presentation,
    evidenceImages: [
      {
        id: 'image-1', chunkId: 'chunk-1', sourceRef: '1',
        url: '/web/api/images/default/chunk-1?size=full', thumbnailUrl: '',
        label: 'First', answerImageSent: true,
      },
      {
        id: 'image-2', chunkId: 'chunk-2', sourceRef: '1',
        url: '/web/api/images/default/chunk-2?size=full', thumbnailUrl: '',
        label: 'Second', answerImageSent: true,
      },
    ],
  };
  document.body.appendChild(element);
  await element.updateComplete;
  const images = element.querySelectorAll<HTMLElement>('[data-answer-image]');

  const detail = await new Promise<CustomEvent>((resolve) => {
    element.addEventListener('dl-image-open', (event) => resolve(event), {once: true});
    images[0].click();
  });
  const lightbox = document.createElement('dl-image-lightbox') as DlImageLightbox;
  document.body.appendChild(lightbox);
  await lightbox.open(detail.detail.src, detail.detail.returnFocus, detail.detail.gallery);
  const next = lightbox.querySelector<HTMLButtonElement>('[aria-label="Next"]')!;
  expect(next.hidden).to.equal(false);
  next.click();
  await lightbox.updateComplete;
  expect(lightbox.querySelector<HTMLImageElement>('img')?.src).to.equal(
    new URL(images[1].dataset.src ?? '', window.location.origin).href,
  );
  lightbox.close();
});

it('renders sanitized source chunks and rejects cross-origin download links', async () => {
  const view = document.createElement('dl-inspector-sources') as DlInspectorSources;
  view.sources = presentation.sources;
  view.setSelection('1', '2');
  document.body.appendChild(view);
  await view.updateComplete;

  expect(view.querySelector('.source-doc')?.classList.contains('expanded')).to.equal(true);
  expect(view.querySelector('.source-chunk-content')?.textContent).to.equal('Evidence');
  expect(view.querySelector('script')).to.equal(null);
  expect(view.querySelector('a[download]')).to.equal(null);
  expect(view.querySelector<HTMLAnchorElement>('a[aria-label="Open source"]')?.href).to.equal(
    'https://example.com/report',
  );
});

it('hardens external links in answer and source content while downloads keep their contract', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = {
    ...presentation,
    parts: [{
      ...presentation.parts[0],
      html: '<p><a href="https://elsewhere.example/x">open</a>' +
        '<a href="/web/api/files/raw/doc" download>save</a></p>',
    }],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const [open, save] = [...element.querySelectorAll<HTMLAnchorElement>('a')];
  expect(open.target).to.equal('_blank');
  expect(open.rel).to.equal('noopener noreferrer');
  expect(save.hasAttribute('target')).to.equal(false);

  const view = document.createElement('dl-inspector-sources') as DlInspectorSources;
  view.sources = [{
    ...presentation.sources[0],
    chunks: [{
      ...presentation.sources[0].chunks[0],
      contentHtml: '<p><a href="https://elsewhere.example/y">open</a></p>',
    }],
  }];
  view.setSelection('1');
  document.body.appendChild(view);
  await view.updateComplete;

  const chunkLink = view.querySelector<HTMLAnchorElement>('.source-chunk-content a')!;
  expect(chunkLink.target).to.equal('_blank');
  expect(chunkLink.rel).to.equal('noopener noreferrer');
});

it('answer and source surfaces typeset through the one rich-content pipeline', async () => {
  const typesetContainers: Element[][] = [];
  const windowRef = window as {MathJax?: unknown};
  const original = windowRef.MathJax;
  windowRef.MathJax = {
    typesetPromise: (containers: Element[]) => {
      typesetContainers.push(containers);
      return Promise.resolve();
    },
  };
  try {
    const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
    element.presentation = {
      ...presentation,
      parts: [{
        ...presentation.parts[0],
        html: '<p>See</p><pre class="mermaid-source" data-lang="mermaid">' +
          '<code>graph TD;\nA--&gt;B;</code></pre>',
      }],
    };
    document.body.appendChild(element);
    await element.updateComplete;

    const answerHost = element.querySelector<HTMLElement>('[data-answer-part]')!;
    expect(typesetContainers.flat().includes(answerHost)).to.equal(true);
    expect(element.querySelector('pre.mermaid-source')).to.not.equal(null);

    const view = document.createElement('dl-inspector-sources') as DlInspectorSources;
    view.sources = [{
      ...presentation.sources[0],
      chunks: [{
        ...presentation.sources[0].chunks[0],
        contentHtml: '<p>before</p><pre class="mermaid-source">' +
          '<code>graph TD;\nA--&gt;B;</code></pre>',
      }],
    }];
    view.setSelection('1');
    document.body.appendChild(view);
    await view.updateComplete;

    const chunkHost = view.querySelector<HTMLElement>('.source-doc-chunks')!;
    expect(typesetContainers.flat().includes(chunkHost)).to.equal(true);
  } finally {
    if (original === undefined) delete windowRef.MathJax;
    else windowRef.MathJax = original;
  }
});

it('typesets an answer once for each presentation, and still redraws its typed resources for any other change', async () => {
  const typeset: Element[] = [];
  const windowRef = window as {MathJax?: unknown};
  const original = windowRef.MathJax;
  windowRef.MathJax = {
    typesetPromise: (containers: Element[]) => {
      typeset.push(...containers);
      return Promise.resolve();
    },
  };
  const card = {
    url: 'https://example.com/film', title: 'A short film', description: '', site: 'Example', image: '',
  };
  const withCard = (words: string): AnswerPresentation => ({
    ...presentation,
    sources: Array.from({length: 3}, (_, index) => ({...presentation.sources[0], id: String(index + 1)})),
    parts: [{
      ...presentation.parts[0],
      html: `<p>${words} <a href="${card.url}">watch</a></p>`,
    }],
    linkCards: [card],
  });
  try {
    const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
    element.presentation = withCard('First answer');
    document.body.appendChild(element);
    await element.updateComplete;
    // One part, so one typeset for the presentation.
    expect(typeset).to.have.length(1);

    // Opening the References redraws the same presentation: the words are not typeset again.
    element.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click();
    await element.updateComplete;
    expect(element.querySelector('button[aria-expanded]')?.getAttribute('aria-expanded')).to.equal('true');
    expect(typeset, 'a redraw for the References typeset the answer again').to.have.length(1);

    // A language change redraws it too, and the typed resources speak the new language.
    expect(element.querySelector('[data-answer-link-card]')?.getAttribute('aria-label')).to.equal('Open A short film');
    try {
      await setLanguagePreference('zh');
      await element.updateComplete;
      expect(element.querySelector('[data-answer-link-card]')?.getAttribute('aria-label')).to.equal('打开 A short film');
      expect(typeset, 'a redraw for the language typeset the answer again').to.have.length(1);
    } finally {
      await setLanguagePreference('en');
    }

    // Another presentation is other words: they are typeset.
    element.presentation = withCard('Second answer');
    await element.updateComplete;
    expect(element.querySelector('.answer-rich-content')?.textContent).to.contain('Second answer');
    expect(typeset).to.have.length(2);
  } finally {
    if (original === undefined) delete windowRef.MathJax;
    else windowRef.MathJax = original;
  }
});

it('typesets an answer again at a redraw while a diagram in it is still its source, and not once it is drawn', async () => {
  const typeset: Element[] = [];
  const windowRef = window as {MathJax?: unknown};
  const original = windowRef.MathJax;
  windowRef.MathJax = {
    typesetPromise: (containers: Element[]) => {
      typeset.push(...containers);
      return Promise.resolve();
    },
  };
  try {
    const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
    element.presentation = {
      ...presentation,
      sources: Array.from({length: 3}, (_, index) => ({...presentation.sources[0], id: String(index + 1)})),
      parts: [{
        ...presentation.parts[0],
        html: '<p>See</p><pre class="mermaid-source" data-lang="mermaid"><code>graph TD;\nA--&gt;B;</code></pre>',
      }],
    };
    document.body.appendChild(element);
    await element.updateComplete;
    expect(typeset, 'typeset once for the presentation').to.have.length(1);

    // The Mermaid chunk has not come, or never will: the fence is still its source. A redraw for anything
    // else (here, opening the References) is another try, which is how a failed load is made good.
    expect(element.querySelector('pre.mermaid-source')).not.to.equal(null);
    element.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click();
    await element.updateComplete;
    expect(typeset, 'the redraw typeset it again').to.have.length(2);

    // Once nothing in the answer is a source, a redraw leaves what is typeset alone.
    element.querySelector('pre.mermaid-source')!.replaceWith(document.createElement('figure'));
    element.querySelector<HTMLButtonElement>('button[aria-expanded]')!.click();
    await element.updateComplete;
    expect(typeset, 'the next redraw did not').to.have.length(2);
  } finally {
    if (original === undefined) delete windowRef.MathJax;
    else windowRef.MathJax = original;
  }
});

function artifactWithAttestation(attestation: AnswerArtifact['attestation']): AnswerArtifact {
  return {
    resourceId: 'artifact-report',
    mediaType: 'text/markdown',
    label: 'Report',
    filename: 'report.md',
    byteSize: 20,
    digest: 'a'.repeat(64),
    presentation: 'markdown',
    status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/artifact-report',
    width: null,
    height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-report',
    downloadUrl: '/web/api/answer/run-1/artifacts/artifact-report?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/artifact-report/presentation',
    issue: null,
    attestation,
  };
}

/** One Artifact placed in an Answer as a card, drawn and read back the way a reader sees it. */
async function placedArtifact(
  attestation: AnswerArtifact['attestation'],
): Promise<AnswerPresentationElement> {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  const artifact = artifactWithAttestation(attestation);
  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact, evidenceImage: null, inline: false}],
    artifacts: [artifact],
  };
  document.body.appendChild(element);
  await element.updateComplete;
  return element;
}

/** The record's own visible line: the verdict and its one control, without the folded evidence. */
const recordLine = (element: AnswerPresentationElement): string =>
  element.querySelector('.answer-artifact-record > .artifact-attestation')
    ?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

/** What the record's disclosure holds: the reason or verifier, then the per-read lines. */
const recordEvidence = (element: AnswerPresentationElement): string =>
  element.querySelector('.answer-artifact-record-evidence')
    ?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

it('draws the Run\'s delivery record once under the files, and nothing when the server derived none', async () => {
  // No record: the cards are the cards they were before the record existed, and nothing reads as a
  // problem. `null` is "the server did not derive one", never "the Run loaded no Skill".
  const none = await placedArtifact(null);
  expect(none.querySelector('.answer-artifact-record')).to.equal(null);
  expect(none.querySelector('.artifact-attestation')).to.equal(null);
  expect(none.querySelector('.answer-artifact-card')?.textContent?.replace(/\s+/g, ' ').trim())
    .to.equal('Report report.md Open file');

  const attested = await placedArtifact({
    status: 'attested',
    skill: 'sealed-research-brief',
    verifier: 'scripts/validate_model.py',
    reason: null,
    basis: 'named_in_a_recorded_command',
    requiredReadsMissing: null,
  });
  // The record is the Answer's, not a line inside a card: the card names what it opens.
  expect(attested.querySelector('.answer-artifact-card .artifact-attestation')).to.equal(null);
  expect(attested.querySelector('.answer-artifact-record .artifact-attestation')?.getAttribute('data-attestation'))
    .to.equal('attested');
  // The verdict is the line under the files; what the claim rests on travels with it as evidence,
  // behind one disclosure, because the line has to stay a row.
  expect(recordLine(attested)).to.equal('Attested by sealed-research-brief Details');
  expect(recordEvidence(attested)).to.equal('verifier scripts/validate_model.py');
  expect(attested.querySelector<HTMLElement>('.answer-artifact-record-evidence')?.hidden,
    'the basis starts folded').to.equal(true);

  // The reason is the finding, but it is evidence too: "Not attested" stays the line, and the
  // server's own reason opens under it.
  const unattested = await placedArtifact({
    status: 'unattested', skill: null, verifier: null, reason: 'no_skill_loaded', basis: null,
    requiredReadsMissing: null,
  });
  expect(unattested.querySelector('.answer-artifact-record .artifact-attestation')?.getAttribute('data-attestation'))
    .to.equal('unattested');
  expect(recordLine(unattested)).to.equal('Not attested Details');
  expect(recordEvidence(unattested)).to.equal('the Run loaded no Agent Skill at all');

  // Records that cannot settle the question are not a finding: said plainly, never dressed as a
  // failure, with the reason the server gave.
  const unknown = await placedArtifact({
    status: 'unknown', skill: null, verifier: null, reason: 'records_pruned', basis: null,
    requiredReadsMissing: null,
  });
  expect(unknown.querySelector('.answer-artifact-record .artifact-attestation')?.getAttribute('data-attestation'))
    .to.equal('unknown');
  expect(recordLine(unknown)).to.equal('Attestation unknown Details');
  expect(recordEvidence(unknown)).to.contain('not the whole Run');
  expect(unknown.querySelector('.answer-artifact-record')?.closest('[role="alert"]')).to.equal(null);

  // A reason this client has not met is named as the server gave it, and takes nothing down.
  const future = await placedArtifact({
    status: 'unattested', skill: null, verifier: null, reason: 'a_reason_from_the_future', basis: null,
    requiredReadsMissing: null,
  });
  expect(recordEvidence(future)).to.contain('a_reason_from_the_future');
});

it('folds the reason and one neutral line for each declared read the Run did not make', async () => {
  const dashboard = {
    skill: 'interactive-html',
    paths: ['references/example-brief.html', 'references/example-dashboard.html'],
    when: 'before writing the fragment, the example nearest the request',
  };
  const fragment = {skill: 'interactive-html', paths: ['references/fragment.md'], when: 'before the build'};

  // The verdict is the line a reader sees; the reason and the per-read lines start folded behind one
  // control, so a card does not grow by one line per skipped read.
  const attested = await placedArtifact({
    status: 'attested', skill: 'interactive-html', verifier: null, reason: null,
    basis: 'named_in_a_recorded_command', requiredReadsMissing: [dashboard, fragment],
  });
  const record = attested.querySelector<HTMLElement>('.answer-artifact-record')!;
  expect(record.querySelector('.artifact-attestation')?.getAttribute('data-attestation'))
    .to.equal('attested');
  expect(recordLine(attested)).to.equal('Attested by interactive-html 2 reads skipped');
  const toggle = record.querySelector<HTMLButtonElement>('.answer-artifact-record-toggle')!;
  expect(toggle.textContent?.replace(/\s+/g, ' ').trim()).to.equal('2 reads skipped');
  expect(toggle.getAttribute('aria-expanded'), 'folded at rest').to.equal('false');
  const evidence = record.querySelector<HTMLElement>('.answer-artifact-record-evidence')!;
  expect(toggle.getAttribute('aria-controls')).to.equal(evidence.id);
  expect(evidence.hidden, 'the evidence is not on screen until asked for').to.equal(true);
  const lines = [...evidence.querySelectorAll('.artifact-attestation-unread')]
    .map((line) => line.textContent?.trim());
  // An `any_of` entry reads "a or b": reading either would have satisfied it.
  expect(lines).to.deep.equal([
    'not read: references/example-brief.html or references/example-dashboard.html '
      + '(before writing the fragment, the example nearest the request)',
    'not read: references/fragment.md (before the build)',
  ]);

  toggle.click();
  await attested.updateComplete;
  expect(toggle.getAttribute('aria-expanded')).to.equal('true');
  expect(evidence.hidden).to.equal(false);
  // It folds again on the second ask, so a reader is never left with the record open by accident.
  toggle.click();
  await attested.updateComplete;
  expect(toggle.getAttribute('aria-expanded'), 'folded again on the second ask').to.equal('false');
  expect(evidence.hidden).to.equal(true);
  // Evidence beside the verdict, never an alarm: neutral tone, no alert role anywhere above it.
  expect(record.closest('[role="alert"]')).to.equal(null);
  expect(record.querySelector('[role]')).to.equal(null);

  // A Skill that declares no verifier made no claim, so the lines are the whole record: no status,
  // no "Not attested", and no reason -- the count is what the record says.
  const claimless = await placedArtifact({
    status: 'unattested', skill: 'interactive-html', verifier: null,
    reason: 'skill_has_no_verifier_contract', basis: null, requiredReadsMissing: [dashboard],
  });
  expect(claimless.querySelector('.artifact-attestation-status')).to.equal(null);
  expect(recordLine(claimless)).to.equal('1 read skipped');
  const claimlessToggle = claimless.querySelector<HTMLButtonElement>('.answer-artifact-record-toggle')!;
  expect(claimlessToggle.textContent?.replace(/\s+/g, ' ').trim()).to.equal('1 read skipped');
  // No verdict leads the line, so the control sits where the words would rather than at its edge.
  expect(claimlessToggle.classList.contains('answer-artifact-record-toggle--solo')).to.equal(true);

  // Nothing missing, or nothing to say: no evidence to fold, and no record to draw.
  const complete = await placedArtifact({
    status: 'attested', skill: 'interactive-html', verifier: null, reason: null,
    basis: 'named_in_a_recorded_command', requiredReadsMissing: [],
  });
  expect(complete.querySelector('.artifact-attestation-unread')).to.equal(null);
  expect(complete.querySelector('.answer-artifact-record-toggle')).to.equal(null);
  expect(complete.querySelector('.answer-artifact-record-evidence')).to.equal(null);
  const claimlessAndComplete = await placedArtifact({
    status: 'unattested', skill: 'interactive-html', verifier: null,
    reason: 'skill_has_no_verifier_contract', basis: null, requiredReadsMissing: null,
  });
  expect(claimlessAndComplete.querySelector('.answer-artifact-record')).to.equal(null);
});

it('draws one record for a Run\'s files, not one per card', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  const record: AnswerArtifact['attestation'] = {
    status: 'unattested', skill: null, verifier: null, reason: 'verifier_reported_failures',
    basis: null,
    requiredReadsMissing: [{
      skill: 'sealed-research-brief',
      paths: ['references/reporting-methods.md'],
      when: 'before valuing',
    }],
  };
  const page = artifactWithAttestation(record);
  const ledger: AnswerArtifact = {
    ...artifactWithAttestation(record),
    resourceId: 'artifact-ledger',
    label: 'Polestar 模型账本 model-output.json',
    filename: 'polestar-ledger.zip',
  };
  element.presentation = {
    ...presentation,
    parts: [
      {type: 'artifact', text: '', html: '', artifact: page, evidenceImage: null, inline: false},
      {type: 'artifact', text: '', html: '', artifact: ledger, evidenceImage: null, inline: false},
    ],
    artifacts: [page, ledger],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  // One Run-level record for two cards, and neither card carries a copy of it.
  expect(element.querySelectorAll('.answer-artifact-record')).to.have.length(1);
  for (const card of element.querySelectorAll('.answer-artifact-card')) {
    expect(card.querySelector('.artifact-attestation')).to.equal(null);
  }
  // It follows the files it covers rather than sitting inside one of them.
  const parts = element.querySelector('.answer-parts')!;
  const drawn = element.querySelector('.answer-artifact-record')!;
  expect(parts.compareDocumentPosition(drawn) & Node.DOCUMENT_POSITION_FOLLOWING)
    .to.equal(Node.DOCUMENT_POSITION_FOLLOWING);
});

it('names a publication issue on an unavailable card, and invents no Run-level record for it', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  const description = 'The file was refused because it could not be sanitized safely.';
  const refused: AnswerArtifact = {
    ...artifactWithAttestation({
      status: 'unattested', skill: null, verifier: null, reason: 'no_skill_loaded', basis: null,
      requiredReadsMissing: null,
    }),
    status: 'unavailable',
    dataUrl: null,
    downloadUrl: null,
    presentationUrl: null,
    issue: {
      kind: 'unsafe_file',
      description,
      resourceId: 'artifact-report',
    },
  };
  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact: refused, evidenceImage: null, inline: false}],
    artifacts: [refused],
    // A publication failure is the Run's issue to report; it is not an attestation record.
    artifactOutcome: {
      status: 'partial',
      issues: [{kind: 'unsafe_file', description, resourceId: 'artifact-report'}],
    },
  };
  document.body.appendChild(element);
  await element.updateComplete;

  expect(element.querySelector('.answer-artifact-card')?.textContent).to.contain(description);
  // The delivery record speaks for the files the server handed over, and this one was not among
  // them: the card's own issue is the finding, and no record is invented to repeat it.
  expect(element.querySelector('.answer-artifact-card .artifact-attestation')).to.equal(null);
  expect(element.querySelector('.answer-artifact-record')).to.equal(null);
  // The Run still says that not everything it was asked to publish made it through.
  expect(element.querySelector('.artifact-publication-warning')?.textContent?.trim())
    .to.equal('Some requested files could not be published.');
});

it('draws one delivery record for a Run, with the attestation reason behind its disclosure', async () => {
  const element = await placedArtifact({
    status: 'unknown',
    skill: 'sealed-research-brief',
    verifier: 'scripts/validate_model.py',
    reason: 'arguments_truncated',
    basis: null,
    requiredReadsMissing: null,
  });

  // One Run-level record under the files, and the card that opens the file carries no copy of it.
  expect(element.querySelectorAll('.answer-artifact-record')).to.have.length(1);
  expect(element.querySelector('.answer-artifact-card .artifact-attestation')).to.equal(null);
  expect(
    element.querySelector('.answer-artifact-record .artifact-attestation')?.getAttribute('data-attestation'),
  ).to.equal('unknown');
  expect(recordLine(element)).to.equal('Attestation unknown Details');

  // The reason is evidence beside the verdict: folded at rest, named in the reader's words once the
  // disclosure is opened -- never the runtime's internal vocabulary.
  const evidence = element.querySelector<HTMLElement>('.answer-artifact-record-evidence')!;
  const toggle = element.querySelector<HTMLButtonElement>('.answer-artifact-record-toggle')!;
  expect(evidence.hidden, 'the reason starts folded').to.equal(true);
  toggle.click();
  await element.updateComplete;
  expect(evidence.hidden).to.equal(false);
  expect(recordEvidence(element)).to.equal(
    "part of the Run's record is missing, so whether the Skill's verifier ran cannot be confirmed",
  );
  const shown = element.textContent?.replace(/\s+/g, ' ') ?? '';
  expect(shown).to.not.contain('answer.generation.artifact_attestation');
  expect(shown).to.not.contain("runtime's own display bound");
});

it('carries the record once in the Answer when the Artifact is placed inline', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  const image: AnswerArtifact = {
    ...artifactWithAttestation({
      status: 'attested', skill: 'chart-craft', verifier: null,
      reason: null, basis: 'named_in_a_recorded_command', requiredReadsMissing: null,
    }),
    resourceId: 'artifact-chart',
    mediaType: 'image/png',
    label: 'Chart',
    filename: 'chart.png',
    presentation: 'image',
    width: 100,
    height: 100,
    dataUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  };
  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact: image, evidenceImage: null, inline: true}],
    artifacts: [image],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  // The figure keeps the picture and its caption; the record is drawn once for the Answer.
  expect(element.querySelector('.answer-artifact-image .artifact-attestation')).to.equal(null);
  expect(element.querySelectorAll('.answer-artifact-record')).to.have.length(1);
  expect(recordLine(element)).to.equal('Attested by chart-craft');
});

it('captions an inline image with the Model sentence and names the file in its alt', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  // Three places said one sentence: the alt, the caption, and the Artifact card. The caption is
  // where a sentence belongs, and the alt is where a broken image and a screen reader need the
  // file's own name -- so each carries one thing, and neither repeats the other.
  const description = '2026-10-15 Photo of the Day：Rotation Station（Garret Suhrie）';
  const image: AnswerArtifact = {
    ...artifactWithAttestation(null),
    resourceId: 'artifact-photo',
    mediaType: 'image/jpeg',
    label: description,
    filename: 'rotation-station.jpg',
    presentation: 'image',
    width: 3072,
    height: 2048,
    dataUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  };
  element.presentation = {
    ...presentation,
    parts: [{type: 'artifact', text: '', html: '', artifact: image, evidenceImage: null, inline: true}],
    artifacts: [image],
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const picture = element.querySelector('.answer-artifact-image img');
  expect(picture?.getAttribute('alt')).to.equal('rotation-station.jpg');
  const caption = element.querySelector('.answer-artifact-image figcaption')?.textContent ?? '';
  expect(caption).to.contain(description);
  expect(caption).to.contain('3072×2048');
  // The filename names the bytes, so it is not repeated back as visible copy under the picture.
  expect(caption).to.not.contain('rotation-station.jpg');
  // Nothing repeats the sentence visually: the control's accessible name is where a reader who
  // navigates by control rather than by picture still finds it.
  const control = element.querySelector('.answer-artifact-image button');
  expect(control?.getAttribute('title')).to.equal(null);
  expect(control?.getAttribute('aria-label')).to.contain(description);
  // A Skill that declares no verifier publishes no record, so no record is drawn for this Answer.
  expect(element.querySelector('.answer-artifact-image .artifact-attestation')).to.equal(null);
  expect(element.querySelector('.answer-artifact-record')).to.equal(null);
});


it('answers only the Run its trace was opened from', async () => {
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = answerWithReferences(['1', '2', '3']);
  element.runId = 'run-1';
  document.body.appendChild(element);
  await element.updateComplete;

  askForReference('1', 'run-2');
  await element.updateComplete;
  expect(litReferences(element), 'another Run\'s request is not this Answer\'s').to.deep.equal([]);

  askForReference('1', 'run-1');
  await element.updateComplete;
  expect(litReferences(element)).to.deep.equal(['1']);

  // A request that names no Run is the older contract, and every Answer still answers it.
  element.runId = null;
  askForReference('2');
  await element.updateComplete;
  expect(litReferences(element)).to.deep.equal(['2']);
});


it('names the declaring Skill when a skipped read belongs to another loaded Skill', async () => {
  const other = {
    skill: 'interactive-html',
    paths: ['references/fragment.md'],
    when: 'before writing or restructuring the fragment',
  };
  const attested = await placedArtifact({
    status: 'attested', skill: 'sealed-research-brief', verifier: 'scripts/validate_model.py',
    reason: null, basis: 'named_in_a_recorded_command', requiredReadsMissing: [other],
  });
  const evidence = attested.querySelector<HTMLElement>('.answer-artifact-record-evidence')!;
  const lines = [...evidence.querySelectorAll('.artifact-attestation-unread')]
    .map((line) => line.textContent?.trim());
  expect(lines).to.deep.equal([
    'not read: references/fragment.md (before writing or restructuring the fragment; '
      + 'declared by interactive-html)',
  ]);
});

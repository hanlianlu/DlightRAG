// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
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
  evidenceImages: [],
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
    sources: Array.from({length: 5}, (_, index) => ({
      ...presentation.sources[0],
      id: String(index + 1),
      title: `Reference ${index + 1}`,
    })),
  };
  document.body.appendChild(element);
  await element.updateComplete;

  const toggle = element.querySelector<HTMLButtonElement>('.answer-references-show-all')!;
  expect(toggle.textContent?.trim()).to.equal('Show all 5');
  expect(toggle.getAttribute('aria-expanded')).to.equal('false');
  toggle.click();
  await element.updateComplete;

  const collapse = element.querySelector<HTMLButtonElement>('.answer-references-show-all')!;
  expect(element.querySelector('.answer-reference-list')?.classList.contains('expanded')).to.equal(true);
  expect(collapse.textContent?.trim()).to.equal('Show fewer');
  expect(collapse.getAttribute('aria-expanded')).to.equal('true');
  collapse.click();
  await element.updateComplete;
  expect(element.querySelector('.answer-reference-list')?.classList.contains('expanded')).to.equal(false);
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

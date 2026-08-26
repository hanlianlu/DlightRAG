// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
import './answer_presentation.ts';
import './inspector_sources.ts';
import type {AnswerPresentationElement} from './answer_presentation.ts';
import type {DlImageLightbox} from './image_lightbox.ts';
import './image_lightbox.ts';
import type {DlInspectorSources} from './inspector_sources.ts';

const presentation: AnswerPresentation = {
  answer_text: 'Safe [1].',
  parts: [{
    type: 'markdown',
    text: 'Safe [1].',
    html: '<p>Safe <cite class="citation-badge" data-ref="1">1</cite></p><script>x()</script>',
    artifact: null,
    evidence_image: null,
    inline: false,
  }],
  sources: [
    {
      id: '1',
      title: '<img src=x>',
      source_url: 'https://example.com/report',
      download_url: 'https://evil.example.com/steal',
      chunks: [
        {
          chunk_idx: 2,
          page_number: 4,
          content_html: '<p>Evidence</p><script>x()</script>',
          image_url: null,
          thumbnail_url: null,
        },
      ],
    },
  ],
  evidence_images: [],
  artifacts: [],
  artifact_outcome: {status: 'complete', issues: []},
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
  element.addEventListener('answer-source-open', (event) => {
    const detail = (event as CustomEvent).detail;
    referenceId = detail.referenceId;
    eventPresentation = detail.presentation;
  });
  element.querySelector<HTMLElement>('.citation-badge')?.click();
  expect(referenceId).to.equal('1');
  expect(eventPresentation).to.equal(element.presentation);
});

it('renders Artifact intent and semantic Visual Evidence in approved order', async () => {
  const artifact = {
    resource_id: 'artifact-report',
    role: 'primary_report' as const,
    media_type: 'text/markdown',
    label: 'Quarterly report',
    filename: 'report.md',
    byte_size: 20,
    digest: 'a'.repeat(64),
    presentation: 'markdown' as const,
    status: 'available' as const,
    uri: 'dlightrag://answer/run-1/artifacts/artifact-report',
    width: null,
    height: null,
    data_url: '/web/api/answer/run-1/artifacts/artifact-report',
    download_url: '/web/api/answer/run-1/artifacts/artifact-report?download=1',
    presentation_url: '/web/api/answer/run-1/artifacts/artifact-report/presentation',
    issue: null,
  };
  const value: AnswerPresentation = {
    ...presentation,
    parts: [
      presentation.parts[0],
      {type: 'artifact', text: '', html: '', artifact, evidence_image: null, inline: false},
    ],
    artifacts: [artifact],
    evidence_images: [{
      id: 'image-1', chunk_id: 'chunk-1', source_ref: '1',
      url: '/web/api/images/default/chunk-1?size=full',
      thumbnail_url: '/web/api/images/default/chunk-1?size=thumb',
      label: 'Chart', answer_image_sent: true,
    }],
  };
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = value;
  document.body.appendChild(element);
  await element.updateComplete;

  let opened = '';
  let imageSource = '';
  element.addEventListener('artifact-open', (event) => {
    opened = (event as CustomEvent).detail.artifact.resource_id;
  });
  element.addEventListener('dl-image-open', (event) => {
    imageSource = event.detail.src;
  });
  element.querySelector<HTMLButtonElement>('.answer-artifact-card .ui-btn')?.click();
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
    evidence_images: [
      {
        id: 'image-1', chunk_id: 'chunk-1', source_ref: '1',
        url: '/web/api/images/default/chunk-1?size=full', thumbnail_url: '',
        label: 'First', answer_image_sent: true,
      },
      {
        id: 'image-2', chunk_id: 'chunk-2', source_ref: '1',
        url: '/web/api/images/default/chunk-2?size=full', thumbnail_url: '',
        label: 'Second', answer_image_sent: true,
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

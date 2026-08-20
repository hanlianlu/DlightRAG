// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerPresentation} from '../api/conversations.ts';
import './answer_presentation.ts';
import './source_panel_view.ts';
import type {AnswerPresentationElement} from './answer_presentation.ts';
import type {SourcePanelView} from './source_panel_view.ts';

const presentation: AnswerPresentation = {
  answer_text: 'Safe [1].',
  answer_html: '<p>Safe <cite class="citation-badge" data-ref="1">1</cite></p><script>x()</script>',
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
  answer_images: [],
  primary_report: null,
};

afterEach(() => { document.body.replaceChildren(); });

it('sanitizes rich answer HTML while Lit escapes structured references', async () => {
  const element = document.createElement('answer-presentation') as AnswerPresentationElement;
  element.presentation = presentation;
  document.body.appendChild(element);
  await element.updateComplete;

  expect(element.querySelector('.answer-rich-content')?.textContent).to.contain('Safe 1');
  expect(element.querySelector('script')).to.equal(null);
  expect(element.querySelector('.answer-ref-title')?.textContent).to.equal('<img src=x>');
  expect(element.querySelector('.answer-ref-title img')).to.equal(null);
});

it('renders sanitized source chunks and rejects cross-origin download links', async () => {
  const view = document.createElement('source-panel-view') as SourcePanelView;
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

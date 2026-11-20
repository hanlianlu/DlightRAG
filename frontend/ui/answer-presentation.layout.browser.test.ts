// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Answer's References and typed resources as the shipped stylesheets lay them out (the product-styles
 *  page of the runner). */

import {expect} from '@esm-bundle/chai';
import {resetMouse, sendMouse, setViewport} from '@web/test-runner-commands';
import type {AnswerArtifact, AnswerPresentation, LinkCard} from '../api/conversations.ts';
import answerStyles from '../styles/answer-presentation.module.css';
import chatStyles from '../styles/chat.module.css';
import type {AnswerPresentationElement} from './answer-presentation.ts';
import './answer-presentation.ts';

const originalViewport = {width: window.innerWidth, height: window.innerHeight};

function presentation(sourceCount: number, html = '<p>Revenue grew.</p>'): AnswerPresentation {
  return {
    answerText: 'Revenue grew.',
    parts: [{type: 'markdown', text: 'Revenue grew.', html, artifact: null, evidenceImage: null, inline: false}],
    sources: Array.from({length: sourceCount}, (_, index) => ({
      id: String(index + 1), title: `Source number ${index + 1}`, sourceUrl: null, downloadUrl: null, chunks: [],
    })),
    evidenceImages: [], linkCards: [], artifacts: [],
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

afterEach(async () => {
  document.body.replaceChildren();
  await setViewport(originalViewport);
  await resetMouse();
});

const cls = (name: string): string => `.${answerStyles[name]}`;
const shown = (node: Element): boolean => node.getClientRects().length > 0;
const rowsOf = (element: Element): HTMLElement[] => [...element.querySelectorAll<HTMLElement>(cls('answer-ref-item'))];

for (const width of [390, 800, 1440]) {
  it(`folds the References to their first two rows, and unfolds them all, at ${width}px`, async () => {
    await setViewport({width, height: 900});
    const element = await mount(presentation(5));
    const toggle = element.querySelector<HTMLButtonElement>(cls('answer-references-toggle'))!;

    expect(rowsOf(element).map(shown), 'two rows at rest').to.deep.equal([true, true, false, false, false]);
    expect(toggle.getAttribute('aria-expanded')).to.equal('false');

    toggle.click();
    await element.updateComplete;
    expect(rowsOf(element).every(shown), 'every row once unfolded').to.equal(true);
    expect(toggle.getAttribute('aria-expanded')).to.equal('true');

    toggle.click();
    await element.updateComplete;
    expect(rowsOf(element).filter(shown)).to.have.length(2);
  });
}

describe('the References caption that folds the list', () => {
  const typography = (node: Element) => {
    const style = getComputedStyle(node);
    return {
      fontSize: style.fontSize, fontWeight: style.fontWeight, color: style.color,
      textTransform: style.textTransform, letterSpacing: style.letterSpacing,
    };
  };

  it('reads as the caption of a list that cannot fold, and its count is an aside in the caption\'s size', async () => {
    const plain = await mount(presentation(2));
    const folding = await mount(presentation(5));
    const caption = plain.querySelector(cls('answer-section-title'))!;
    const toggle = folding.querySelector<HTMLButtonElement>(cls('answer-references-toggle'))!;
    const count = toggle.querySelector<HTMLElement>(cls('answer-references-count'))!;

    expect(typography(toggle), 'the bar states the caption\'s typography again').to.deep.equal(typography(caption));
    expect(getComputedStyle(toggle).textTransform).to.equal('uppercase');
    expect(getComputedStyle(count).textTransform, 'the count is not in caps').to.equal('none');
    expect(getComputedStyle(count).fontWeight).to.equal('400');
    expect(getComputedStyle(count).fontSize).to.equal(getComputedStyle(toggle).fontSize);
  });

  it('fills the caption\'s own line, so the bar the reader lights is the line, and shows a pointer on it', async () => {
    const element = await mount(presentation(5));
    const toggle = element.querySelector<HTMLButtonElement>(cls('answer-references-toggle'))!;
    const title = toggle.parentElement!;

    expect(title.tagName).to.equal('H3');
    expect(toggle.getBoundingClientRect().width).to.be.closeTo(title.clientWidth, 0.5);
    expect(getComputedStyle(toggle).cursor).to.equal('pointer');
  });

  it('lights under the pointer and rings on keyboard focus, as the list\'s own rows do', async () => {
    const element = await mount(presentation(5));
    const toggle = element.querySelector<HTMLButtonElement>(cls('answer-references-toggle'))!;
    const count = toggle.querySelector<HTMLElement>(cls('answer-references-count'))!;
    const rowBackground = (): string => getComputedStyle(rowsOf(element)[0]!).backgroundColor;
    const rest = {background: getComputedStyle(toggle).backgroundColor, count: getComputedStyle(count).color};
    expect(rest.background).to.equal('rgba(0, 0, 0, 0)');

    const box = toggle.getBoundingClientRect();
    await sendMouse({type: 'move', position: [Math.round(box.left + 4), Math.round(box.top + box.height / 2)]});
    await Promise.all(toggle.getAnimations({subtree: true}).map((animation) => animation.finished));
    expect(getComputedStyle(toggle).backgroundColor, 'tinted under the pointer').to.not.equal(rest.background);
    expect(getComputedStyle(count).color, 'the count wakes with it').to.not.equal(rest.count);

    // The lit bar is the lit row the list draws under the pointer.
    const row = rowsOf(element)[0]!;
    const rowBox = row.getBoundingClientRect();
    await sendMouse({type: 'move', position: [Math.round(rowBox.left + 4), Math.round(rowBox.top + rowBox.height / 2)]});
    await Promise.all(row.getAnimations().map((animation) => animation.finished));
    expect(getComputedStyle(toggle).backgroundColor, 'untinted once the pointer leaves').to.equal(rest.background);
    expect(rowBackground()).to.not.equal('rgba(0, 0, 0, 0)');

    toggle.focus();
    expect(toggle.matches(':focus-visible')).to.equal(true);
    const ring = getComputedStyle(toggle);
    const probe = document.createElement('div');
    probe.style.color = 'var(--color-control-ring)';
    document.body.append(probe);
    expect(ring.outlineStyle).to.equal('solid');
    expect(ring.outlineWidth).to.equal('2px');
    expect(ring.outlineColor).to.equal(getComputedStyle(probe).color);
  });
});

it('keeps every reference title on one gutter, however wide the ids grow', async () => {
  const element = await mount(presentation(12));
  element.querySelector<HTMLButtonElement>(cls('answer-references-toggle'))!.click();
  await element.updateComplete;
  const starts = rowsOf(element).map((row) => (
    row.querySelector(cls('answer-ref-title'))!.getBoundingClientRect().left
  ));

  expect(new Set(starts.map((left) => Math.round(left * 10))).size, starts.join(', ')).to.equal(1);
  expect(rowsOf(element)[0]!.querySelector(cls('answer-ref-id'))!.getBoundingClientRect().width)
    .to.be.at.least(24);
});

it('isolates typed thumbnails and cards from prose image and link styles', async () => {
  const image = {
    id: 'shot', chunkId: '', sourceRef: '', label: 'Shot', answerImageSent: true, thumbnailUrl: '',
    url: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  };
  const card: LinkCard = {
    url: 'https://example.com/x', title: 'Film', description: 'Description', site: 'Example',
    image: 'https://example.com/cover.png',
  };
  const value = presentation(0, '<p><span class="answer-resource-slot-0"></span> <a href="https://example.com/x">film</a></p>');
  value.parts.push({type: 'evidence_image', text: '', html: '', artifact: null, evidenceImage: image, inline: true, slot: 0});
  value.linkCards = [card];
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
  probe.style.color = 'var(--color-text-primary)';
  element.append(probe);
  expect(getComputedStyle(cardLink).color).to.equal(getComputedStyle(probe).color);
});

/** The two files the reported Answer delivered, whose Run skipped the same two declared reads. The
 *  record is Run-level, and each card's material is longer than a phone's Answer column: the shape
 *  that decides whether a card's action stays a control and whether the record can be read where
 *  it is drawn. */
function attestedArtifact(index: number): AnswerArtifact {
  const base = `/web/api/answer/run-1/artifacts/artifact-${index}`;
  return {
    resourceId: `artifact-${index}`,
    mediaType: index === 0 ? 'text/html' : 'application/json',
    label: index === 0
      ? 'Polestar (PSNY) 投资性研究页 — 2026-10-19'
      : 'Polestar 模型账本 model-output.json',
    filename: index === 0 ? 'polestar.html' : 'polestar-ledger.zip',
    byteSize: 512,
    digest: 'a'.repeat(64),
    presentation: 'markdown',
    status: 'available',
    uri: `dlightrag://answer/run-1/artifacts/artifact-${index}`,
    width: null,
    height: null,
    dataUrl: base,
    downloadUrl: `${base}?download=1`,
    presentationUrl: `${base}/presentation`,
    issue: null,
    attestation: {
      status: 'unattested',
      skill: null,
      verifier: null,
      reason: 'verifier_reported_failures',
      basis: null,
      requiredReadsMissing: [
        {
          skill: 'sealed-research-brief',
          paths: ['references/reporting-methods.md'],
          when: 'before valuing — the WACC dossier, the terminal value, the price inversion',
        },
        {
          skill: 'sealed-research-brief',
          paths: ['references/audit-checklist.md'],
          when: 'before delivery — the rule table and the instant-fail list',
        },
      ],
    },
  };
}

async function mountArtifactCards(width: number): Promise<AnswerPresentationElement> {
  // The answer column the cards are drawn in: `--layout-content-width` keeps the real one under
  // 880px at every viewport, and `.aiMessageContent` is where the cards inherit wrapping from.
  const column = document.createElement('div');
  column.className = chatStyles.aiMessageContent!;
  column.style.width = `${width}px`;
  const artifacts = [attestedArtifact(0), attestedArtifact(1)];
  const value: AnswerPresentation = {
    ...presentation(0),
    parts: artifacts.map((artifact) => ({
      type: 'artifact' as const, text: '', html: '', artifact, evidenceImage: null, inline: false,
    })),
    artifacts,
  };
  const element = document.createElement('dl-answer-presentation') as AnswerPresentationElement;
  element.presentation = value;
  column.append(element);
  document.body.append(column);
  await element.updateComplete;
  return element;
}

/** What one element would lay out at, free of any column: the width its own content asks for. */
function ownWidth(element: HTMLElement): number {
  const free = element.cloneNode(true) as HTMLElement;
  free.style.cssText = 'position: absolute; width: max-content;';
  document.body.append(free);
  const width = free.getBoundingClientRect().width;
  free.remove();
  return width;
}

/** The rects one element's own text laid out, one per line. */
function labelLines(element: HTMLElement): DOMRect[] {
  const range = document.createRange();
  range.selectNodeContents(element);
  return [...range.getClientRects()].filter((rect) => rect.width > 0 && rect.height > 0);
}

for (const [viewportWidth, columnWidth] of [[800, 390], [1600, 880]] as const) {
  it(`keeps each card's action one line and the record one row in a ${columnWidth}px column`, async () => {
    await setViewport({width: viewportWidth, height: 900});
    const element = await mountArtifactCards(columnWidth);
    const columnBox = element.parentElement!.getBoundingClientRect();

    // The action never gives its width to anything beside it: a fixed-height control cannot host
    // a wrapped label, so it keeps the width the one line asks for and the card wraps the rest.
    for (const card of element.querySelectorAll<HTMLElement>('.answer-artifact-card')) {
      const button = card.querySelector<HTMLButtonElement>('button.dl-btn')!;
      expect(button.getBoundingClientRect().width, 'the control keeps its label\'s width')
        .to.be.at.least(ownWidth(button) - 0.5);
      expect(labelLines(button).length, 'the label reads on one line').to.equal(1);
      expect(button.scrollWidth, 'the label fits the control').to.be.at.most(button.clientWidth + 1);
      expect(button.scrollHeight, 'the control is not shorter than its label')
        .to.be.at.most(button.clientHeight + 1);
      expect(card.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
    }

    // The record is one Run-level line under the files, and its evidence is folded: the row reads
    // at the column's width, with the disclosure on screen rather than pushed out of it.
    expect(element.querySelectorAll('.answer-artifact-record')).to.have.length(1);
    const record = element.querySelector<HTMLElement>('.answer-artifact-record')!;
    expect(record.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
    const toggle = record.querySelector<HTMLButtonElement>('.answer-artifact-record-toggle')!;
    expect(toggle.scrollWidth, 'the disclosure\'s words fit the control')
      .to.be.at.most(toggle.clientWidth + 1);
    expect(toggle.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
    const evidence = record.querySelector<HTMLElement>('.answer-artifact-record-evidence')!;
    expect(evidence.getClientRects().length, 'the evidence starts folded').to.equal(0);
  });

  it(`opens the record's folded evidence inside a ${columnWidth}px column`, async () => {
    await setViewport({width: viewportWidth, height: 900});
    const element = await mountArtifactCards(columnWidth);
    const columnBox = element.parentElement!.getBoundingClientRect();
    const record = element.querySelector<HTMLElement>('.answer-artifact-record')!;
    const toggle = record.querySelector<HTMLButtonElement>('.answer-artifact-record-toggle')!;
    const evidence = record.querySelector<HTMLElement>('.answer-artifact-record-evidence')!;
    const reason = evidence.querySelector<HTMLElement>('.answer-artifact-record-reason')!;
    const lines = [...evidence.querySelectorAll<HTMLElement>('.artifact-attestation-unread')];
    expect(reason.textContent, 'the reason folds with the lines').to.contain('reported failures');
    expect(lines).to.have.length(2);

    toggle.click();
    await element.updateComplete;
    expect(toggle.getAttribute('aria-expanded')).to.equal('true');
    expect(evidence.getClientRects().length, 'the evidence is drawn once asked for').to.be.above(0);
    expect(evidence.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
    expect(reason.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
    for (const line of lines) {
      // Each line wraps at the column's words rather than running out of it.
      expect(line.getBoundingClientRect().right).to.be.at.most(columnBox.right + 0.5);
      expect(line.scrollWidth).to.be.at.most(line.clientWidth + 1);
    }
  });
}

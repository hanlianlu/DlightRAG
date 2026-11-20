// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import type {AnswerArtifact} from '../api/conversations.ts';
import {defineDesignSystemElements} from '../design-system/index.ts';
import './artifact-canvas.ts';

defineDesignSystemElements();

import type {DlActiveArtifactFrame} from './active-artifact-frame.ts';
import type {DlArtifactCanvas} from './artifact-canvas.ts';
import './active-artifact-frame.ts';

const originalFetch = window.fetch;
const originalMatchMedia = window.matchMedia;

function desktopMedia(query: string): MediaQueryList {
  return {
    matches: false,
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  };
}

function htmlArtifact(): AnswerArtifact {
  return {
    resourceId: 'artifact-html',
    mediaType: 'text/html',
    label: 'Interactive report',
    filename: 'report.html',
    byteSize: 42,
    digest: 'a'.repeat(64),
    presentation: 'html',
    status: 'available',
    uri: 'dlightrag://answer/run-1/artifacts/artifact-html',
    width: null,
    height: null,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-html',
    downloadUrl: '/web/api/answer/run-1/artifacts/artifact-html?download=1',
    presentationUrl: '/web/api/answer/run-1/artifacts/artifact-html/presentation',
    issue: null,
    // Most Artifacts carry no derived record, which is not a verdict.
    attestation: null,
  };
}

function artifact(id: string, label: string, presentation: 'markdown' | 'html'): AnswerArtifact {
  const extension = presentation === 'markdown' ? 'md' : 'html';
  const mediaType = presentation === 'markdown' ? 'text/markdown' : 'text/html';
  return {
    ...htmlArtifact(),
    resourceId: id,
    mediaType: mediaType,
    label,
    filename: `${id}.${extension}`,
    presentation,
    uri: `dlightrag://answer/run-1/artifacts/${id}`,
    dataUrl: `/web/api/answer/run-1/artifacts/${id}`,
    downloadUrl: `/web/api/answer/run-1/artifacts/${id}?download=1`,
    presentationUrl: `/web/api/answer/run-1/artifacts/${id}/presentation`,
  };
}

// The canvas parses the presentation response through the api schema, so the
// stub must serve the snake_case wire shape.
function markdownPresentation(text: string): Record<string, unknown> {
  return {
    answer_text: text,
    parts: [{
      type: 'markdown', text, html: `<p>${text}</p>`, artifact: null,
      evidence_image: null, inline: false,
    }],
    sources: [],
    evidence_images: [],
    artifacts: [],
    artifact_outcome: {status: 'complete', issues: []},
  };
}

afterEach(() => {
  window.fetch = originalFetch;
  window.matchMedia = originalMatchMedia;
  document.body.replaceChildren();
  document.body.className = '';
});

it('ignores a previous Markdown response that finishes decoding after an Artifact switch', async () => {
  let finishStaleDecode!: (value: Record<string, unknown>) => void;
  let markDecodeStarted!: () => void;
  const staleDecode = new Promise<Record<string, unknown>>((resolve) => { finishStaleDecode = resolve; });
  const decodeStarted = new Promise<void>((resolve) => { markDecodeStarted = resolve; });
  const currentPresentation = markdownPresentation('Current report');
  window.fetch = async (input) => {
    if (String(input).includes('/stale-markdown/')) {
      return {
        ok: true,
        json: async () => {
          markDecodeStarted();
          return staleDecode;
        },
      } as Response;
    }
    return new Response(JSON.stringify(currentPresentation));
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  const staleOpen = canvas.open(artifact('stale-markdown', 'Stale report', 'markdown'));
  await decodeStarted;
  await canvas.open(artifact('current-markdown', 'Current report', 'markdown'));
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current report');
  expect(canvas.querySelector<HTMLAnchorElement>('[download]')?.getAttribute('href'))
    .to.equal(`${window.location.origin}/web/api/answer/run-1/artifacts/current-markdown?download=1`);
  expect(canvas.querySelector('.answer-rich-content')?.textContent).to.equal('Current report');

  finishStaleDecode(markdownPresentation('Stale report'));
  await staleOpen;
  await canvas.updateComplete;
  await canvas.querySelector('dl-answer-presentation')?.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current report');
  expect(canvas.querySelector('.answer-rich-content')?.textContent).to.equal('Current report');
  expect(canvas.textContent).not.to.contain('Stale report');
});

it('ignores previous HTML text that finishes decoding after an Artifact switch', async () => {
  let finishStaleDecode!: (value: string) => void;
  let markDecodeStarted!: () => void;
  const staleDecode = new Promise<string>((resolve) => { finishStaleDecode = resolve; });
  const decodeStarted = new Promise<void>((resolve) => { markDecodeStarted = resolve; });
  window.fetch = async (input) => {
    if (String(input).includes('/stale-html')) {
      return {
        ok: true,
        text: async () => {
          markDecodeStarted();
          return staleDecode;
        },
      } as Response;
    }
    return new Response('<!doctype html><html><body>Current HTML</body></html>');
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = false;
  document.body.appendChild(canvas);

  const staleOpen = canvas.open(artifact('stale-html', 'Stale HTML', 'html'));
  await decodeStarted;
  await canvas.open(artifact('current-html', 'Current HTML', 'html'));
  await canvas.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current HTML');
  expect(canvas.querySelector<HTMLAnchorElement>('[download]')?.getAttribute('href'))
    .to.equal(`${window.location.origin}/web/api/answer/run-1/artifacts/current-html?download=1`);
  const frame = canvas.querySelector<DlActiveArtifactFrame>('dl-active-artifact-frame');
  expect(frame?.source).to.contain('Current HTML');

  finishStaleDecode('<!doctype html><html><body>Stale HTML</body></html>');
  await staleOpen;
  await canvas.updateComplete;

  expect(canvas.querySelector('.artifact-canvas-title')?.textContent).to.equal('Current HTML');
  expect(frame?.source).to.contain('Current HTML');
  expect(frame?.source).not.to.contain('Stale HTML');
});

it('closes on Escape, and leaves the Escape that cancels an IME composition to the composition', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);
  await canvas.open(htmlArtifact());
  await canvas.updateComplete;
  const press = (init: KeyboardEventInit): KeyboardEvent => {
    const event = new KeyboardEvent('keydown', {key: 'Escape', bubbles: true, cancelable: true, ...init});
    canvas.querySelector<HTMLElement>('[data-action="close"]')!.dispatchEvent(event);
    return event;
  };

  // Chromium marks the key of a composition; WebKit sends it after it ended, as keyCode 229.
  for (const init of [{isComposing: true}, {keyCode: 229}]) {
    expect(press(init).defaultPrevented, 'the Escape of a composition was taken').to.equal(false);
    expect(canvas.classList.contains('open'), 'the Canvas closed under a composition').to.equal(true);
  }
  expect(press({}).defaultPrevented).to.equal(true);
  expect(canvas.classList.contains('open')).to.equal(false);
});

it('opens the interactive frame directly without a consent step', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  const returnFocus = document.createElement('button');
  document.body.appendChild(returnFocus);
  returnFocus.focus();
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = true;
  document.body.appendChild(canvas);

  await canvas.open(htmlArtifact(), returnFocus);
  await canvas.updateComplete;

  expect(canvas.classList.contains('open')).to.equal(true);
  expect(canvas.inert).to.equal(false);
  expect(canvas.getAttribute('role')).to.equal('dialog');
  expect(canvas.getAttribute('aria-labelledby')).to.equal('artifact-canvas-title');
  expect(canvas.querySelector('#artifact-canvas-title')?.textContent?.trim())
    .to.equal('Interactive report');
  const layoutGroup = canvas.querySelector<HTMLElement>('.artifact-canvas-layout-actions')!;
  expect(layoutGroup.getAttribute('role')).to.equal('group');
  expect(layoutGroup.getAttribute('aria-labelledby')).to.equal('artifact-canvas-title');
  expect(canvas.getAttribute('aria-modal')).to.equal(null);
  expect(canvas.textContent).not.to.contain('Open interactive report');
  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(true);

  frame?.dispatchEvent(new CustomEvent('dl-artifact-frame-escape', {bubbles: true, composed: true}));
  await canvas.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(canvas.classList.contains('open')).to.equal(false);
  expect(canvas.inert).to.equal(true);
  expect(canvas.getAttribute('aria-hidden')).to.equal('true');
  expect(canvas.hasAttribute('aria-labelledby')).to.equal(false);
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(document.activeElement).to.equal(returnFocus);

  await canvas.open(htmlArtifact(), returnFocus);
  canvas.querySelector<HTMLButtonElement>('[aria-pressed][type="button"]')?.click();
  await canvas.updateComplete;
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  canvas.close();
  await canvas.updateComplete;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  expect(canvas.querySelector('dl-active-artifact-frame')).to.equal(null);
  expect(document.activeElement).to.equal(returnFocus);
});

it('publishes modal, overlay, and wide predicates for Shell composition', async () => {
  window.matchMedia = desktopMedia;
  window.fetch = async () => new Response('<!doctype html><html><body>Report</body></html>');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  const states: Array<{open: boolean; modal: boolean; overlay: boolean; wide: boolean}> = [];
  canvas.addEventListener('dl-artifact-canvas-state-change', (event) => {
    states.push(event.detail);
  });
  document.body.append(canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  // HTML suggests wide: docked expansion, not a modal, no overlay.
  expect(canvas.layout).to.equal('wide');
  expect(canvas.getAttribute('aria-modal')).to.equal(null);
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: true});

  // Opening Sources preserves a docked layout that can coexist with Inspector.
  canvas.prepareForInspector();
  await canvas.updateComplete;
  expect(canvas.layout).to.equal('wide');
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: true});

  // Fullscreen is the modal; focus trapping applies there.
  const layoutButtons = Array.from(canvas.querySelectorAll<HTMLButtonElement>(
    '.artifact-canvas-layout-actions button',
  ));
  const fullscreenButton = layoutButtons
    .find((button) => button.textContent?.trim() === 'Fullscreen');
  fullscreenButton?.click();
  await canvas.updateComplete;
  expect(canvas.getAttribute('aria-modal')).to.equal('true');
  expect(states.at(-1)).to.deep.equal({open: true, modal: true, overlay: true, wide: false});
  const focusable = Array.from(canvas.querySelectorAll<HTMLElement>(
    'button:not([disabled]), dl-icon-button:not([disabled]), a[href]',
  ));
  focusable.at(-1)?.focus();
  focusable.at(-1)?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  expect(document.activeElement).to.equal(focusable[0]);

  canvas.prepareForInspector();
  await canvas.updateComplete;
  expect(canvas.layout).to.equal('wide');
  expect(canvas.hasAttribute('aria-modal')).to.equal(false);
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: true});

  const sideButton = layoutButtons.find((button) => button.textContent?.trim() === 'Side');
  sideButton?.click();
  canvas.prepareForInspector();
  await canvas.updateComplete;
  expect(canvas.layout).to.equal('side');
  expect(states.at(-1)).to.deep.equal({open: true, modal: false, overlay: false, wide: false});
});

it('reopening the Artifact it drew is not a second read, and a new digest is', async () => {
  // What the Canvas holds is identified by the digest, so an edge that is slow
  // or failing cannot turn a second open of the same Artifact into a failure.
  window.matchMedia = desktopMedia;
  const reads: string[] = [];
  window.fetch = async (input) => {
    reads.push(String(input));
    return new Response('<!doctype html><html><body>Report</body></html>');
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;
  canvas.close();
  await canvas.updateComplete;
  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  expect(reads.length).to.equal(1);
  const frame = canvas.querySelector<DlActiveArtifactFrame>('dl-active-artifact-frame');
  expect(frame?.source).to.contain('Report');
  expect(canvas.querySelector('.artifact-error')).to.equal(null);

  // The same resource published again is a new digest, and therefore new bytes.
  await canvas.open({...htmlArtifact(), digest: 'b'.repeat(64)});
  await canvas.updateComplete;
  expect(reads.length).to.equal(2);
});

it('says which failure it met: a server answer, an unreachable server, or bytes it cannot vouch for', async () => {
  window.matchMedia = desktopMedia;
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  // A refused read names the status: an edge answering 530 is not a safety refusal.
  window.fetch = async () => new Response('<html>error</html>', {status: 530});
  await canvas.open(htmlArtifact());
  await canvas.updateComplete;
  expect(canvas.querySelector('.artifact-error')?.textContent).to.contain('530');
  expect(canvas.querySelector('.artifact-error')?.textContent).not.to.contain('safely');
  canvas.close();

  // A read that never reached a response says so.
  window.fetch = async () => { throw new TypeError('Failed to fetch'); };
  await canvas.open(artifact('unreachable', 'Unreachable', 'html'));
  await canvas.updateComplete;
  expect(canvas.querySelector('.artifact-error')?.textContent).to.contain('could not be reached');
  canvas.close();

  // Bytes that fail their schema keep the wording that says the Canvas refused them.
  window.fetch = async () => new Response('not the presentation');
  await canvas.open(artifact('unreadable', 'Unreadable', 'markdown'));
  await canvas.updateComplete;
  expect(canvas.querySelector('.artifact-error')?.textContent).to.contain('safely');
});

it('operator-disabled HTML uses only the script-disabled static frame', async () => {
  window.fetch = async () => new Response('<!doctype html><html><body>Static</body></html>');
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  canvas.activePreviewEnabled = false;
  document.body.appendChild(canvas);

  await canvas.open(htmlArtifact());
  await canvas.updateComplete;

  const frame = canvas.querySelector('dl-active-artifact-frame');
  expect(frame).not.to.equal(null);
  expect(frame?.active).to.equal(false);
  expect(frame?.source).to.contain('Static');
  expect(canvas.textContent).not.to.contain('Open interactive report');
  expect(canvas.querySelector('details')).to.equal(null);
});

it('shows a PDF in a no-referrer frame that is not sandboxed', async () => {
  // Chrome refuses the PDF viewer inside a sandboxed frame, empty or fully permissive, so the
  // sandbox that used to be asserted here bought a broken-page icon rather than isolation.
  const artifact = {...htmlArtifact(),
    media_type: 'application/pdf',
    filename: 'report.pdf',
    presentation: 'pdf' as const,
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  const iframe = canvas.querySelector<HTMLIFrameElement>('iframe.artifact-pdf');
  expect(iframe?.hasAttribute('sandbox')).to.equal(false);
  expect(iframe?.referrerPolicy).to.equal('no-referrer');
});

it('plays a published video from the range-capable Artifact URL', async () => {
  const artifact = {...htmlArtifact(),
    mediaType: 'video/mp4',
    filename: 'clip.mp4',
    presentation: 'video' as const,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-video',
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  const video = canvas.querySelector<HTMLVideoElement>('[data-artifact-video]');
  expect(video).not.to.equal(null);
  expect(video?.src).to.equal(
    new URL('/web/api/answer/run-1/artifacts/artifact-video', window.location.origin).href,
  );
  expect(video?.hasAttribute('controls')).to.equal(true);
  expect(video?.getAttribute('preload')).to.equal('metadata');
  expect(canvas.querySelector('.artifact-download-only')).to.equal(null);
});

it('keeps the video player reachable inside the modal focus trap', async () => {
  const artifact = {...htmlArtifact(),
    mediaType: 'video/mp4',
    filename: 'clip.mp4',
    presentation: 'video' as const,
    dataUrl: '/web/api/answer/run-1/artifacts/artifact-video',
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  const layoutButtons = Array.from(canvas.querySelectorAll<HTMLButtonElement>(
    '.artifact-canvas-layout-actions button',
  ));
  layoutButtons.find((button) => button.textContent?.trim() === 'Fullscreen')?.click();
  await canvas.updateComplete;
  expect(canvas.getAttribute('aria-modal')).to.equal('true');

  // Mirrors the trap's own selector: the player is a legitimate stop, so a
  // keyboard user can reach the native controls instead of losing them.
  const trapped = Array.from(canvas.querySelectorAll<HTMLElement>(
    'button:not([disabled]), dl-icon-button:not([disabled]), a[href], iframe, video[controls], [tabindex]:not([tabindex="-1"])',
  )).filter((element) => element.getClientRects().length > 0);
  const video = canvas.querySelector<HTMLVideoElement>('[data-artifact-video]');
  expect(trapped.at(-1)).to.equal(video);

  video?.focus();
  expect(document.activeElement).to.equal(video);
  const propagated = video?.dispatchEvent(new KeyboardEvent('keydown', {
    key: 'Tab', bubbles: true, cancelable: true,
  }));
  // Focus stays inside the dialog: the trap consumes Tab at its last stop.
  expect(propagated).to.equal(false);
  expect(document.activeElement).to.equal(trapped[0]);
});

it('an unavailable Artifact renders a persistent safe issue without fetching', async () => {
  let fetched = false;
  window.fetch = async () => {
    fetched = true;
    return new Response();
  };
  const artifact = htmlArtifact();
  artifact.status = 'unavailable';
  artifact.issue = {
    kind: 'missing_file',
    description: 'Referenced Artifact is missing.',
    resourceId: artifact.resourceId,
  };
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  await canvas.open(artifact);
  await canvas.updateComplete;

  expect(canvas.querySelector('[role="alert"]')?.textContent).to.contain('missing');
  expect(fetched).to.equal(false);
});

it('names the Artifact attestation status in the Canvas header, and nothing under it', async () => {
  window.matchMedia = desktopMedia;
  const canvas = document.createElement('dl-artifact-canvas') as DlArtifactCanvas;
  document.body.appendChild(canvas);

  // An Artifact the server derived no record for draws no badge at all: `null` is not a verdict.
  canvas.artifact = {...htmlArtifact(), attestation: null};
  await canvas.updateComplete;
  expect(canvas.querySelector('.artifact-attestation')).to.equal(null);

  const badgeFor = async (attestation: AnswerArtifact['attestation']) => {
    canvas.artifact = {...htmlArtifact(), attestation};
    await canvas.updateComplete;
    const badge = canvas.querySelector('.artifact-attestation');
    if (badge !== null) {
      expect(canvas.querySelector('.artifact-canvas-heading')?.contains(badge)).to.equal(true);
    }
    return badge;
  };
  const words = (badge: Element | null): string =>
    badge?.textContent?.replace(/\s+/g, ' ').trim() ?? '';

  const attested = await badgeFor({
    status: 'attested',
    skill: 'sealed-research-brief',
    verifier: 'scripts/validate_model.py',
    reason: null,
    basis: 'named_in_a_recorded_command',
    requiredReadsMissing: null,
  });
  expect(attested?.getAttribute('data-attestation')).to.equal('attested');
  // The claim, not its basis: the Answer's delivery record names the verifier the claim rests on.
  expect(words(attested)).to.equal('Attested by sealed-research-brief');
  expect(attested?.querySelector('.artifact-attestation-detail')).to.equal(null);

  const unattested = await badgeFor({
    status: 'unattested', skill: null, verifier: null, reason: 'no_skill_loaded', basis: null,
    requiredReadsMissing: null,
  });
  expect(words(unattested)).to.equal('Not attested');

  // The Answer that brought a reader here: the records cannot settle the question because a recorded
  // call was cut, and the Run skipped three declared reads. The Canvas header stops at the status;
  // the reason and the lines are the Answer's delivery record, which says them once for the Run.
  const unknown = await badgeFor({
    status: 'unknown', skill: null, verifier: null, reason: 'arguments_truncated', basis: null,
    requiredReadsMissing: [
      {
        skill: 'sealed-research-brief',
        paths: ['references/industry-structure.md'],
        when: 'before any forecast — the industry condition behind every parameter',
      },
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
  });
  expect(unknown?.getAttribute('data-attestation')).to.equal('unknown');
  expect(words(unknown)).to.equal('Attestation unknown');
  expect(unknown?.querySelector('.artifact-attestation-detail')).to.equal(null);
  expect(unknown?.querySelectorAll('.artifact-attestation-unread')).to.have.length(0);
  // Records that cannot settle the question are not a failure, so nothing here is announced as one.
  expect(canvas.querySelector('[role="alert"]')).to.equal(null);

  // A Skill that declares no verifier made no claim for a status to judge: the Canvas names no
  // verdict, and the skipped read is the Answer's record to draw.
  const claimless = await badgeFor({
    status: 'unattested', skill: 'interactive-html', verifier: null,
    reason: 'skill_has_no_verifier_contract', basis: null,
    requiredReadsMissing: [
      {skill: 'interactive-html', paths: ['references/fragment.md'], when: 'before the build'},
    ],
  });
  expect(claimless).to.equal(null);
});

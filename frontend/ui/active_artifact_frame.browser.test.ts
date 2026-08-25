// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './active_artifact_frame.ts';
import type {DlActiveArtifactFrame} from './active_artifact_frame.ts';

afterEach(() => { document.body.replaceChildren(); });

it('creates exactly one opaque-origin active iframe after source is provided', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  frame.source = '<script>parent.document.body.dataset.compromised="yes"</script>';
  document.body.appendChild(frame);
  await frame.updateComplete;

  const iframe = frame.shadowRoot?.querySelector('iframe');
  expect(frame.shadowRoot?.querySelectorAll('iframe').length).to.equal(1);
  expect(iframe?.getAttribute('sandbox')).to.equal('allow-scripts');
  expect(iframe?.getAttribute('sandbox')).not.to.contain('allow-same-origin');
  expect(iframe?.referrerPolicy).to.equal('no-referrer');
  expect(iframe?.getAttribute('allow')).to.contain("camera 'none'");
  const source = iframe?.srcdoc || '';
  expect(source.indexOf('Content-Security-Policy')).to.be.lessThan(source.indexOf('compromised'));
  expect(source).to.contain("connect-src 'none'");
  expect(source).to.contain("worker-src 'none'");
  expect(source).to.contain("frame-src 'none'");
  expect(document.body.dataset.compromised).to.equal(undefined);
});

it('uses an opaque script-disabled iframe for the operator-disabled fallback', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = false;
  frame.source = '<script>document.body.dataset.ran="yes"</script><p>Static</p>';
  document.body.appendChild(frame);
  await frame.updateComplete;

  const iframe = frame.shadowRoot?.querySelector('iframe');
  expect(iframe?.getAttribute('sandbox')).to.equal('');
  expect(iframe?.srcdoc).to.contain("script-src 'none'");
});

it('destroy removes the iframe and its browser-held bytes', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  frame.source = '<p>Secret bytes</p>';
  document.body.appendChild(frame);
  await frame.updateComplete;

  frame.destroy();
  await frame.updateComplete;

  expect(frame.shadowRoot?.querySelector('iframe')).to.equal(null);
  expect(frame.source).to.equal(null);
});

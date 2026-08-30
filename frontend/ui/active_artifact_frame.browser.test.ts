// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './active_artifact_frame.ts';
import type {DlActiveArtifactFrame} from './active_artifact_frame.ts';

afterEach(() => { document.body.replaceChildren(); });

it('executes scripts inside one opaque-origin active iframe without reaching the parent DOM', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  const scriptRan = new Promise<void>((resolve) => {
    const receive = (event: MessageEvent): void => {
      if (event.data !== 'dl-test-script-ran') return;
      window.removeEventListener('message', receive);
      resolve();
    };
    window.addEventListener('message', receive);
  });
  frame.source = '<script>try{parent.document.body.dataset.compromised="yes"}catch(_){};' +
    'parent.postMessage("dl-test-script-ran","*")</script>';
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
  await scriptRan;
  expect(document.body.dataset.compromised).to.equal(undefined);
});

it('captures its private Escape signal before hostile Artifact listeners', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  const scriptReady = new Promise<void>((resolve) => {
    const receive = (event: MessageEvent): void => {
      if (event.data !== 'dl-test-escape-ready') return;
      window.removeEventListener('message', receive);
      resolve();
    };
    window.addEventListener('message', receive);
  });
  frame.source = '<script>document.addEventListener("keydown",event=>' +
    'event.stopImmediatePropagation(),true);addEventListener("message",event=>{' +
    'if(event.data==="dl-test-trigger-forged"){' +
    'parent.postMessage({type:"dl-artifact-frame-escape",token:"forged"},"*");' +
    'parent.postMessage("dl-test-forged-sent","*")}' +
    'if(event.data==="dl-test-trigger-escape")document.body.dispatchEvent(' +
    'new KeyboardEvent("keydown",{key:"Escape",bubbles:true,composed:true}))});' +
    'parent.postMessage("dl-test-escape-ready","*")</script>';
  let escapes = 0;
  const escaped = new Promise<void>((resolve) => {
    frame.addEventListener('artifact-frame-escape', () => {
      escapes += 1;
      resolve();
    });
  });
  document.body.appendChild(frame);
  await frame.updateComplete;
  const iframe = frame.shadowRoot?.querySelector('iframe');
  await scriptReady;

  const forgedSent = new Promise<void>((resolve) => {
    const receive = (event: MessageEvent): void => {
      if (event.data !== 'dl-test-forged-sent') return;
      window.removeEventListener('message', receive);
      resolve();
    };
    window.addEventListener('message', receive);
  });
  iframe?.contentWindow?.postMessage('dl-test-trigger-forged', '*');
  await forgedSent;
  expect(escapes).to.equal(0);
  iframe?.contentWindow?.postMessage('dl-test-trigger-escape', '*');
  await escaped;
  expect(escapes).to.equal(1);
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

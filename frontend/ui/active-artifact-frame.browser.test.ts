// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import './active-artifact-frame.ts';
import type {DlActiveArtifactFrame} from './active-artifact-frame.ts';

afterEach(() => { document.body.replaceChildren(); });

/** Resolves when the page next receives exactly this message from a sandboxed (opaque-origin) frame. */
function messageFromFrame(text: string): Promise<void> {
  return new Promise((resolve) => {
    const receive = (event: MessageEvent): void => {
      if (event.origin !== 'null' || event.data !== text) return;
      window.removeEventListener('message', receive);
      resolve();
    };
    window.addEventListener('message', receive);
  });
}

it('executes scripts inside one opaque-origin active iframe without reaching the parent DOM', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  const scriptRan = messageFromFrame('dl-test-script-ran');
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
  const scriptReady = messageFromFrame('dl-test-escape-ready');
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
    frame.addEventListener('dl-artifact-frame-escape', () => {
      escapes += 1;
      resolve();
    });
  });
  document.body.appendChild(frame);
  await frame.updateComplete;
  const iframe = frame.shadowRoot?.querySelector('iframe');
  await scriptReady;

  const forgedSent = messageFromFrame('dl-test-forged-sent');
  iframe?.contentWindow?.postMessage('dl-test-trigger-forged', '*');
  await forgedSent;
  expect(escapes).to.equal(0);
  iframe?.contentWindow?.postMessage('dl-test-trigger-escape', '*');
  await escaped;
  expect(escapes).to.equal(1);
});

it('leaves the Escape that cancels an IME composition to the composition, and leaves the preview for the next', async () => {
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  // Chromium marks the key of a composition; WebKit sends it after it ended, as keyCode 229. Every
  // press is followed by a message, which reaches the page after the Escape the frame forwards, if any.
  frame.source = '<script>addEventListener("message",event=>{' +
    'if(event.data!=="dl-test-press")return;' +
    'for(const init of [{isComposing:true},{keyCode:229},{}])' +
    'document.body.dispatchEvent(new KeyboardEvent("keydown",{key:"Escape",bubbles:true,composed:true,...init}));' +
    'parent.postMessage("dl-test-pressed","*")});' +
    'parent.postMessage("dl-test-press-ready","*")</script>';
  let escapes = 0;
  frame.addEventListener('dl-artifact-frame-escape', () => { escapes += 1; });
  const ready = messageFromFrame('dl-test-press-ready');
  document.body.appendChild(frame);
  await frame.updateComplete;
  await ready;

  const pressed = messageFromFrame('dl-test-pressed');
  frame.shadowRoot?.querySelector('iframe')?.contentWindow?.postMessage('dl-test-press', '*');
  await pressed;

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

it('hands the product colour mode to the frame and keeps it current', async () => {
  document.documentElement.dataset.colorMode = 'light';
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = true;
  const ready = messageFromFrame('dl-test-mode-ready');
  frame.source = '<script>addEventListener("message",event=>{' +
    'if(!event.data||event.data.type!=="dl-artifact-frame-color-mode")return;' +
    'parent.postMessage("dl-test-mode:"+document.documentElement.getAttribute("data-color-mode"),"*")});' +
    'parent.postMessage("dl-test-mode-ready","*")</script>';
  document.body.appendChild(frame);
  try {
    await frame.updateComplete;
    await ready;
    const iframe = frame.shadowRoot?.querySelector('iframe');
    expect(iframe?.srcdoc).to.contain('data-color-mode="light"');

    const switched = messageFromFrame('dl-test-mode:dark');
    document.documentElement.dataset.colorMode = 'dark';
    await switched;
  } finally {
    document.documentElement.removeAttribute('data-color-mode');
  }
});

it('wears the product colour mode even where the iframe may not run scripts', async () => {
  document.documentElement.dataset.colorMode = 'dark';
  const frame = document.createElement('dl-active-artifact-frame') as DlActiveArtifactFrame;
  frame.active = false;
  frame.source = '<p>Static</p>';
  document.body.appendChild(frame);
  try {
    await frame.updateComplete;
    const iframe = frame.shadowRoot?.querySelector('iframe');
    expect(iframe?.srcdoc).to.contain('data-color-mode="dark"');
  } finally {
    document.documentElement.removeAttribute('data-color-mode');
  }
});

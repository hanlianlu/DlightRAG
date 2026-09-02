// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {RunController} from '../lib/run-controller.ts';
import {answerEventCursorStore} from '../stores/answer-event-cursor-store.ts';

it('uses the browser fetch interface with its required global receiver', async () => {
  const originalFetch = window.fetch;
  const conversationId = 'browser-fetch-receiver';
  const runId = 'run-browser-fetch';
  let receiver: unknown = null;
  window.fetch = function(this: Window): Promise<Response> {
    receiver = this;
    return Promise.resolve(new Response(
      'id: 1\nevent: done\ndata: {"status":"cancelled","presentation":null}\n\n',
      {status: 200, headers: {'Content-Type': 'text/event-stream'}},
    ));
  } as typeof fetch;
  answerEventCursorStore.trackRun(conversationId, runId);
  try {
    const controller = new RunController();
    controller.beginFollow(runId, false);
    const result = await controller.follow(conversationId, runId, () => {});

    expect(receiver).to.equal(window);
    expect(result.kind).to.equal('terminal');
    controller.finish(runId);
  } finally {
    answerEventCursorStore.clear(conversationId);
    window.fetch = originalFetch;
  }
});

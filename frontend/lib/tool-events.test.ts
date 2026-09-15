// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {prettyToolName, toolDisplay} from './tool-display.ts';
import {applyToolEvent, MAX_TOOL_ROWS, rowDurationMs, toolStatusText} from './tool-events.ts';

test('known tools map to verbs with i18n ids', () => {
  const display = toolDisplay('load_skill');
  assert.deepEqual(display, {
    verb: 'Loading skill',
    verbId: 'chatFeature.tool.load_skill',
    known: true,
  });
});

test('unknown tools fall back to a prettified raw name without an i18n id', () => {
  assert.deepEqual(toolDisplay('acme_custom_tool'), {
    verb: 'Acme Custom Tool',
    verbId: null,
    known: false,
  });
  assert.equal(prettyToolName('agent-child:abc'), 'Agent Child Abc');
});

test('tool events build, update, and settle one trace row', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'load_skill', call_id: 'c1'}, 1000);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].state, 'running');
  assert.equal(rows[0].object, '');

  rows = applyToolEvent(rows, 'tool_progress', {call_id: 'c1', object_label: 'code-review'}, 1000);
  assert.equal(rows[0].object, 'code-review');

  rows = applyToolEvent(
    rows,
    'tool_end',
    {call_id: 'c1', outcome: 'succeeded', duration_ms: 1200},
    1000,
  );
  assert.equal(rows[0].state, 'done');
  assert.equal(rows[0].durationMs, 1200);
  assert.equal(toolStatusText(rows), 'Loading skill — code-review');
});

test('search status identifies the original query', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'search_web', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_progress', {
    call_id: 'c1',
    object_label: 'quarterly revenue 2026',
  }, 1000);

  assert.equal(toolStatusText(rows), 'Searching the web — quarterly revenue 2026');
});

test('base file tools keep curated verbs with catalog entries', () => {
  assert.deepEqual(toolDisplay('bash'), {
    verb: 'Running a command',
    verbId: 'chatFeature.tool.bash',
    known: true,
  });
  assert.deepEqual(toolDisplay('grep'), {
    verb: 'Searching files',
    verbId: 'chatFeature.tool.grep',
    known: true,
  });

  let rows = applyToolEvent([], 'tool_start', {tool_name: 'bash', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_progress', {call_id: 'c1', object_label: 'npm test'}, 1000);
  assert.equal(toolStatusText(rows), 'Running a command — npm test');
});

test('failed outcome marks the row failed', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'bash', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_end', {call_id: 'c1', outcome: 'failed', duration_ms: 30}, 1000);
  assert.equal(rows[0].state, 'failed');
});

test('the trace keeps at most the newest rows', () => {
  let rows: ReturnType<typeof applyToolEvent> = [];
  for (let i = 0; i < MAX_TOOL_ROWS + 3; i++) {
    rows = applyToolEvent(rows, 'tool_start', {tool_name: 'read', call_id: `c${i}`}, 1000);
  }
  assert.equal(rows.length, MAX_TOOL_ROWS);
  assert.equal(rows[rows.length - 1].callId, `c${MAX_TOOL_ROWS + 2}`);
});

test('malformed events never corrupt the trace', () => {
  assert.deepEqual(applyToolEvent([], 'tool_start', {}, 1000), []);
  assert.deepEqual(applyToolEvent([], 'tool_end', {call_id: 'c1'}, 1000), []);
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'read', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_progress', {call_id: 'c1', object_label: ''}, 1000);
  assert.equal(rows[0].object, '');
  assert.deepEqual(applyToolEvent(rows, 'telemetry', {call_id: 'c1'}, 1000), rows);
});

test('a server label names a Connection tool and never survives as a hash', () => {
  let rows = applyToolEvent([], 'tool_start', {
    tool_name: 'mcp_connection_deadbeef',
    call_id: 'c1',
    tool_label: 'Personal tools · search_issues',
  }, 1000);
  assert.equal(rows[0].label, 'Personal tools · search_issues');
  assert.equal(toolStatusText(rows), 'Personal tools · search_issues');

  rows = applyToolEvent(rows, 'tool_end', {
    call_id: 'c1',
    tool_label: 'Personal tools · search_issues',
    outcome: 'succeeded',
    duration_ms: 2400,
  }, 1000);
  assert.equal(toolStatusText(rows), 'Personal tools · search_issues');

  const unresolved = applyToolEvent([], 'tool_start', {
    tool_name: 'mcp_connection_deadbeef',
    call_id: 'c2',
  }, 1000);
  assert.equal(unresolved[0].label, null);
  assert.equal(toolStatusText(unresolved), 'Calling an MCP tool');
});

test('a running row ticks from the instant this viewer saw it start', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'read', call_id: 'c1'}, 1000);
  assert.equal(rows[0].startedAt, 1000);
  assert.equal(rowDurationMs(rows[0], 1000), 0);
  assert.equal(rowDurationMs(rows[0], 12_500), 11_500);

  rows = applyToolEvent(rows, 'tool_end', {
    call_id: 'c1',
    outcome: 'succeeded',
    duration_ms: 11_000,
  }, 20_000);
  assert.equal(rows[0].startedAt, null);
  assert.equal(rowDurationMs(rows[0], 99_999), 11_000, 'server truth, not the viewer clock');
});

test('a settlement without a measured duration shows none', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'read', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_end', {call_id: 'c1', outcome: 'outcome_unknown'}, 5000);
  assert.equal(rows[0].state, 'failed');
  assert.equal(rowDurationMs(rows[0], 9999), null);
});

test('every non-succeeded outcome fails the row', () => {
  for (const outcome of ['failed', 'invalid_arguments', 'tool_contract_changed', 'outcome_unknown']) {
    const rows = applyToolEvent(
      applyToolEvent([], 'tool_start', {tool_name: 'read', call_id: 'c1'}, 1000),
      'tool_end',
      {call_id: 'c1', outcome},
      1000,
    );
    assert.equal(rows[0].state, 'failed', outcome);
  }
});

test('a label arriving later fills a row that started unnamed', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'mcp_x', call_id: 'c1'}, 1000);
  rows = applyToolEvent(rows, 'tool_progress', {
    call_id: 'c1',
    tool_label: 'Notes · ping',
    object_label: 'ping',
  }, 1000);
  assert.equal(rows[0].label, 'Notes · ping');
  assert.equal(toolStatusText(rows), 'Notes · ping — ping');
});

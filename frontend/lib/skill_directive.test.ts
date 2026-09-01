// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {
  committedSkillDirective,
  parseSkillDirective,
  skillDirectiveState,
  skillGhostSuffix,
} from './skill_directive.ts';

test('parses the canonical directive with a trailing question', () => {
  assert.deepEqual(parseSkillDirective('/skill:review Check this plan'), {
    skill: 'review',
    query: 'Check this plan',
  });
});

test('parses the shorthand directive', () => {
  assert.deepEqual(parseSkillDirective('/review Check this plan'), {
    skill: 'review',
    query: 'Check this plan',
  });
});

test('parses a directive with no question as an empty query', () => {
  assert.deepEqual(parseSkillDirective('/skill:review'), {skill: 'review', query: ''});
  assert.deepEqual(parseSkillDirective('/review'), {skill: 'review', query: ''});
});

test('rejects off-path or invalid text', () => {
  assert.equal(parseSkillDirective('/skill:Bad_Name x'), null);
  assert.equal(parseSkillDirective('not /skill:review'), null);
  assert.equal(parseSkillDirective('plain text'), null);
});

test('tolerates surrounding whitespace and multiline questions', () => {
  assert.deepEqual(parseSkillDirective('  /skill:code-review \n  Line one\nLine two  '), {
    skill: 'code-review',
    query: 'Line one\nLine two',
  });
});

test('tracks canonical directive states while typing', () => {
  assert.deepEqual(skillDirectiveState('/'), {
    kind: 'canonical', typedKeyword: '', prefix: '',
  });
  assert.deepEqual(skillDirectiveState('/s'), {
    kind: 'canonical', typedKeyword: 's', prefix: '',
  });
  assert.deepEqual(skillDirectiveState('/skill:'), {
    kind: 'canonical', typedKeyword: 'skill', prefix: '',
  });
  assert.deepEqual(skillDirectiveState('/skill:r'), {
    kind: 'canonical', typedKeyword: 'skill', prefix: 'r',
  });
});

test('tracks shorthand states and resolves keyword-prefix ambiguity canonically', () => {
  assert.deepEqual(skillDirectiveState('/td'), {
    kind: 'shorthand', typedKeyword: '', prefix: 'td',
  });
  assert.deepEqual(skillDirectiveState('/code-r'), {
    kind: 'shorthand', typedKeyword: '', prefix: 'code-r',
  });
  // 'seo-checklist' style names never collide with the 'skill' keyword prefix.
  assert.deepEqual(skillDirectiveState('/seo'), {
    kind: 'shorthand', typedKeyword: '', prefix: 'seo',
  });
});

test('directive state rejects off-path or complete drafts', () => {
  assert.equal(skillDirectiveState(''), null);
  assert.equal(skillDirectiveState('abc'), null);
  assert.equal(skillDirectiveState('/skill:review the plan'), null);
  assert.equal(skillDirectiveState('/skill:Bad_Name'), null);
});

test('shorthand state covers bare slash and arbitrary name prefixes', () => {
  assert.deepEqual(skillDirectiveState('/x'), {
    kind: 'shorthand', typedKeyword: '', prefix: 'x',
  });
});

test('ghost teaches the full canonical form during the keyword phase', () => {
  assert.equal(skillGhostSuffix('/', 'tdd'), 'skill:tdd');
  assert.equal(skillGhostSuffix('/s', 'tdd'), 'kill:tdd');
  assert.equal(skillGhostSuffix('/skill', 'tdd'), ':tdd');
  assert.equal(skillGhostSuffix('/skill:', 'tdd'), 'tdd');
  assert.equal(skillGhostSuffix('/skill:t', 'tdd'), 'dd');
});

test('ghost narrows shorthand and canonical name prefixes', () => {
  assert.equal(skillGhostSuffix('/td', 'tdd'), 'd');
  assert.equal(skillGhostSuffix('/code-r', 'code-review'), 'eview');
});

test('ghost is empty off the directive path', () => {
  assert.equal(skillGhostSuffix('hello', 'tdd'), '');
  assert.equal(skillGhostSuffix('/skill:tdd now', 'tdd'), '');
});

test('committed forms carry the trailing space and preserve the typed style', () => {
  assert.equal(committedSkillDirective('/', 'tdd'), '/skill:tdd ');
  assert.equal(committedSkillDirective('/skill:t', 'tdd'), '/skill:tdd ');
  assert.equal(committedSkillDirective('/td', 'tdd'), '/tdd ');
  assert.equal(committedSkillDirective('plain', 'tdd'), '/skill:tdd ');
});

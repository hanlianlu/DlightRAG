// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The splits lay themselves out for what the Shell tells them is open, and open each panel at
 *  its approved share of the split it lives in. */

import {expect} from '@esm-bundle/chai';
import {type DlSplitLayout, defineDesignSystemElements} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA, DESKTOP_SHELL_MEDIA, MOBILE_MEDIA} from '../lib/breakpoints.ts';
import {IDLE_SHELL_SPLIT_STATE, setupPanelSplits, syncPanelSplitState} from './split-panel.ts';

defineDesignSystemElements();

const originalMatchMedia = window.matchMedia;
/** The Shell's compact breakpoint, flipped by the test that crosses it. */
let compactShell = false;
/** Whether the compact Shell is a phone, which a drawer fills edge to edge. */
let phoneShell = false;

let conversation: DlSplitLayout;
let inspector: DlSplitLayout;
let canvas: DlSplitLayout;

const frame = (): Promise<unknown> => new Promise((resolve) => requestAnimationFrame(resolve));

before(async () => {
  window.matchMedia = ((query: string) => ({
    matches: {
      [DESKTOP_SHELL_MEDIA]: !compactShell,
      [COMPACT_SHELL_MEDIA]: compactShell,
      [MOBILE_MEDIA]: phoneShell,
    }[query] ?? false,
    media: query,
    onchange: null,
    addListener() {},
    removeListener() {},
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent: () => true,
  })) as typeof window.matchMedia;
  document.documentElement.style.setProperty('--layout-chat-sidebar-width', '260px');
  document.documentElement.style.setProperty('--panel-width', '420px');
  document.documentElement.style.setProperty('--artifact-canvas-width', '420px');
  document.body.innerHTML = `
    <dl-split-layout id="conversation-split" primary="start" size="0" min="240"
      style="display:block;width:1400px;height:500px">
      <div id="conversation-sidebar" slot="start"></div>
      <dl-split-layout id="panel-split" slot="end" primary="end" size="0" min="320">
        <dl-split-layout id="artifact-canvas-split" slot="start" primary="end" size="0" min="320">
          <div slot="start"></div>
          <div slot="end" id="artifact-canvas" class="open"></div>
        </dl-split-layout>
        <div slot="end" id="inspector"></div>
      </dl-split-layout>
    </dl-split-layout>
  `;
  conversation = document.querySelector<DlSplitLayout>('#conversation-split')!;
  inspector = document.querySelector<DlSplitLayout>('#panel-split')!;
  canvas = document.querySelector<DlSplitLayout>('#artifact-canvas-split')!;
  setupPanelSplits();
  await frame();
});

after(() => {
  document.body.replaceChildren();
  window.matchMedia = originalMatchMedia;
});

afterEach(() => {
  conversation.style.width = '1400px';
  compactShell = false;
  phoneShell = false;
  shellInspectorKind = null;
  syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
  for (const key of [
    'dlightrag-conversation-sidebar-width',
    'dlightrag-panel-width',
    'dlightrag-artifact-canvas-width',
  ]) {
    window.localStorage.removeItem(key);
  }
});

describe('syncPanelSplitState', () => {
  it('opens the conversation split when the Shell says the list is open', async () => {
    syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
    await frame();
    expect(conversation.size, 'a Shell with nothing open').to.equal(0);

    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE, conversationOpen: true});
    await frame();
    expect(conversation.size).to.be.at.least(240);
  });

  it('gives the Canvas the whole column when the Shell says it is wide, and holds it back otherwise', async () => {
    syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
    await frame();
    const shared = canvas.max;
    expect(shared, 'beside the chat, the Canvas leaves the chat its share').to.be.lessThan(canvas.clientWidth);

    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE, canvasWide: true});
    await frame();
    expect(canvas.max, 'wide, it may take the whole column').to.equal(canvas.clientWidth);
    expect(canvas.disabled, 'and its divider is gone: the width is the column\'s').to.equal(true);

    syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
    await frame();
    expect(canvas.max, 'and holds it back once the Shell says it is not').to.equal(shared);
  });

  it('lets the Canvas cover the screen without taking room from the Inspector it is over', async () => {
    syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
    await frame();
    const beside = inspector.max;

    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE, canvasOverlay: true});
    await frame();
    expect(canvas.disabled, 'its divider is gone while it covers the screen').to.equal(true);
    expect(inspector.max, 'the Inspector made room for a Canvas that is over it').to.be.greaterThan(beside);
  });

  it('goes on with what the Shell last said when the splits lay themselves out', async () => {
    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE, conversationOpen: true});
    await frame();
    const size = conversation.size;
    expect(size).to.be.at.least(240);

    window.dispatchEvent(new Event('resize'));
    await frame();
    expect(conversation.size, 'a resize closed what the Shell had open').to.equal(size);
  });
});

/** Show or put away one of the Shell's panels, as the component's own open class says it is. */
function showPanel(id: string, open: boolean): void {
  document.getElementById(id)!.classList.toggle('open', open);
}

/** The grain the Inspector is on, as the Shell states it to the splits on every sync. */
let shellInspectorKind: 'files' | 'sources' | 'traces' | 'live' | null = null;

/** The desktop Shell with nothing the splits watch open, so a test can open one thing at its default. */
function closeEveryPanel(): void {
  shellInspectorKind = null;
  showPanel('conversation-sidebar', false);
  showPanel('inspector', false);
  showPanel('artifact-canvas', false);
  syncPanelSplitState(IDLE_SHELL_SPLIT_STATE);
}

/** Lay the splits out for the desktop Shell with the conversation list showing. */
function syncConversationOpen(): void {
  syncPanelSplitState({
    ...IDLE_SHELL_SPLIT_STATE,
    conversationOpen: true,
    inspectorKind: shellInspectorKind,
  });
}

/** The width of the split the conversation list lives in, which stands in for the window's. */
function setAxisWidth(width: number): void {
  conversation.style.width = `${width}px`;
}

// ADR 0067: a panel's default width is a share of the split it lives in, not a fixed token. It is
// asked for on first open and on a grain change, and held to the split's own room; a reader's drag
// is what stays in pixels.
describe('proportional panel defaults', () => {
  it('opens the conversation list at 18% of the axis it spans, within its own range', async () => {
    closeEveryPanel();
    setAxisWidth(1800);
    syncConversationOpen();
    expect(conversation.size, '18% of the window').to.equal(324);

    closeEveryPanel();
    setAxisWidth(1000);
    syncConversationOpen();
    expect(conversation.size, 'a narrow window keeps the minimum').to.equal(240);

    closeEveryPanel();
    setAxisWidth(2400);
    syncConversationOpen();
    expect(conversation.size, 'a wide window holds the maximum').to.equal(360);
  });

  it('opens Files, Sources and the live browser at 38% of the panel split', async () => {
    closeEveryPanel();
    setAxisWidth(1600);
    syncConversationOpen();
    const axis = inspector.clientWidth;
    showPanel('inspector', true);
    syncConversationOpen();
    expect(axis, 'the panel split is what is beside the conversation list').to.be.below(1600);
    expect(inspector.size, 'the panel split, not the window').to.equal(Math.round(0.38 * axis));
    expect(inspector.size).to.be.within(320, 560);
  });

  it('opens the traces grain at half the panel split, within a wider range', async () => {
    closeEveryPanel();
    setAxisWidth(1800);
    syncConversationOpen();
    const axis = inspector.clientWidth;
    shellInspectorKind = 'traces';
    showPanel('inspector', true);
    syncConversationOpen();
    expect(inspector.size, 'the split beside the conversation list, halved').to.equal(Math.round(0.5 * axis));
    expect(inspector.size).to.be.within(420, 960);
  });

  it('caps the traces default at the room the answer keeps', async () => {
    closeEveryPanel();
    setAxisWidth(1260);
    syncConversationOpen();
    const axis = inspector.clientWidth;
    shellInspectorKind = 'traces';
    showPanel('inspector', true);
    syncConversationOpen();
    expect(inspector.size, 'half the split would crowd the answer').to.equal(axis - 520 - 1);
    expect(inspector.size, 'so it is the room left, not the half').to.be.below(Math.round(0.5 * axis));
    expect(inspector.size).to.be.at.least(420);
  });

  it('opens the Artifact Canvas at 42% of its own split', async () => {
    closeEveryPanel();
    setAxisWidth(2200);
    syncConversationOpen();
    showPanel('inspector', true);
    syncConversationOpen();
    const axis = canvas.clientWidth;
    showPanel('artifact-canvas', true);
    syncConversationOpen();
    expect(axis, 'the Canvas split is what the panel split leaves beside the Inspector')
      .to.be.below(inspector.clientWidth);
    expect(canvas.size, 'the Canvas split, not the panel split or the window').to.equal(Math.round(0.42 * axis));
    expect(canvas.size).to.be.within(320, 720);
  });

  it('re-derives an unstated width when the Inspector changes grain', async () => {
    closeEveryPanel();
    setAxisWidth(1700);
    syncConversationOpen();
    showPanel('inspector', true);
    syncConversationOpen();
    const axis = inspector.clientWidth;
    expect(inspector.size, 'Files opens at its share').to.equal(Math.round(0.38 * axis));
    shellInspectorKind = 'traces';
    syncConversationOpen();
    expect(inspector.size, 'traces asks for the wider half').to.equal(Math.round(0.5 * axis));
    shellInspectorKind = null;
    syncConversationOpen();
    expect(inspector.size, 'and Files asks for its share again').to.equal(Math.round(0.38 * axis));
  });

  it('leaves storage empty while every width is still a default', async () => {
    closeEveryPanel();
    setAxisWidth(1800);
    syncConversationOpen();
    shellInspectorKind = 'traces';
    showPanel('inspector', true);
    syncConversationOpen();
    showPanel('artifact-canvas', true);
    syncConversationOpen();
    expect(window.localStorage.getItem('dlightrag-conversation-sidebar-width')).to.equal(null);
    expect(window.localStorage.getItem('dlightrag-panel-width')).to.equal(null);
    expect(window.localStorage.getItem('dlightrag-artifact-canvas-width')).to.equal(null);
  });

  it('clamps what is drawn on a window resize without re-deriving the default', async () => {
    closeEveryPanel();
    setAxisWidth(1800);
    syncConversationOpen();
    expect(conversation.size).to.equal(324);

    setAxisWidth(800);
    window.dispatchEvent(new Event('resize'));
    expect(conversation.size, 'a narrow window keeps the answer its room').to.equal(800 - 520 - 1);

    setAxisWidth(900);
    window.dispatchEvent(new Event('resize'));
    expect(conversation.size, 'growing back shows the width it opened at, not a new share').to.equal(324);
  });

  it('clamps rather than re-deriving when the layout crosses the compact breakpoint', async () => {
    closeEveryPanel();
    setAxisWidth(2200);
    shellInspectorKind = 'traces';
    syncConversationOpen();
    showPanel('inspector', true);
    syncConversationOpen();
    const opened = inspector.size;
    expect(opened, 'traces opened at its share').to.equal(Math.round(0.5 * inspector.clientWidth));

    compactShell = true;
    window.dispatchEvent(new Event('resize'));
    compactShell = false;
    window.dispatchEvent(new Event('resize'));
    expect(inspector.size, 'coming back is a resize, not a fresh share').to.equal(opened);
  });
});

// ADR 0068: compact drawers keep their existing behavior. A drawer asks for no share: it fills a
// phone's column, opens at 420px beyond it, and is owed its share until it docks.
describe('compact drawers', () => {
  /** The compact Shell with the Inspector opened in it, the list being a drawer of its own. */
  function openInspectorAsDrawer(window: number): void {
    closeEveryPanel();
    setAxisWidth(window);
    compactShell = true;
    showPanel('inspector', true);
    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE});
  }

  it('fills the whole column on a phone', async () => {
    phoneShell = true;
    openInspectorAsDrawer(390);
    expect(inspector.clientWidth, 'the column is the phone\'s width').to.equal(390);
    expect(inspector.size, 'a phone\'s drawer is full bleed, not its share (which is the minimum)').to.equal(390);
  });

  it('opens at 420px on a tablet, within the room the column has', async () => {
    openInspectorAsDrawer(900);
    expect(inspector.size, 'not 38% of 900').to.equal(420);

    openInspectorAsDrawer(380);
    expect(inspector.size, 'a column narrower than the drawer gives it all of it').to.equal(380);
  });

  it('asks for its share when it docks, not before', async () => {
    openInspectorAsDrawer(1200);
    expect(inspector.size).to.equal(420);

    compactShell = false;
    syncPanelSplitState({...IDLE_SHELL_SPLIT_STATE});
    expect(inspector.size, 'docked at last, it takes its share of the split').to.equal(
      Math.round(0.38 * inspector.clientWidth),
    );
  });
});

// What the reader states is kept in pixels, and a double-click asks the kind's share back.
describe('stated panel widths', () => {
  it('keeps a dragged width when the Inspector changes grain', async () => {
    closeEveryPanel();
    setAxisWidth(1700);
    syncConversationOpen();
    showPanel('inspector', true);
    syncConversationOpen();
    inspector.size = 700;
    inspector.dispatchEvent(new CustomEvent('dl-split-change', {
      bubbles: true,
      composed: true,
      detail: {position: 700},
    }));
    expect(inspector.size, 'the drag took it there').to.equal(700);

    shellInspectorKind = 'traces';
    syncConversationOpen();
    expect(inspector.size, 'a stated width is not replaced by the traces share').to.equal(700);
    expect(window.localStorage.getItem('dlightrag-panel-width')).to.equal('700');
  });

  it('asks for the grain\'s share on a double-click on the sash', async () => {
    // The Inspector is explicit from the drag above; a double-click is what asks for a default again.
    closeEveryPanel();
    setAxisWidth(1700);
    syncConversationOpen();
    shellInspectorKind = 'traces';
    showPanel('inspector', true);
    syncConversationOpen();
    const axis = inspector.clientWidth;
    const doubleClick = (): void => {
      inspector.divider.dispatchEvent(new MouseEvent('dblclick', {bubbles: true}));
    };

    doubleClick();
    expect(inspector.size, 'the traces share, not the pixels it had').to.equal(Math.round(0.5 * axis));
    expect(window.localStorage.getItem('dlightrag-panel-width')).to.equal(String(inspector.size));

    shellInspectorKind = null;
    syncConversationOpen();
    doubleClick();
    expect(inspector.size, 'and the Files share after the grain changes').to.equal(Math.round(0.38 * axis));
    expect(window.localStorage.getItem('dlightrag-panel-width')).to.equal(String(inspector.size));
  });
});

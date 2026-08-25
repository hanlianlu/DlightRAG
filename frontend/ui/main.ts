// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/primitives.css';
import '../styles/layout.css';
import '../styles/panels.css';
import '../styles/artifacts.css';
import '../styles/files.css';
import '../styles/sources.css';

import './app.ts';
import {requestDeleteAll, setupConversations} from './conversations.ts';
import {setupSettings} from './settings.ts';
import {setupFilesPanel} from './files-panel.ts';
import {setupImageLightbox} from './images.ts';
import {setupMathRendering} from './mathjax.ts';
import {setupChatMemoryOperationAdapter} from './memory.ts';
import {setupNotifications} from './notifications.ts';
import {setupPanel} from './panel.ts';
import {setupPanelSplits} from './split_panel.ts';
import {setupSourcePanel} from './source-panel.ts';
import {setupTheme} from './theme.ts';
import {initWorkspaces} from './workspaces.ts';

// Every module below is needed on first paint, so splitting them into dynamic
// imports only added round-trips and blocked cross-module tree-shaking.
document.addEventListener('DOMContentLoaded', function() {
    const app = document.querySelector('dl-app');
    if (!app) return;
    void app.ready.then(function() {
        setupTheme();
        initWorkspaces();
        setupPanelSplits();
        setupPanel();
        setupSourcePanel();
        setupFilesPanel();
        setupSettings(requestDeleteAll);
        setupChatMemoryOperationAdapter();
        setupImageLightbox();
        setupMathRendering();
        setupConversations();
        setupNotifications();
    });
});

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/layout.css';
import '../styles/files.css';
import '../styles/sources.css';

import {setupAttachmentInputs} from './attachments.ts';
import {setupQueryForm} from './chat.ts';
import {setupConversations} from './conversations.ts';
import {setupFilesPanel} from './files-panel.ts';
import {setupImageLightbox} from './images.ts';
import {setupMathRendering} from './mathjax.ts';
import {setupPanel} from './panel.ts';
import {setupPanelResize} from './resize.ts';
import {setupSourcePanel} from './source-panel.ts';
import {setupTheme} from './theme.ts';
import {initWorkspaces} from './workspaces.ts';

// Every module below is needed on first paint, so splitting them into dynamic
// imports only added round-trips and blocked cross-module tree-shaking.
document.addEventListener('DOMContentLoaded', function() {
    setupTheme();
    initWorkspaces();
    setupPanel();
    setupSourcePanel();
    setupFilesPanel();
    setupPanelResize();
    setupImageLightbox();
    setupAttachmentInputs();
    setupQueryForm();
    setupMathRendering();
    setupConversations();
});

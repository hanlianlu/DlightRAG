// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {setupQueryForm} from './chat.js';
import {setupHtmxInteractions} from './htmx.js';
import {setupImageInputs} from './images.js';
import {setupMathRendering} from './mathjax.js';
import {setupPanel} from './panel.js';
import {setupPanelResize} from './resize.js';
import {setupFolderInput} from './folder-upload.js';
import {initWorkspaces} from './workspaces.js';

document.addEventListener('DOMContentLoaded', function() {
    initWorkspaces();
    setupPanel();
    setupPanelResize();
    setupHtmxInteractions();
    setupImageInputs();
    setupFolderInput();
    setupQueryForm();
    setupMathRendering();
});

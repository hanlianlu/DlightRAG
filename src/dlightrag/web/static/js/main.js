// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {setupQueryForm} from './chat.js';
import {setupImageInputs} from './images.js';
import {setupPanel} from './panel.js';
import {initWorkspaces} from './workspaces.js';

document.addEventListener('DOMContentLoaded', function() {
    initWorkspaces();
    setupPanel();
    setupImageInputs();
    setupQueryForm();
});

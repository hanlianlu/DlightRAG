// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import '../tokens/utopia.css';
import '../styles/global.css';
import '../styles/layout.css';
import '../styles/files.css';
import '../styles/sources.css';

document.addEventListener('DOMContentLoaded', async function() {
    const [
        {setupQueryForm},
        {setupHtmxInteractions},
        {setupImageInputs},
        {setupMathRendering},
        {setupPanel},
        {setupPanelResize},
        {setupFolderInput},
        {initWorkspaces},
    ] = await Promise.all([
        import('./chat.ts'),
        import('./htmx.ts'),
        import('./images.ts'),
        import('./mathjax.ts'),
        import('./panel.ts'),
        import('./resize.ts'),
        import('./folder-upload.ts'),
        import('./workspaces.ts'),
    ]);

    initWorkspaces();
    setupPanel();
    setupPanelResize();
    setupHtmxInteractions();
    setupImageInputs();
    setupFolderInput();
    setupQueryForm();
    setupMathRendering();
});

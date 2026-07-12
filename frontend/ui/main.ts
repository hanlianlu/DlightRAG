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
        {setupSourcePanel},
        {setupFilesPanel},
        {setupPanelResize},
        {initWorkspaces},
    ] = await Promise.all([
        import('./chat.ts'),
        import('./htmx.ts'),
        import('./images.ts'),
        import('./mathjax.ts'),
        import('./panel.ts'),
        import('./source-panel.ts'),
        import('./files-panel.ts'),
        import('./resize.ts'),
        import('./workspaces.ts'),
    ]);

    initWorkspaces();
    setupPanel();
    setupSourcePanel();
    setupFilesPanel();
    setupPanelResize();
    setupHtmxInteractions();
    setupImageInputs();
    setupQueryForm();
    setupMathRendering();

    // Conversation storage is independent of workspace/file/source controls.
    // Bootstrap in the background; submitQuery retries if it is unavailable.
    const {conversationStore} = await import('../stores/conversationStore.ts');
    void conversationStore.initialize();

});

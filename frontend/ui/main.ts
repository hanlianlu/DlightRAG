// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

function versionedModule(path: string): string {
    const version = window.__DLIGHTRAG_STATIC_VERSION__ || 'dev';
    return `${path}?v=${encodeURIComponent(version)}`;
}

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
        import(versionedModule('./chat.ts')),
        import(versionedModule('./htmx.ts')),
        import(versionedModule('./images.ts')),
        import(versionedModule('./mathjax.ts')),
        import(versionedModule('./panel.ts')),
        import(versionedModule('./resize.ts')),
        import(versionedModule('./folder-upload.ts')),
        import(versionedModule('./workspaces.ts')),
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

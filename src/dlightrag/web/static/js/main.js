// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

function versionedModule(path) {
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
        import(versionedModule('./chat.js')),
        import(versionedModule('./htmx.js')),
        import(versionedModule('./images.js')),
        import(versionedModule('./mathjax.js')),
        import(versionedModule('./panel.js')),
        import(versionedModule('./resize.js')),
        import(versionedModule('./folder-upload.js')),
        import(versionedModule('./workspaces.js')),
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

// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {getPrimaryWorkspace} from './state.js';
import {showToast} from './panel.js';

async function traverseDirectory(entry, basePath) {
    const files = [];
    if (entry.isFile) {
        return new Promise(function (resolve) {
            entry.file(function (file) {
                file._relativePath = basePath ? basePath + '/' + file.name : file.name;
                files.push(file);
                resolve(files);
            });
        });
    }
    if (entry.isDirectory) {
        const dirPath = basePath ? basePath + '/' + entry.name : entry.name;
        const entries = await readAllEntries(entry);
        for (let i = 0; i < entries.length; i++) {
            const childFiles = await traverseDirectory(entries[i], dirPath);
            files.push.apply(files, childFiles);
        }
    }
    return files;
}

function readAllEntries(dirEntry) {
    return new Promise(function (resolve) {
        const reader = dirEntry.createReader();
        const all = [];
        function readBatch() {
            reader.readEntries(function (entries) {
                if (entries.length === 0) {
                    resolve(all);
                } else {
                    all.push.apply(all, entries);
                    readBatch();
                }
            });
        }
        readBatch();
    });
}

async function traverseFileSystemDirectory(handle, basePath) {
    const files = [];
    for await (const entry of handle.entries()) {
        const name = entry[0];
        const child = entry[1];
        const childPath = basePath + '/' + name;
        if (child.kind === 'file') {
            const file = await child.getFile();
            file._relativePath = childPath;
            files.push(file);
        } else if (child.kind === 'directory') {
            const childFiles = await traverseFileSystemDirectory(child, childPath);
            files.push.apply(files, childFiles);
        }
    }
    return files;
}

/**
 * Detect if drop contains folders using progressive enhancement.
 * imageHandler(file) is called for image files that should go to composer thumbnails.
 * Returns {files: File[], folderName: string|null}
 */
export async function detectDropItems(items, imageHandler) {
    const allFiles = [];
    let folderName = null;

    const supportsFileSystemHandle = 'getAsFileSystemHandle' in DataTransferItem.prototype;
    const supportsWebkitGetAsEntry = 'webkitGetAsEntry' in DataTransferItem.prototype;

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind !== 'file') continue;

        // Tier 1: File System Access API
        if (supportsFileSystemHandle) {
            try {
                const handle = await item.getAsFileSystemHandle();
                if (handle && handle.kind === 'directory') {
                    folderName = folderName || handle.name;
                    const dirFiles = await traverseFileSystemDirectory(handle, handle.name);
                    for (let j = dirFiles.length - 1; j >= 0; j--) {
                        if (dirFiles[j].type.startsWith('image/') && imageHandler) {
                            imageHandler(dirFiles[j]);
                            dirFiles.splice(j, 1);
                        }
                    }
                    allFiles.push.apply(allFiles, dirFiles);
                    continue;
                }
                if (handle && handle.kind === 'file') {
                    const f = await handle.getFile();
                    f._relativePath = f.name;
                    if (f.type.startsWith('image/') && imageHandler) {
                        imageHandler(f);
                    } else {
                        allFiles.push(f);
                    }
                    continue;
                }
            } catch (_) { /* fall through */ }
        }

        // Tier 2: webkitGetAsEntry
        if (supportsWebkitGetAsEntry) {
            const entry = item.webkitGetAsEntry();
            if (entry) {
                if (entry.isDirectory) {
                    folderName = folderName || entry.name;
                    const dirFiles2 = await traverseDirectory(entry, entry.name);
                    for (let j = dirFiles2.length - 1; j >= 0; j--) {
                        if (dirFiles2[j].type.startsWith('image/') && imageHandler) {
                            imageHandler(dirFiles2[j]);
                            dirFiles2.splice(j, 1);
                        }
                    }
                    allFiles.push.apply(allFiles, dirFiles2);
                    continue;
                }
                if (entry.isFile) {
                    const f2 = await new Promise(function (resolve) {
                        entry.file(function (file) {
                            file._relativePath = file.name;
                            resolve(file);
                        });
                    });
                    if (f2.type.startsWith('image/') && imageHandler) {
                        imageHandler(f2);
                    } else {
                        allFiles.push(f2);
                    }
                    continue;
                }
            }
        }

        // Tier 3: Plain file
        const file = item.getAsFile();
        if (file) {
            file._relativePath = file.name;
            if (file.type.startsWith('image/') && imageHandler) {
                imageHandler(file);
            } else {
                allFiles.push(file);
            }
        }
    }

    return {files: allFiles, folderName: folderName};
}

/**
 * Upload files to workspace, preserving directory structure.
 */
export async function uploadFolderToWorkspace(files, folderName) {
    if (files.length === 0) return;

    const formData = new FormData();
    formData.append('workspace', getPrimaryWorkspace());
    files.forEach(function (file) {
        const path = file._relativePath || file.name;
        formData.append('files', file, path);
    });

    const label = folderName || (files.length === 1 ? files[0].name : files.length + ' files');
    showToast('Uploading ' + label + '...');

    try {
        const resp = await fetch('/web/files/upload', {
            method: 'POST',
            body: formData,
        });
        if (resp.ok) {
            const html = await resp.text();
            const panelContent = document.getElementById('panel-content');
            if (panelContent) {
                panelContent.innerHTML = html;
            }
            showToast('Uploaded ' + label);
        } else {
            const text = await resp.text();
            showToast('Upload failed: ' + (text || resp.statusText));
        }
    } catch (err) {
        showToast('Upload failed: ' + err.message);
    }
}

/**
 * Set up folder input (webkitdirectory) and Files panel button delegation.
 */
export function setupFolderInput() {
    const folderInput = document.getElementById('folder-input');
    if (!folderInput) return;

    folderInput.addEventListener('change', async function () {
        const fileList = folderInput.files;
        if (!fileList || fileList.length === 0) return;
        const rawFiles = Array.from(fileList);

        let folderName = null;
        const augmented = rawFiles.map(function (f) {
            const path = f.webkitRelativePath || f.name;
            f._relativePath = path;
            if (!folderName && f.webkitRelativePath) {
                const parts = f.webkitRelativePath.split('/');
                folderName = parts[0];
            }
            return f;
        });

        await uploadFolderToWorkspace(augmented, folderName);
        folderInput.value = '';
    });

    // Delegated click for "Upload Folder" button in Files panel
    document.addEventListener('click', function (e) {
        const btn = e.target.closest('[data-action="upload-folder"]');
        if (btn) {
            folderInput.click();
        }
    });
}

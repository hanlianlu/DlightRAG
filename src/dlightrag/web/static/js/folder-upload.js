// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {getPrimaryWorkspace} from './state.js';
import {showToast} from './panel.js';

/**
 * Recursively traverse a directory entry, collecting all files.
 */
async function traverseDirectory(entry, basePath) {
    var files = [];
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
        var dirPath = basePath ? basePath + '/' + entry.name : entry.name;
        var entries = await readAllEntries(entry);
        for (var i = 0; i < entries.length; i++) {
            var childFiles = await traverseDirectory(entries[i], dirPath);
            files.push.apply(files, childFiles);
        }
    }
    return files;
}

/**
 * Read all entries from a directory reader (handles batch limits).
 */
function readAllEntries(dirEntry) {
    return new Promise(function (resolve) {
        var reader = dirEntry.createReader();
        var all = [];
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

/**
 * Traverse directory via File System Access API handles.
 */
async function traverseFileSystemDirectory(handle, basePath) {
    var files = [];
    for await (var entry of handle.entries()) {
        var name = entry[0];
        var child = entry[1];
        var childPath = basePath + '/' + name;
        if (child.kind === 'file') {
            var file = await child.getFile();
            file._relativePath = childPath;
            files.push(file);
        } else if (child.kind === 'directory') {
            var childFiles = await traverseFileSystemDirectory(child, childPath);
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
    var allFiles = [];
    var folderName = null;

    var supportsFileSystemHandle = 'getAsFileSystemHandle' in DataTransferItem.prototype;
    var supportsWebkitGetAsEntry = 'webkitGetAsEntry' in DataTransferItem.prototype;

    for (var i = 0; i < items.length; i++) {
        var item = items[i];
        if (item.kind !== 'file') continue;

        // Tier 1: File System Access API
        if (supportsFileSystemHandle) {
            try {
                var handle = await item.getAsFileSystemHandle();
                if (handle && handle.kind === 'directory') {
                    folderName = folderName || handle.name;
                    var dirFiles = await traverseFileSystemDirectory(handle, handle.name);
                    allFiles.push.apply(allFiles, dirFiles);
                    continue;
                }
                if (handle && handle.kind === 'file') {
                    var f = await handle.getFile();
                    f._relativePath = f.name;
                    allFiles.push(f);
                    continue;
                }
            } catch (_) { /* fall through */ }
        }

        // Tier 2: webkitGetAsEntry
        if (supportsWebkitGetAsEntry) {
            var entry = item.webkitGetAsEntry();
            if (entry) {
                if (entry.isDirectory) {
                    folderName = folderName || entry.name;
                    var dirFiles2 = await traverseDirectory(entry, entry.name);
                    allFiles.push.apply(allFiles, dirFiles2);
                    continue;
                }
                if (entry.isFile) {
                    var f2 = await new Promise(function (resolve) {
                        entry.file(function (file) {
                            file._relativePath = file.name;
                            resolve(file);
                        });
                    });
                    allFiles.push(f2);
                    continue;
                }
            }
        }

        // Tier 3: Plain file
        var file = item.getAsFile();
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

    var formData = new FormData();
    formData.append('workspace', getPrimaryWorkspace());
    files.forEach(function (file) {
        var path = file._relativePath || file.name;
        formData.append('files', file, path);
    });

    var label = folderName || (files.length === 1 ? files[0].name : files.length + ' files');
    showToast('Uploading ' + label + '...');

    try {
        var resp = await fetch('/web/files/upload', {
            method: 'POST',
            body: formData,
        });
        if (resp.ok) {
            var html = await resp.text();
            var panelContent = document.getElementById('panel-content');
            if (panelContent) {
                panelContent.innerHTML = html;
            }
            showToast('Uploaded ' + label);
        } else {
            var text = await resp.text();
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
    var folderInput = document.getElementById('folder-input');
    if (!folderInput) return;

    folderInput.addEventListener('change', async function () {
        var fileList = folderInput.files;
        if (!fileList || fileList.length === 0) return;
        var rawFiles = Array.from(fileList);

        var folderName = null;
        var augmented = rawFiles.map(function (f) {
            var path = f.webkitRelativePath || f.name;
            f._relativePath = path;
            if (!folderName && f.webkitRelativePath) {
                var parts = f.webkitRelativePath.split('/');
                folderName = parts[0];
            }
            return f;
        });

        await uploadFolderToWorkspace(augmented, folderName);
        folderInput.value = '';
    });

    // Delegated click for "Upload Folder" button in Files panel
    document.addEventListener('click', function (e) {
        var btn = e.target.closest('[data-action="upload-folder"]');
        if (btn) {
            folderInput.click();
        }
    });
}

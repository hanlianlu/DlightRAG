// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type RelativeFile = File & {
    _relativePath?: string;
    webkitRelativePath?: string;
};

interface WebkitFileSystemFileEntry {
    isFile: true;
    isDirectory: false;
    name: string;
    file(callback: (file: File) => void): void;
}

interface WebkitFileSystemDirectoryEntry {
    isFile: false;
    isDirectory: true;
    name: string;
    createReader(): {
        readEntries(callback: (entries: WebkitFileSystemEntry[]) => void): void;
    };
}

type WebkitFileSystemEntry = WebkitFileSystemFileEntry | WebkitFileSystemDirectoryEntry;

type DragItemWithHandles = DataTransferItem & {
    getAsFileSystemHandle?: () => Promise<FileSystemFileHandle | FileSystemDirectoryHandle | null>;
    webkitGetAsEntry?: () => WebkitFileSystemEntry | null;
};

interface DropDetectionResult {
    files: RelativeFile[];
    folderName: string | null;
}

export function withRelativePath(file: File, path: string): RelativeFile {
    const relativeFile = file as RelativeFile;
    relativeFile._relativePath = path;
    return relativeFile;
}

async function traverseDirectory(entry: WebkitFileSystemEntry, basePath: string): Promise<RelativeFile[]> {
    const files: RelativeFile[] = [];
    if (entry.isFile) {
        return new Promise(function (resolve) {
            entry.file(function (file) {
                files.push(withRelativePath(file, basePath ? basePath + '/' + file.name : file.name));
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

function readAllEntries(dirEntry: WebkitFileSystemDirectoryEntry): Promise<WebkitFileSystemEntry[]> {
    return new Promise(function (resolve) {
        const reader = dirEntry.createReader();
        const all: WebkitFileSystemEntry[] = [];
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

async function traverseFileSystemDirectory(
    handle: FileSystemDirectoryHandle,
    basePath: string,
): Promise<RelativeFile[]> {
    const files: RelativeFile[] = [];
    for await (const entry of handle.entries()) {
        const name = entry[0];
        const child = entry[1];
        const childPath = basePath + '/' + name;
        if (child.kind === 'file') {
            const file = await child.getFile();
            files.push(withRelativePath(file, childPath));
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
 * Returns {files: File[], folderName: string | null}
 */
export async function detectDropItems(
    items: DataTransferItemList,
    imageHandler?: (file: File) => void,
): Promise<DropDetectionResult> {
    const allFiles: RelativeFile[] = [];
    let folderName: string | null = null;

    const supportsFileSystemHandle = 'getAsFileSystemHandle' in DataTransferItem.prototype;
    const supportsWebkitGetAsEntry = 'webkitGetAsEntry' in DataTransferItem.prototype;

    for (let i = 0; i < items.length; i++) {
        const item = items[i] as DragItemWithHandles;
        if (item.kind !== 'file') continue;

        // Tier 1: File System Access API
        if (supportsFileSystemHandle && item.getAsFileSystemHandle) {
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
                    const f = withRelativePath(await handle.getFile(), handle.name);
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
        if (supportsWebkitGetAsEntry && item.webkitGetAsEntry) {
            const entry = item.webkitGetAsEntry() as WebkitFileSystemEntry | null;
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
                    const f2 = await new Promise<RelativeFile>(function (resolve) {
                        entry.file(function (file) {
                            resolve(withRelativePath(file, file.name));
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
            const relativeFile = withRelativePath(file, file.name);
            if (file.type.startsWith('image/') && imageHandler) {
                imageHandler(file);
            } else {
                allFiles.push(relativeFile);
            }
        }
    }

    return {files: allFiles, folderName: folderName};
}

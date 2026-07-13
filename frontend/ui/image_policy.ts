// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type CapabilityStatus = 'supported' | 'unsupported' | 'unknown';

export interface ImageAdmissionPolicy {
    effectiveCurrentUploadLimit: number;
    maxUploadBytes: number;
    capabilityStatus: CapabilityStatus;
}

export function canAddImage(input: {
    status: CapabilityStatus;
    limit: number;
    current: number;
}): boolean {
    if (input.status !== 'supported') return false;
    return input.current < input.limit;
}

interface ImageUploadCandidate {
    type: string;
    size: number;
}

interface ImageReadAdmissionOptions {
    getPolicy: () => ImageAdmissionPolicy | null;
    getCommittedCount: () => number;
    onReady: (file: File, dataUrl: string) => void;
    createReader?: () => FileReader;
}

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

function parseCapabilityStatus(value: string | undefined): CapabilityStatus | null {
    if (value === 'supported' || value === 'unsupported' || value === 'unknown') return value;
    return null;
}

export function getImageAdmissionPolicy(
    root: Pick<HTMLElement, 'dataset'> | null = document.getElementById('app'),
): ImageAdmissionPolicy | null {
    if (!root) return null;
    const effectiveCurrentUploadLimit = parseInteger(root.dataset.effectiveCurrentUploadLimit, 0);
    const maxUploadBytes = parseInteger(root.dataset.maxUploadBytes, 1);
    const capabilityStatus = parseCapabilityStatus(root.dataset.answerImageCapability);
    if (
        effectiveCurrentUploadLimit === null ||
        maxUploadBytes === null ||
        capabilityStatus === null
    ) {
        return null;
    }
    return {effectiveCurrentUploadLimit, maxUploadBytes, capabilityStatus};
}

export function acceptsImageUpload(
    file: ImageUploadCandidate,
    currentCount: number,
    policy: ImageAdmissionPolicy,
): boolean {
    return (
        canAddImage({
            status: policy.capabilityStatus,
            limit: policy.effectiveCurrentUploadLimit,
            current: currentCount,
        }) &&
        file.type.startsWith('image/') &&
        file.size <= policy.maxUploadBytes
    );
}

export class ImageReadAdmissionController {
    readonly #getPolicy: () => ImageAdmissionPolicy | null;
    readonly #getCommittedCount: () => number;
    readonly #onReady: (file: File, dataUrl: string) => void;
    readonly #createReader: () => FileReader;
    readonly #readers = new Set<FileReader>();

    constructor(options: ImageReadAdmissionOptions) {
        this.#getPolicy = options.getPolicy;
        this.#getCommittedCount = options.getCommittedCount;
        this.#onReady = options.onReady;
        this.#createReader = options.createReader || (() => new FileReader());
    }

    get inFlightCount(): number {
        return this.#readers.size;
    }

    admit(file: File): boolean {
        const policy = this.#getPolicy();
        const admittedCount = this.#getCommittedCount() + this.#readers.size;
        if (!policy || !acceptsImageUpload(file, admittedCount, policy)) return false;

        const reader = this.#createReader();
        this.#readers.add(reader);
        const release = (): boolean => this.#readers.delete(reader);
        reader.onload = (event): void => {
            if (!release()) return;
            const dataUrl = typeof event.target?.result === 'string' ? event.target.result : '';
            if (dataUrl) this.#onReady(file, dataUrl);
        };
        reader.onerror = function(): void { release(); };
        reader.onabort = function(): void { release(); };
        try {
            reader.readAsDataURL(file);
        } catch {
            release();
            return false;
        }
        return true;
    }

    clear(): void {
        const readers = Array.from(this.#readers);
        this.#readers.clear();
        readers.forEach(function(reader) { reader.abort(); });
    }
}

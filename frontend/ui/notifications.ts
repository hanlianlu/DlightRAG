// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {bus} from '../events/bus.ts';

const ASKED_STORAGE_KEY = 'dlightrag-notify-asked';

// Set while an answer streams with the tab hidden, so the offer only appears to
// someone who actually missed one.
let missedAnswer = false;
let streaming = false;

function supported(): boolean {
    // Absent outside a secure context, so a plain-HTTP deployment shows nothing.
    return typeof window !== 'undefined' && 'Notification' in window;
}

function alreadyAsked(): boolean {
    try {
        return window.localStorage.getItem(ASKED_STORAGE_KEY) === '1';
    } catch (_error) {
        return false;
    }
}

function rememberAsked(): void {
    try {
        window.localStorage.setItem(ASKED_STORAGE_KEY, '1');
    } catch (_error) {
        // Ignore unavailable or blocked storage.
    }
}

function notifyAnswerReady(): void {
    try {
        const notification = new Notification('Answer ready', {
            body: 'DlightRAG finished generating your answer.',
            tag: 'dlightrag-answer',
        });
        notification.onclick = function() {
            window.focus();
            notification.close();
        };
    } catch (_error) {
        // Some browsers reject construction outside a service worker.
    }
}

function hideOffer(): void {
    document.getElementById('notify-offer')?.setAttribute('hidden', '');
}

function showOffer(): void {
    const offer = document.getElementById('notify-offer');
    if (!offer) return;
    offer.removeAttribute('hidden');
}

async function acceptOffer(): Promise<void> {
    rememberAsked();
    hideOffer();
    try {
        await Notification.requestPermission();
    } catch (_error) {
        // Denied or unavailable; nothing further to do.
    }
}

function declineOffer(): void {
    rememberAsked();
    hideOffer();
}

export function setupNotifications(): void {
    if (!supported()) return;

    bus.on('conversationStreamChanged', function({active}) {
        if (active) {
            streaming = true;
            missedAnswer = document.hidden;
            return;
        }
        if (!streaming) return;
        streaming = false;
        if (document.hidden) missedAnswer = true;
        if (!document.hidden || !missedAnswer) return;
        if (Notification.permission === 'granted') notifyAnswerReady();
    });

    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            if (streaming) missedAnswer = true;
            return;
        }
        if (!streaming && missedAnswer && Notification.permission === 'default' && !alreadyAsked()) {
            showOffer();
        }
        missedAnswer = false;
    });

    document.getElementById('notify-offer-accept')?.addEventListener('click', function() {
        void acceptOffer();
    });
    document.getElementById('notify-offer-decline')?.addEventListener('click', declineOffer);
}

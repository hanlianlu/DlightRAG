// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {chatSessionStore} from '../stores/chatSessionStore.ts';

const ASKED_STORAGE_KEY = 'dlightrag-notify-asked';

// Set while an answer streams with nobody watching, so the offer only appears
// to someone who actually missed one.
let missedAnswer = false;
let streaming = false;

function away(): boolean {
    // Switching apps leaves the tab visible, so focus is the other half of this.
    return document.hidden || !document.hasFocus();
}

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
        // No tag: a shared one silently replaces the parked notification
        // instead of alerting again, so every answer but the first goes unseen.
        const notification = new Notification('Answer ready', {
            body: 'DlightRAG finished generating your answer.',
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
    hideOffer();
    try {
        // A dismissed prompt leaves the choice open, so the offer may come back.
        if (await Notification.requestPermission() !== 'default') rememberAsked();
    } catch (_error) {
        rememberAsked();
    }
}

function declineOffer(): void {
    rememberAsked();
    hideOffer();
}

export function setupNotifications(): void {
    if (!supported()) return;

    chatSessionStore.subscribe(function() {
        const active = chatSessionStore.active;
        if (active) {
            streaming = true;
            missedAnswer = away();
            return;
        }
        if (!streaming) return;
        streaming = false;
        if (!away()) return;
        missedAnswer = true;
        if (Notification.permission === 'granted') notifyAnswerReady();
    });

    function leftPage(): void {
        if (streaming) missedAnswer = true;
    }

    function cameBack(): void {
        // Returning fires visibilitychange before focus, so re-check both.
        if (away()) return;
        if (!streaming && missedAnswer && Notification.permission === 'default' && !alreadyAsked()) {
            showOffer();
        }
        missedAnswer = false;
    }

    window.addEventListener('blur', leftPage);
    window.addEventListener('focus', cameBack);
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) leftPage();
        else cameBack();
    });

    document.getElementById('notify-offer-accept')?.addEventListener('click', function() {
        void acceptOffer();
    });
    document.getElementById('notify-offer-decline')?.addEventListener('click', declineOffer);
}

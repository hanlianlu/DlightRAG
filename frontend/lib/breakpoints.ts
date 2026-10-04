// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Browser media contracts shared by Shell and Artifact Canvas behavior. */
export const COMPACT_SHELL_MEDIA = '(width < 1200px)';
export const DESKTOP_SHELL_MEDIA = '(min-width: 1200px)';
export const MOBILE_MEDIA = '(max-width: 640px)';
/** A full-screen dialog replaces the centered one: a phone, or any viewport too short for it.
 *  The Settings stylesheets repeat this query for their layouts; script reads it for what only
 *  script can decide, such as where focus starts and which words a button wears. */
export const PHONE_DIALOG_MEDIA = '(width <= 720px), (height <= 480px)';

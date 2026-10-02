// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Corpus Mutations the server offers, as browser tests describe workspaces. */

import {WORKSPACE_CHANGES} from '../api/workspaces.ts';

/** A workspace its caller may change in every way. */
export const EVERY_CHANGE = WORKSPACE_CHANGES;

/** The deployment default: no one deletes it. */
export const DEFAULT_CHANGES = WORKSPACE_CHANGES.filter((change) => change !== 'delete_workspace');

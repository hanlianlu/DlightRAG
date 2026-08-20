// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {createBrowserRouter} from '../lib/router.ts';

/** One browser navigation owner shared by the shell and answer submission. */
export const webRouter = createBrowserRouter(window);

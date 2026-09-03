// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export {icon, type IconOptions, type IconSize} from './icons/icon.ts';
export {
  ICON_REGISTRY,
  ICON_SOURCES,
  type IconName,
  type IconSource,
  type IconSourceMetadata,
} from './icons/registry.generated.ts';
export {defineDesignSystemElements} from './elements/define.ts';
export {DlIconButton} from './elements/icon-button.ts';
export {DlMenu} from './elements/menu.ts';
export {
  DlSplitLayout,
  type SplitOrientation,
  type SplitPositionDetail,
  type SplitPrimary,
} from './elements/split-layout.ts';

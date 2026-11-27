# Web Theme Design

The Web UI offers a `System / Light / Dark` appearance preference: the dark theme
and the warm-neutral **Mineral Light** palette, correct on the first painted
frame, with role-based Soft geometry and token-bridged split panels. There are no
custom palettes, account-level preferences, or server-side preference storage.

## Product Decisions

- New users default to `System` and follow the operating-system preference.
- Explicit `Light` or `Dark` selection overrides the system preference.
- `System` responds to operating-system changes while the page is open.
- The light palette is Mineral Light: warm stone surfaces, graphite text, and a darker accessible gold accent.
- The control is an icon button to the right of `Files` in the topbar.
- The trigger shows a Lucide `Moon` in the effective dark mode and `Sun` in the effective light mode.
- The menu uses Lucide `Monitor`, `Sun`, and `Moon` icons for `System`, `Light`, and `Dark`.

## Architecture Constraints

- Keep theme state dependency-free. Do not add a theme framework, state-management layer, or external icon runtime.
- Prefer semantic CSS tokens over component-specific light-mode overrides.
- Reuse the existing popover dismissal and keyboard-navigation infrastructure.
- The package-owned design system's runtime CSS remains authoritative and is projected deterministically for design tooling.
- The design system's reset sets every page's body text: Body text on the page background, in `--font-body`, antialiased. A page's own stylesheet adds only its chrome, such as the application's clipped viewport or the catalog's 390px minimum width.
- Every page loads only the stylesheets its own entry imports, so one page's chrome never reaches another.
- Native Drawer/Dialog behavior remains product-owned; split behavior belongs to the package-owned design system.

## State Model

The root element carries both the stored preference and the effective color mode:

```html
<html data-theme="system" data-color-mode="dark">
```

- `data-theme`: `system | light | dark`; this is the user preference.
- `data-color-mode`: `light | dark`; this is the currently rendered appearance.

The preference is stored in local storage under `dlightrag-theme`. Theme persistence is a browser-only presentation concern; no API endpoint, cookie, database column, or server request state is required.

Vite emits `frontend/theme-init.ts` as a dedicated hashed classic script. The static `<head>` loads it before any stylesheet or application module. It validates the saved preference, resolves `System` with `matchMedia('(prefers-color-scheme: dark)')`, and updates both root attributes. The HTML defaults to `system + dark`, so any bootstrap failure keeps the safe dark appearance.

The document also declares native `color-scheme` support. The effective mode controls form controls, scrollbars, and browser-owned UI consistently with the page.

## Runtime Module

A focused `frontend/ui/theme.ts` module owns runtime behavior:

- parse and validate the stored preference;
- resolve the effective color mode;
- update root attributes and native `color-scheme`;
- persist an explicit selection when storage is available;
- update trigger icons, menu selection, and ARIA state;
- listen for system color-scheme changes only while `System` is selected;
- listen for the browser `storage` event to synchronize other tabs;
- degrade to an in-memory selection when local storage is unavailable.

Pure preference parsing and color-mode resolution remain separate from DOM wiring so the meaningful state rules are easy to test without a simulated browser.

No nanoevents bus or dedicated store is needed: theme state has one owner and only mutates `<html>` plus its own control.

## Theme Control

The topbar trigger is a square, borderless icon button sized with `--size-button`. Both decorative SVGs are present in the Vite/Lit application shell; CSS selects the correct one from `data-color-mode`, so the first frame never shows an empty or stale icon.

The trigger has:

- accessible name and tooltip `Appearance`;
- `aria-haspopup="menu"`;
- synchronized `aria-expanded`.

The popover uses:

- `role="menu"`;
- three `role="menuitemradio"` choices;
- synchronized `aria-checked` and a visual checkmark;
- `Monitor System`, `Sun Light`, and `Moon Dark` rows.

Interaction behavior (the product-wide menu contract):

- click, ArrowDown, Enter, or Space opens the menu on the first choice;
  ArrowUp opens it on the last;
- ArrowUp and ArrowDown move with wrapping, Home and End jump to the ends, and
  a typed letter moves to the next choice whose label starts with it;
- Enter or Space applies a choice and closes the menu;
- Escape closes and restores trigger focus;
- outside pointer click closes while preserving the clicked target's natural focus;
- a choice applies immediately without reload or a server request.

The generic popover dismissal helper owns the menu lifecycle. The design
system's `dl-menu` element applies the in-menu keys, and its
`menuButtonFocus()` maps the button's keys, so the theme, answer mode, agent
effort, and conversation actions menus behave identically. The workspace and
file pickers are dialogs, but reuse the same `rovingFocusKeydown()` step.

Semantic icon geometry is generated from the pinned, generation-only `lucide-static` package into the checked-in design-system registry. Production bundles retain only selected geometry rendered with `currentColor`; controls own accessible names and the repository NOTICE records the Lucide license. No runtime icon dependency is loaded. The stroke weight is `1.5` stated in the glyph's own 24-unit viewBox rather than in CSS pixels — a quarter lighter than Lucide's `2` — and the 12px step keeps a `2`-unit floor, so the ladder carries 1.00/1.00/1.25/1.50px of ink and only the 12px step states a weight of its own. One 24-unit master geometry serves every step: the weight is a function of the size and no step ships a second optical drawing, because one grid and one weight rule are what keep every mark in a single design language. The 16/20px steps therefore render at a fractional scale (2/3, 5/6) and take the engine's antialiasing; that is the accepted cost of scaling one master grid, not a defect to patch per size.

A semantic icon whose state varies keeps one glyph per state (the theme control's `Monitor`/`Sun`/`Moon`). Geometry a state ladder needs and Lucide does not ship is authored in the same registry as a derived `dlightrag` entry that keeps the Lucide shape: the agent effort ladder is Lucide's `gauge` dial with the needle at each level's stop, and `status-dot` is a product coin. Those entries are generated, reviewed, and tested like the selected Lucide ones, and no state-bearing glyph ever carries accessible meaning on its own.

## Color System

Theme-specific values live only in the token layer. Components consume semantic aliases.

Core palette:

Surfaces, borders, tints and the gold accent are Tailwind stone and gold,
mirrored by step, so a value's place in the ramp can be read rather than measured.
**Text is measured instead**, in both themes: the reading roles are an ink ladder
whose names state the APCA level each step reaches on that theme's page, because
mirroring stone by step put the dark top two roles above APCA's dark-mode maximum,
the dark caption below its content minimum, and the light top two roles on one
ink ([below](#why-the-reading-ladder-is-measured)).

| Semantic role | Dark | Mineral Light |
|---|---|---|
| Page background | `#0c0a09` stone-950 | `#fafaf9` stone-50 |
| Primary surface | `#1c1917` stone-900 | `#f5f5f4` stone-100 |
| Elevated surface | `#292524` stone-800 | `#e7e5e4` stone-200 |
| Primary text | `#e5e3e1` ink-90, \|Lc\| 90 | `#1c1917` ink-100, \|Lc\| 99 |
| Body text | `#ccc9c6` ink-75, \|Lc\| 75 | `#44403c` ink-87, \|Lc\| 87 |
| Muted text | `#b3aea9` ink-60, \|Lc\| 60 | `#57534e` ink-80, \|Lc\| 80 |
| Subtle marks | `#78716c` stone-500 | `#78716c` stone-500 |
| Dim, disabled only | `#57534e` stone-600 | `#a8a29e` stone-400 |
| Primary accent | `#d2b661` gold-200 | `#7e6c37` gold-400 |
| Accent text | `#d2b661` gold-200 | `#574a24` gold-500 |
| Focus ring | gold-200 at 64% | gold-500 at 80% |
| Danger | `#f87171` | `#b91c1c` |

### Why the reading ladder is measured

WCAG 2's contrast ratio is symmetric: it reads white-on-black and black-on-white
as the same number, so it cannot tell a dark theme that it has gone too far, and
it cannot see a caption that has gone too faint either. APCA (`Lc`, the model
WCAG 3 is built on) is polarity-aware, and it found four things in this palette.
Its ceiling is **`|Lc|` 90**: past that the luminance step at a glyph's edge is
what a reader calls glare, and small text gains nothing from more. Its floor for
reading text is `|Lc|` 60 for content that is not a column of body text, and 75
for a column of it.

Dark, read as text on `#0c0a09`, put Primary and Secondary at 100.6 and 91.1 — over
the ceiling — and Muted at 54.0, under the floor that WCAG's 4.5:1 could not see
it missing. Light, read on `#fafaf9`, put Primary and Secondary at 99.3 and 96.3:
three levels apart, so the two roles were one ink, and the hierarchy those two
names promise did not exist. Its Muted was fine on the page (80.0) but not on an
elevated card (58.8), because the light surfaces spanned three stone steps: the
same body colour lost **21 `|Lc|`** between the page and a card, where Material 3's
tone-based surfaces lose about five.

So each theme cuts its four reading roles from its own ladder, each step measured
on the page to the level its name states, and the light surfaces sit two steps
apart instead of three. Dark takes ink-90/82/75/60, under the ceiling. Light takes
ink-100/92/87/80: near-black ink is what paper wants and positive polarity has no
published ceiling, so its ladder is only held to the floors — including on the
brightest panel, where it now reads 87.8 / 80.4 / 75.8 / 68.5.
`frontend/tokens/ramp.test.ts` recomputes APCA and fails if a name drifts more than
two levels, if either ladder falls out of rank order, if the dark top passes 90, or
if the light quiet end drops under 60 on an elevated surface. The steps keep
stone's warm tint; three of the light four land on stone values anyway, and only
Secondary needed a measured ink. Subtle and Dim are not reading roles — one is a
mark held to the 3:1 non-text floor, the other a disabled control — so they stay on
the stone step those floors were measured against.

Muted is the faintest text role that clears WCAG AA, 4.5:1, on every surface in
both modes (6.0:1 at worst in dark, 6.1:1 in Mineral Light), and it is the role
held to APCA's `|Lc|` 60 floor for content text. Subtle (`--color-text-subtle`) is the ramp's midpoint, so the mirror
leaves it stone-500 in both modes, and it reaches only about 3.2:1 on an
elevated surface. It colours non-text marks held to 3:1, such as ghost icon
buttons at rest and unselected radio rings, and never text: a caption or label
that would sit at Subtle takes Muted and keeps its rank through size, weight, or
case.

Dim (`--color-text-dim`) sits one step below Subtle and clears neither floor:
1.99:1 on an elevated surface in dark and 2.01:1 in Mineral Light. WCAG sets no
contrast for an inactive control, so Dim colours only a disabled control, such as
the Send button's glyph while there is nothing to send. The shared primitives
(`.dl-btn`, `.dl-switch`, `.dl-icon-button`, and the `<dl-icon-button>` element)
fade a disabled control to half opacity instead, which keeps the colours it
already has; `frontend/ui/primitives.layout.browser.test.ts` holds the four
to one opacity. Placeholder and ghost
text are still text and take Muted, a clear step fainter than the Primary text a
person types. A mark that shows a state, such as an idle status light, a
disclosure chevron, or an unchecked switch thumb, takes Subtle.

Accent text (`--color-accent-text`) is the gold for text, such as Fork under an
answer, a reason the reader has chosen under it, the reference ids, Load older
in the conversation list, the All workspaces row, the ingest target pill, and
the Mine badge in the skill menu. On
a plain surface it reaches AAA's 7:1 in dark (7.66:1 on an elevated surface) and
on every Mineral Light surface: 8.34:1 on the page, 7.99:1 on a panel, 6.94:1 on
an elevated one. Small text
often sits on a tint, a hovered row or a pill's accent fill, so the role clears
4.5:1 on every surface and under every tint in both modes: gold-200 in dark, the
Primary accent itself (5.73:1 at worst), and gold-500 in Mineral Light, one step
darker (5.61:1 at worst). `frontend/ui/geometry.browser.test.ts` enforces that
floor. One step quieter, gold-300 read 4.26:1 under the ingest pill's hover tint
in dark, and in Mineral Light gold-400 reads only 4.09:1 on an elevated surface.
The composer's attach glyph, the create-workspace glyph, and the thumbs under an
answer take it too, well above the 3:1 a mark needs.

In Mineral Light the Primary accent (`--color-accent-action`) reads 4.92:1 on
the page background but 4.71:1 on a panel and 4.09:1 on an elevated surface, so
it colours no text. It colours marks held to 3:1 on every surface, such as the
underline under an answer's link, the answer's status dot, checked radios, the
theme menu's check, and the ingest target's dot. Fork and Retry under a history
image sit only on the page, where Accent text reads 8.34:1 against the Primary
accent's 4.92:1. Mineral Light has no darker gold for their hover, so they
underline under the pointer.

The focus ring (`--color-control-ring`, aliased as `--focus-ring-color`) is a
2px outline drawn 2px outside its control, so it reads against the surface or
row tint around the control, not the control's own fill. It is held to 3:1 on
every surface and row tint in both modes (3.68:1 at worst in dark, 4.33:1 in
Mineral Light), which covers both its contrast with adjacent colours and its
change from the unfocused state. `frontend/ui/geometry.browser.test.ts`
enforces that floor.

The composer is the one control that shows focus by its border and a soft halo
rather than the outline: its border takes the ring colour, and the halo is the
accent ghost around it. The border is held to the same 3:1, measured as the
change from the border it has at rest, on every surface in both modes (3.40:1 at
worst in dark, 3.35:1 in Mineral Light), and the same test enforces it. The
softer accent border it replaced moved only 1.9:1 in dark and 1.1:1 in Mineral
Light, so the blinking caret was the only sign of focus.

The stone ramps keep perceptual surface steps comparable; the gold
ramp supplies an accessible accent in each mode. Borders and row tints use
low-alpha stone values. `frontend/tokens/ramp.test.ts` enforces ramp membership,
elevation direction, both contrast floors, both reading ladders, and that only a
disabled control takes Dim.

Docked panels use tone plus a hairline border, not shadows. Only overlapping
popovers, menus, dialogs, and toasts cast shadows. Components consume semantic
aliases; primitive palette values remain private to the token layer.

Spacing and typography remain unchanged by the theme. Geometry and panel behavior follow the separate role rules below.

Type and spacing are two scales that behave differently. The type steps
(caption, detail, body, subhead) are fluid: body text is 15px at a 390px
viewport and 15.5px at 1440px. The spacing steps (`--space-3xs` to `--space-l`)
are fixed rem values, 0.25, 0.5, 0.75, 1, 1.5 and 2rem, at every width; the
interface was tuned at those values, and a spacing token that claimed to flow
would state something nothing renders. Every `font-size` outside rendered
Markdown takes a type token, and stylelint rejects a raw `px`, `rem` or `em`
there. Markdown headings and inline code stay in `em` because they scale with
the text they sit in.

Under a touch pointer (`(pointer: coarse)`) every text-entry control (a text-like
`input`, `textarea`, `select`) is at least 16px (`--font-size-touch-entry`), because
iOS Safari zooms the page in when a smaller field takes focus and leaves it
zoomed. The one rule sits in the design system's utilities layer, so it outranks
the sizes the primitives and features give their own fields, and the viewport
stays zoomable. The composer's mirror and the minimum height of its input line
follow `--composer-text-size`, which takes the same floor, so the line holds its
text. `frontend/ui/geometry.browser.test.ts` applies the touch rules and
measures the fields.

## Geometry And Panels

Geometry follows surface role rather than component size. Controls use 10px,
cards and rich-content containers 16px, popovers 18px, dialogs 22px, and the
composer 24px. Pills remain `999px` and circles remain 50%. Full-viewport app
shells, docked sidebars and panels, structural sections, and internal seams stay
square at every viewport. This keeps Soft contained surfaces from rounding the
application silhouette or opening dark corner wedges.

Inspector and Artifact Canvas use nested local `dl-split-layout` elements on
wide screens. The design-system element owns axis layout, pointer and keyboard
input, and separator ARIA; the app adapter owns open state, breakpoints,
clamping, and persistence. Inspector and Artifact Canvas persist separate
preferred pixel widths; clamping for the conversation sidebar and minimum chat
width never overwrites those preferences. A single token-backed hairline has an
invisible 12px hit area.

Opening an Artifact citation is Shell-mediated. On desktop, Side remains Side,
Wide remains Wide, and Fullscreen reduces only to Wide while Sources opens. On
compact screens, the Canvas closes and Sources opens in the Inspector drawer.

The Inspector shows one content at a time: Files, Sources, or one Run's Agent
traces. Agent traces is the content the reader watches beside the chat, so a
click in the chat leaves it open where it would close Files or Sources (a Canvas
still closes). Below 1200px the Inspector is a drawer over the chat, and there
the click on the scrim closes Agent traces like the others; Escape and its close
button always close it. It measures its own pane rather than the viewport:
narrower than 40rem it shows the list of agents or one agent, and from 40rem
the list beside the agent on show, which opens on the main agent. On a phone it
is the Inspector's full-bleed sheet.

Agent traces is one roster for a Run's whole agent family: the main agent
first, and its children indented beneath it on a guide line. A Run that started
no child has no list, only the main agent. Every agent has the same page: its
objective, its state with how many tokens it used, its Result with an Evidence
fold, and its Activity. The main agent's page is a child's without the commands
(its Run is steered from the chat): its objective is the question, its Result
the answer or why the Run failed, and its Evidence the sources the answer cites.
Evidence lines open their source in Sources when the answer cites them, for the
main agent and for a child alike. Evidence and Activity each scroll in a pane of
their own, as the Result does when it is long. Activity is one timeline, newest
page first, with the earlier steps on request, so a reader can go back through
all of it; the main agent's spans the whole conversation. The fold is open while
the agent works, its pane opens on the newest step, and it keeps that step in
view for a reader who is at the bottom.

Below 1200px resizing is disabled and the panel is an overlay: the primary app
remains full viewport width under the scrim, while modal focus, inert state,
Escape, and focus restoration remain native DlightRAG behavior. At phone widths
the active panel becomes full bleed. External Drawer and Dialog components remain
rejected; existing native overlays keep these geometry rules.

## Motion

Every transition and animation names its duration with a `--duration-*` token from
`frontend/design-system/foundations/motion.css`, and stylelint rejects a numeric
duration and `transition: all`. The four change durations, instant (100ms),
control (150ms), overlay (200ms), and shell (180ms, the conversation sidebar
sliding in), are zeroed under `prefers-reduced-motion`, so a colour change and a
slide stop together. The three loop periods, spin (0.7s), pulse (1.2s), and slow
pulse (1.6s), are not zeroed, because a loop with no period freezes or flashes;
each loop states what it does for a reader who asked for no motion instead. The
marks that breathe stand still, and the file spinner breathes in place of turning.

The marks that breathe share one keyframes, `dl-pulse` in the primitives. Each
mark states its own depth through `--pulse-start-opacity`, `--pulse-mid-opacity`,
and `--pulse-mid-scale`; unstated, it is the status dot's breath. A CSS Module
names it `global(dl-pulse)`, because a module renames every animation name it
uses. `frontend/tokens/motion.test.ts` holds the durations, the single definition,
and the reduced-motion rule of every file that loops.

## Answer Feedback

The reader judges an answer with two thumbs in the action row under it: Agent
traces, Fork, the thumbs, then what the Run spent, each part set off from the
next by the row's hairline — the shared `.dl-action-row`, which owns that line,
the measure on either side of it, and the height it takes. Their element is no box
of its own, so the pair is hoisted onto the row's line and declares itself one of
the row's parts (`data-action-part`): the row draws its line and stands it off,
and all the thumbs state is how far apart they stand. So the line beside them is
as tall as the words on it and as far from them as every other line in the row is,
and no feature draws a divider for such a row at all. Only a succeeded
answer that the server has named has them (a failed or cancelled Run is settled
too, and gets none; a turn that has just finished waits for the history that
states which message its answer is), and only for a reader the `answer_feedback`
Runtime Policy offers them. What a child answered is judged in the Child agents
dock, at the right end of the status row of its page once it has succeeded; the
main agent's page has no thumbs, because its answer is judged in the chat.

A thumb is a mark in Accent text, tinted behind it under the pointer, and a
pressed thumb is the filled glyph, so the state is not carried by colour alone.
In the action row that mark is the size of the marks beside it — a 12px dot and
caption words — so the thumb holds a 12px mark in a box of that same size: its
ink stands where the box does and the row's measures — which are measures
between boxes — hold for it too. The weight is a measure of the glyph's
24-unit viewBox, not of CSS pixels, and the 12px floor keeps this mark at one CSS
pixel of ink in a line of words. The 24px the reader presses reaches
out invisibly around the box (`.dl-icon-button--dense`), the pair is set apart by
the row's own measure, and the row stays as tall as its words. In a
child's status row the thumb is drawn at its neighbours' size, among controls
that are `--size-button` tall.
A thumb down opens its reasons: five pills in a wrapping row that takes a line of
its own under the actions (in the dock, inside the child's page), and a note
field beneath them. At rest a reason is Muted words in a strong hairline border;
a chosen one is Accent text on the accent fill and border, and holds a check, since
those colours sit close to the resting pill's and colour alone would not say it.
A reason not chosen answers the pointer with the tint every control has and
borrows nothing of the chosen palette. A thumb down says why in a reason or in the
reader's own words, and a reason is not required: choosing one keeps the rating and
never closes the reasons itself, so that reader has no Submit to press. Choosing a
reason that is already chosen takes it away and leaves them open, storing the rest
without a receipt, and taking away the last one takes the whole rating back only
where the reader has given no words, because a thumb down that says neither is no
judgement. A note typed before the first reason is kept with it, and a note the
judgement does not already hold — on a thumb down just opened as much as on a stored
one, and after the last reason has gone — is sent by a small Send at the field's
corner, round and in the composers' own face, which keeps the reasons already stored
with the new note; Enter sends it and Shift+Enter breaks the line. It is never shown
for a reader who just clicks a reason, and its room is the field's padding, so it
moves nothing when it comes; the corner is its own, so the field has no resize grip.
Nothing is shown as kept until the server has said so. A control waits dimmed while
a request is out, Send with a dot in place of its plane, and stays focusable so the
reader keeps their place; where Send goes with the note it kept, the reader who was
on it lands on the note field.

What closes the reasons is the reader's own pause: three seconds after their last
act on one of the controls — a thumb, a pill, the note field, Send — restarted by
every such act, so a choice never takes the form away from a reader still working in
it. A press on the space around the controls is not an act, and a reader who has only
opened the reasons is still choosing: nothing closes them until they have acted once.
The pause closes them only where it would throw nothing away — a request that is out
is waited for, and words the judgement does not hold are the reader's to keep until
they send them. Three things close them at once instead, with no pause at all:
Escape, a thumb up, and pressing the lit thumb down, which takes the whole rating
back.

Every control is at least 24px each way, and the product's hit target (44px)
where a finger is the pointer: at 640px and narrower, on a coarse pointer, and in
the dock under 1200px. Focus is the 2px ring. A receipt is the app's toast, which
fades out linearly in 1.5 seconds and has no action; a rating the server refuses
is told in the error toast (see Toasts). The page announces the outcome through
the toast regions and nowhere else. A rating the server refuses
as an authorization failure, 403 in its error envelope with the `auth` error type
and no error kind, means the operator has turned feedback off since the page
loaded: the reader's focus moves to what stays beside the thumbs (Fork in the
action row, the child's title in the dock), and the Shell stops offering the
thumbs everywhere for as long as the page lives. A 403 without that
envelope, from a proxy or a firewall, and one that names a kind, such as the
origin guard's, are ordinary errors and leave the thumbs where they are.

## Memory Notice

What the Run left in Profile Memory is said under its answer, as the last part of the action row: a bookmark and
"Remembered 2" ("记住了 2 条"), in the row's own words (caption size, Accent ink, underlined under the pointer)
behind the row's hairline. The Agent writes memory with its tools while it answers and the Session Memory Pass a
few seconds after
([ADR 0045](adr/0045-profile-memory-is-maintained-per-session-and-owned-across-them.md)); neither asked the
reader, so the answer says what they kept, both writers together. Like the thumbs it is no box of its own and
declares itself one of the row's parts (`data-action-part`), so the row draws its line and no feature draws a
divider for it. It is a part and not a toast because it belongs to one answer and is there when the reader comes
back to it; it is no floating element beside the reading column because the column is where the eye already is.
A Run that left nothing, an owner who turned Profile Memory off (the server lists none then), and a failed read
all draw nothing.

The count is the changes still standing, so one taken back, or replaced by a later correction, is not counted.
The part is a button with `aria-haspopup="dialog"` and a synchronized `aria-expanded`. It opens the product's
anchored popover (`TriggerPopover`), titled "Remembered from this conversation": one row for each change, with its
kind (Preference, Fact, Experience) in a small pill, the words and Undo, and a footer, "Manage memory in
Settings", that opens Settings on Profile Memory. Opening puts focus on the first Undo; Escape closes it and gives
focus back to the part, and so does a press outside or focus moving on. It starts where the part starts and is moved
along the line no further than keeps it whole inside the chat's own box, never over the conversations or a
panel beside it, and narrowed to that box where the box is narrower than the card; it opens below the part when
the room under it holds it, so the answer's words stay in view, else above. The chat's own box is the room, not
the window, and it is placed again if the window is resized under an open popover. At 640px and narrower, and on a coarse pointer, it is a sheet from the bottom edge as wide as the screen,
and the part and every control in it are the product's hit target (44px); on a desktop each is at least 24px, the
part reaching that through the invisible extension the thumbs use.

Undo works as it does in Settings: an addition is taken back by forgetting its record, a correction or a merge by
undoing its change. The receipt is the app's toast: "Forgot: …" with an Undo of its own for an addition,
"Restored: …" for a correction and "Profile Memory restored." for a merge, which offers none, since an undo cannot
itself be undone. A refusal is an error toast and leaves the change standing to be tried again. A change taken
back stays in its place with "Undone" where its button was, so the row does not vanish under the reader's hand,
and when every change is taken back the part reads "Undone" and stays. The Undo of a receipt that brings a
forgotten record back lets that row go, because the record is then the reader's own restoring and no longer this
Run's. Focus on a button that has gone moves to the next Undo, or to the footer.

It arrives late. After a Run settles on the page the notice reads the list at 3, 8 and 20 seconds, only while the
page is visible (a try that fell due while it was hidden is made when it is shown), and stops once a record the
pass wrote is among them or after the last try; a Run whose own Agent wrote, which a live memory event says, is
read the moment it settles as well. An answer already on the page when it opened is read once when its turn is
first drawn, and the chat draws only the turns in view. No realtime channel carries it. A part that arrives while
the reader is here fades in over the overlay duration without moving the words around it (it simply appears for a
reader who asked for no motion) and is announced once, politely, through a status region that existed before the
news, without taking focus; one drawn with the page is neither animated nor announced.
`ui/memory-notice.browser.test.ts` holds the behaviour, `ui/memory-notice.layout.browser.test.ts` how the shipped
stylesheets draw it and that it never leaves the chat's box, and `ui/chat-feature-memory-notice.browser.test.ts` when the chat looks.

## Toasts

The app tells a reader how a command went through a toast, and a toast has one of
two tones, each shown in a region of its own. A receipt reports something done: it
sits at the bottom of the screen on one line, fades out after 1.5 seconds, and may
offer Undo. An error reports something that failed or was refused: it sits at the
top, so it never covers the composer the reader retries from, wraps its words, goes
after eight seconds or when the reader presses its close button, and wears
`--color-danger` on its border. A command that fails or is refused requests the
error tone (`tone: 'error'` on `dl-toast-request`); a request with no tone is a
receipt.

The receipt region is `role="status"`, announced politely, and the error region is
`role="alert"`, announced at once, so an error is announced once and not at a
receipt's priority. A new toast replaces the one of its own tone and leaves the
other standing: an error stays up when a receipt follows it, and a second error
replaces the first. Hovering a toast or focusing inside it stops its clock, and the
clock goes on with the time it had left. While a Shell modal keeps the reader from
it, an error or a receipt with an Undo stops its clock too, so neither runs out
unseen; a plain receipt passes. An Undo that fails takes its receipt away and tells
the failure as an error.

A modal stops an error's clock but does not put the error out of reach: it is not
made inert with the parts of the Shell the modal holds, so its alert and its close
button stay in the accessibility tree and the tab order. The Image Lightbox, which
makes the whole page inert, is the exception: the error waits behind it, its clock
stopped, until it closes. An error dismissed from the keyboard gives focus back to
where it was before it came into the toast, or, where that place is gone, to the
Chat's box; a pointer leaves focus alone. The Shell and the Settings dialog each
hold both regions; the dialog's sit in its top layer, because the Shell's would sit
under the scrim. While Settings is open every notice,
whoever raised it (a page of the dialog, or the sidebar reporting a delete-all that
failed), shows in the dialog's regions, and what still awaits the reader when it
closes, an Undo or an error, goes on in the Shell's. `ui/toast.browser.test.ts`
holds the tones, the timing, and the focus a dismissal returns, and
`ui/toast.layout.browser.test.ts` how the shipped stylesheets draw them, with axe's
verdict on an open error in both color modes.

## Scheduled

The owner's Schedules (ADR 0103) live in the conversation sidebar, not in Settings: a Scheduled section
between "New chat" and the conversation list, so recurring work sits beside the conversations it writes.

- **Section.** A header row (a clock, "Scheduled", the count, a disclosure mark, and a "New schedule"
  icon button) toggles the rows open and closed. The choice is this browser's (`localStorage`); with none
  stored the section is open once there is a Schedule, and with none the header stands alone with its
  "+". The rows scroll inside the section past a few, so they never push the conversations out of the
  column.
- **Row.** A Schedule is a conversation row (the global `.conversation-row` family, so its density,
  tint and actions menu are theirs): its name, and a muted line with the cadence in words and either
  "Next: {when}" on the Schedule's own clock or "Paused" with a pause mark. Run now, Pause or Resume, and
  an actions menu (Edit, Delete) stand over the row's end on hover and focus, and the row's text gives
  them the room; on a touch screen a row opens its detail, which carries every action.
- **Dialogs.** The form and the detail are the confirm dialog's face at 35rem: a title, a body that
  scrolls, and a footer that stays. They stand outside the sidebar's `<nav>`, as the conversation dialogs
  do, so the drawer's focus trap never sees them. The form asks a name, a prompt and when (one frequency
  choice with the few controls it needs on its line, all at the one control height), reads the choice
  back in one line with its zone and the server's next slot or refusal, and holds Mode, Workspaces,
  where each run goes, goal and ceiling under Advanced. The detail shows the cadence and zone, the next
  run or why it is paused, the prompt folded to four lines, the actions, and the recent runs with their
  state in a quiet tone and an Open that goes to that run's conversation.
- **Conversations.** A conversation a Schedule wrote carries a small clock before its title, and the
  Schedule's name as its tooltip and its accessible name.

## Settings

Settings is one native modal, `<dialog id="settings-dialog">`, owned by `dl-settings-dialog`: a navigation
column beside the one page it opens. The dialog owns what every page shares (opening and closing, focus,
which page shows, and the title, description and notices around it); each page is an element of its own
that owns its data and reports a short summary for its navigation row.

- **Geometry.** Desktop is `min(880px, 100vw - 2 * --space-layout)` wide and
  `min(640px, 100dvh - 2 * --space-layout)` high. The height is fixed, so no page resizes the dialog.
  It takes `--radius-dialog`, a hairline `--color-border-subtle` border, `--shadow-overlay`,
  `--color-bg-surface` and the shared scrim. The navigation column is 15rem (240px) wide, wide enough
  for "Conversation Sessions" and a count on one line, and a hairline sets it off from the page.
  Docked surfaces inside the dialog (the navigation, the pages, their cards) cast no shadow: the dialog
  is the one overlapping surface, and the notice is the one thing that floats over it, as a toast.
- **Navigation.** `<nav aria-label="Settings">` holds three labelled groups: Agent (Connections, Agent
  Accounts, Profile Memory, Skills), Data (Conversation Sessions, Privacy), and General (Language). Each
  item is a native `.dl-nav-item` button with an icon, its label, and a short status (`1/2`, `3`, `5`,
  `18`; none for Language, for Privacy, or for Memory while it is off). A page with no figure to show
  still says its state in full ("On", "Off") on a phone's list and to assistive technology. The page that
  is showing has `aria-current="page"` and the `--color-selected-row` tint; a dialog that is closed marks
  none. ArrowUp, ArrowDown, Home and End move focus among the items through `rovingFocusKeydown`, and
  Enter or Space opens one. The page pane is a region named by its heading, and a page that is not
  showing is `hidden`, not unmounted.
- **Pages.** A page is a column of cards: `--radius-card`, a 1px `--color-border-subtle` border, rows
  separated by the same hairline. A switch card is its switch's label, so a tap anywhere on it turns the
  switch. A page that holds an owner's own switch draws it only once the setting has been read, because
  an unchecked switch would claim "off" for a state nobody read. When that first read fails, Agent
  Accounts, Skills and Privacy show a note with Retry, and Profile Memory a one-line status without one.
  Privacy holds one such switch and, under it, a note on the tint Profile Memory's note wears: three
  statements, each named by a small bold muted term, saying what a contribution may include, what it
  never includes, and how it works. A destructive action is a `.dl-btn.dl-btn-danger-text` button in its
  own card. A notice is the app's toast (`.toast`, with its shadow and its Undo) in regions the dialog
  owns, one for each tone (see Toasts), because the shell's would sit under the scrim. It sits outside the page pane, so a Memory
  change shows wherever the reader is, including a phone's section list. Agent Accounts is a table of
  websites where its page is at least 36rem wide and three-line rows (the website, how it signs in, when
  it last did) where it is narrower; the page measures its own width, so a narrow window and a phone get
  the same rows. Profile Memory draws each stored memory as its own words first, the width the row has
  to give: the kind it is stands at the tail as a small label before Forget, so every memory begins at
  the same edge. A phone, which has no column beside the words to spare, puts the kind on the line under
  them and keeps Forget at the row's end across both. The words keep their own line breaks. Profile Memory
  is the one page that fills the pane: its switch card holds the top and its Clear card the foot, and the
  stored memories scroll in the card between them, so the pane beside the description never scrolls and the
  list's title, its status, Load more and Clear keep their places. That scrolling box is a named, focusable
  region, so a keyboard reaches it, and a page that opens again opens on the newest memories.
- **Text.** Small text is `--color-text-muted` or stronger: `--color-text-subtle` reads at 3.65 to 1 on the
  dark surface and 3.82 to 1 on the light one, below the 4.5 to 1 that small text needs, so group labels,
  statuses and column headings keep their rank through size and weight instead. A browser check measures
  every word of every page against what is painted under it, in both themes.
- **Controls.** Icon buttons are `dl-icon-button`. Inside the dialog the control ladder sets
  `--control-hit-target: var(--size-button)`, so they are compact beside a pointer and 44px under 1200px.
  Switches are `dl-switch--dense`: the compact track on desktop and the regular 40 by 24 on a phone,
  with a hit area of the same ladder size either way.
- **Phone.** At `(width <= 720px), (height <= 480px)` the dialog fills the screen with square corners and
  becomes two levels: a section list (each row shows its icon, name, a status line such as "MCP · 1 of 2
  enabled", and a disclosure), then one page whose header holds Back, the page title and Close. The list
  is the first level unless a page is named, and Back returns focus to the row of the page it left. Every
  control a finger meets is at least 44px.
- **Focus.** Opening focuses the current navigation item (the first list row on a phone). Closing returns
  focus to the control that opened Settings. Escape closes, as for any native dialog.

## Rich Content

### Pygments

The `/static/pygments.css` stylesheet is built at runtime from the installed Pygments and
contains two root-scoped palettes:

- Pygments `xcode` for `data-color-mode="light"`;
- Pygments `github-dark` for `data-color-mode="dark"`.

Generated selectors are scoped to the effective color mode. Pygments-owned container backgrounds are removed so code blocks continue to use DlightRAG surface tokens. Two low-contrast upstream foregrounds are replaced with fixed accessible values. Building it from the Pygments that renders the markup keeps the class names and their rules on one version, and the response revalidates by ETag.

This prevents light-syntax colors from being displayed on a dark code
background before or after an appearance change.

### MathJax

MathJax output explicitly inherits `currentColor`. Theme changes do not trigger re-typesetting and do not disturb frozen streaming blocks.

### Mermaid

A Mermaid diagram is an isolated `<img>` of SVG that Mermaid draws with its own colors, so it cannot inherit the page's `currentColor` the way MathJax does. `ui/mermaid.ts` draws each diagram in the effective color mode (`data-color-mode` on `<html>`): Mermaid's `dark` theme in the dark mode and its `default` theme in the light one. It watches that attribute, and when it changes it initializes Mermaid with the other theme and draws the diagrams already on the page again, in place, from the source each one kept. The cache of drawn SVG is keyed by mode and source, so switching back reuses the earlier drawing. The diagrams follow the mode whether the change was an explicit choice, the operating system's, or another tab's. What a reader who cannot see a diagram is told (its own `accTitle`, else the kind of diagram it is) follows the page language, and a language change tells it again for the diagrams already on the page. If Mermaid's chunk fails to load, the readable source stays and the next diagram tries the load again, as does the next redraw of an Answer that still holds a diagram as its source.

### Images and Overlays

Lightbox scrims remain dark in both modes because their purpose is image isolation. Caption, border, shadow, and panel colors use semantic tokens. Uploaded images and source page images are not recolored.

### Links

An answer's links, in chat and in a Markdown Artifact, keep the colour of the
text they sit in and carry a 1px underline in the Primary accent at rest, 2px
under the pointer. No colour reads 4.5:1 on the page and 3:1 from the Body text
around it at once: a colour 3:1 from the Body text reads at most 4.0:1 on the
dark page and 3.28:1 on the light one. So colour alone cannot mark a link, and
an underline shown only on hover still fails WCAG 1.4.1 (F73). In its text's
colour a link reads at that text's contrast, AAA for Body text wherever an
answer places it (8.53:1 at worst, on a tinted table row in the Canvas) and AA
in a quote, which is Muted. The underline holds 3:1 on every surface (3.72:1 on
a Mineral Light table header).

## Failure Handling

- Missing storage value resolves to `System`.
- Invalid storage value is ignored, removed when possible, and resolves to `System`.
- Storage read/write failures do not prevent in-page switching.
- Missing `matchMedia` or bootstrap failure preserves the dark fallback.
- System listeners are detached when an explicit preference is selected.
- Runtime system changes update only the effective mode, not the stored preference.
- Lit application rerenders cannot reset the theme because the state lives on `<html>` outside `<dl-app>`.

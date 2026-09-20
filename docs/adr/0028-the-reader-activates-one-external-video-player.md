# The reader activates one external video player

A video link may play where the reader encounters it, without becoming a Published
Artifact or granting arbitrary Answer HTML execution permission. Use free,
self-hosted embed resolution and official provider players, not a paid aggregator
or a second frontend framework. This supersedes only ADR 0026's prohibition on
external playback; its publication, card acquisition and sanitization contracts
remain in force.

## Status

Accepted and implemented after the user confirmed the product boundaries and
authorized implementation. Standards and Spec reviews passed. Local `make ci`
(4829 unit tests, 5 skipped), integration (554 passed, 1 skipped) and browser e2e
(92 passed) passed on 2026-09-20. These are local results, not a remote-CI claim.

Read-only real oEmbed requests also resolved official descriptors for Dailymotion,
TED and Wistia through the same generic path. The wheel includes the pinned data
and its license. Neither descriptor resolution nor deterministic blocked-frame
tests prove media playback: deployment-origin and Chrome/Safari/iOS playback
qualification remains outstanding. No live cross-platform playback acceptance
is claimed.

## Decision

- An actual sanitized prose anchor can offer playback independently of OG metadata.
  Code, image alt text, citation controls and source excerpts do not authorize a
  player. Published Markdown's numeric citation links retain that occurrence role
  when both their source id and normalized destination match an admitted Source;
  a recommendation to the same URL remains eligible. A plain-link fallback remains
  available, including when metadata is absent.
- Recognition is local and does not make an outbound request. A reader's explicit
  Play action resolves the selected public URL and creates the official iframe.
  History reads neither refetch metadata nor rewrite stored Answers. This also
  permits existing eligible links to gain the new affordance without re-settlement.
- One external playback owns the current browser document at a time, including
  pending resolution: selecting B aborts A's pending read and removes A's iframe.
  A returns to its preview. Closing or removing a player tears it down. There is no
  background playback or saved playback position in this slice. Native video
  Artifacts keep their existing independent controls.
- Playback is a browser presentation capability, not a Run, Resource, Evidence,
  Artifact publication or video ingestion operation. It does not use model/tool
  authority or send conversation contents to a provider.
- A bundled, pinned MIT-licensed oembed.com provider registry supplies generic
  URL matching and fixed publisher endpoints, without a hosted aggregator or a
  live registry fetch. YouTube, Vimeo and Bilibili are qualification examples,
  **not the support boundary**. Other registered publishers use the same parser
  and transport, without new per-site player code. Only fixed anonymous HTTPS,
  JSON-capable endpoints are used; XML-only/discovery-only/dynamic endpoints are
  not guessed. Acquisition uses the existing bounded, anonymous public HTTP path.
- A registry match is a playback *candidate*, not proof of video: the remote type
  is deliberately unknown until activation. The affordance says “Try playback”.
  Non-video responses, SDK/script-only players and ambiguous multiple iframes
  retain the original link instead of introducing another kind of embed.
- The registry's publisher and endpoint domain families bound permissible frame
  destinations (the recorded host with a leading `www.` removed, and subdomains).
  Small official mappings and reviewed `additional_player_domains` entries add
  explicit player hosts. The initial data exception is Wistia's `fast.wistia.net`,
  confirmed by its official HTTPS oEmbed response (`fast.wistia.com/oembed.json`,
  publisher example `home.wistia.com/medias/xfepf8u5c4`). Provider HTML cannot
  dynamically grant an unrelated domain. These permissions are projected with the selected
  link and checked again by the browser; resolver responses cannot widen them.
  Same-host application frames are refused. A legitimate player on a separate
  unrecorded CDN remains unsupported until its permission is reviewed as data.
  Browser player subresources/redirects remain provider/browser behavior, not
  backend DNS-pinned acquisition or an absolute browser-egress guarantee.
- Embed HTML is data to parse, never markup to insert into the main document.
  A standard `type: video` response containing one permitted HTTPS iframe yields
  a typed player descriptor; attributes/scripts in its HTML are never copied.
  Small known official address mappings can still resolve when metadata is
  unavailable, preserving their video identity and offsets. Generic providers
  need a successful valid oEmbed response; metadata success is not a guarantee
  that the individual video permits playback.
- The browser creates a typed, separate iframe with only the required playback
  permissions. Generated HTML Artifacts retain their existing sandbox/CSP. No
  provider SDK or player is loaded before activation; existing external covers
  remain external requests, so this is not a zero-third-party-network claim.
- Official playback may expose the reader's IP/browser information and platform
  cookies subject to browser rules. The user accepted this trade-off. Use only
  origin-level cross-origin referrer information; never deliberately supply a
  conversation path, question, answer or DlightRAG credentials. Platform ads,
  region/login restrictions and denied embedding remain the platform's behavior.

## Failure and scope

Resolver failure returns to the preview with an accessible error and retry/open
link. Provider refusal inside a cross-origin frame is not reliably observable
without a provider SDK: the platform can show its own error. An always-available
Stop control restores the preview, and the original external link remains visible.
An iframe load event is not treated as evidence that video playback succeeded.

The free route does not promise arbitrary website compatibility or no maintenance.
Standard metadata handling is shared; the pinned registry supplies ordinary
provider records, and small URL mappings carry exceptional differences. Registry
data lives at `src/dlightrag/adapters/http/browser/oembed-providers.json`, generated
from `iamcal/oembed@f87a746d7701396b86a7b5f4182ea412c870cb58` with the upstream
`build.js` field projection (383 records); the adjacent LICENSE preserves its MIT
notice. `additional_player_domains` is the local, reviewed extension to that
upstream data, not an upstream claim. Updating the snapshot or these permissions
is a reviewed code/data change, not a runtime permission expansion. Do not introduce an automatic SaaS fallback, generic
plugin architecture, transcription, stream extraction or third-party SDK in the
main application merely to make a watch-page URL play.

## Acceptance seams

Tests exercise the browser presentation projection, the authenticated playback
request, and the rendered Answer interaction—the same user-visible seams agreed
during qualification—not private methods. Required evidence includes:

1. A metadata-free eligible link offers Play, while code/citations do not; history
   and Markdown Artifact projection preserve the same rule without network reads.
2. No resolver request or player exists before click. A click loads only the
   approved player, keeps the external link, and does not loosen sanitizers.
3. B replaces A across separate Answers/Canvas, including out-of-order resolver
   completion. Removal and navigation stop playback. Unrelated Lit updates retain
   the active iframe and its position.
4. Invalid/private/credential-bearing URLs and malicious oEmbed HTML/destinations
   cannot introduce a player or bypass public-network admission. Acquisition has
   admission-inclusive time and byte limits.
5. Publishers outside the three examples resolve through registry data and the
   same generic video-response parser. Non-video/script-only responses and
   unrelated frame domains are refused; hostile URL paths cannot spoof a scheme's
   hostname, and an unknown publisher cannot request an arbitrary endpoint.
6. Deterministic browser/HTTP tests pass, followed by real provider qualification
   in the deployment origin. Actual Chrome/Safari/iOS playback is reported only
   where measured, separately from local and remote CI results.

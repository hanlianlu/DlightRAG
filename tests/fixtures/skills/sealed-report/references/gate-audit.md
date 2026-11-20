# Audit gate

Read this card whole before delivery. The rule table below is what the verifier checks, and the
instant-fail list binds from the first step, so scan it before starting.

## Rule table

A sealed document carries its own audit, computed in the run that wrote it. A document whose audit
was copied from another run is refused, because an audit nobody ran is a claim, not a check.

## Instant-fail list

A hand-written document is never sealed. A document edited after its seal is never sealed. A
verifier that was not run is never reported as having passed. Each of these ends the run with an
unsealed memo and the rule that failed.

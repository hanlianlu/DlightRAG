# Gotchas

- The seal covers the bytes of the sealed artifact; renaming or re-zipping it does not carry the seal.
- Each run binds its own workspace root, so run `pwd` and use workspace-relative paths.
- A refusal names the rule; fix that input and run the verifier again.

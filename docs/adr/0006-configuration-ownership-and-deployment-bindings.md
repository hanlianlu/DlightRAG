# Configuration ownership and deployment bindings

DlightRAG gives every effective setting one deployment-time owner. Non-secret product behavior and integration choices belong in `config.yaml`; credentials belong in `.env` for local operation or a secret store in an orchestrator; container topology belongs in Docker Compose or the equivalent deployment manifest. Nested `DLIGHTRAG_*` overrides remain available for secrets, service discovery, process-role bindings, and exceptional operator overrides, but deployment descriptors must not restate application defaults or policy already present in YAML.

## Status

Accepted and implemented for the bundled Docker Compose stack.

## Consequences

A deployment may use the same typed setting through YAML or through a topology override, but must not configure it in both places. Container mounts should align with canonical application paths whenever possible rather than adding path overrides. A Kubernetes deployment should mount its complete non-secret `config.yaml` from a ConfigMap, inject credentials from Secrets, and reserve manifest environment variables for values created by that deployment topology, such as Service DNS and listener or process-role bindings.

# Metadata field-stats residual risks and simplification review

**Repository:** DlightRAG at `92b9d4fa`
**Scope:** the PostgreSQL metadata field-schema aggregation added in Phase C
**Decision:** retain as optional P2 durability hardening, not a current defect or simplification prerequisite; do not refactor the trigger, promotion cutover, or rebuild.

## Executive conclusion

The aggregate-table design is sound. The planner read is bounded, ordinary writes maintain counts in the same transaction, reset paths are trigger-covered, and promotion repairs the one physical-row-move path atomically.

There is one optional residual hardening: startup currently verifies the stats table but not out-of-band damage to the trigger that makes the table correct. The smallest deep-module design, if that operational threat warrants code, is a metadata-local catalog verifier called after both reader verification and writer migration application. It would validate the parent trigger, its bound function and firing shape, and the clones on every attached partition. The final whole-repository review deliberately did not mix this hardening into the simplification implementation.

The other apparent simplifications are false economies:

- suppressing the trigger during promotion adds a privileged or hidden bypass to save work on a rare path;
- merging increment and decrement into a signed `INSERT ... ON CONFLICT` conflicts with the non-negative count constraint;
- a `MERGE`, statement-level transition-table trigger, data-modifying CTE, or generic reconciliation subsystem adds more interface than it removes;
- combining the trigger and backfill SQL builders would create a parameterized shallow helper across two genuinely different SQL shapes.

## 1. Real residual risk: trigger integrity is not verified

### Evidence

`verify_migrations` and `_absent_table_objects` inspect relations, columns, indexes, constraints, keys and partitions, but not `pg_trigger` or `pg_proc` (`src/dlightrag/adapters/postgres/core/_migrations.py:188-303`). The metadata requirements therefore verify the aggregate table's columns, primary key and check constraint, but not `dlightrag_metadata_field_stats` or `dlightrag_sync_metadata_field_stats()` (`src/dlightrag/adapters/postgres/corpus/pg_metadata_index.py:419-441`).

The trigger and function are installed only by the already-ledgered `metadata_field_stats` migration (`pg_metadata_index.py:403-416`). If the trigger is dropped or disabled out of band while the ledger remains intact:

- a reader passes the present verification;
- a writer skips the applied migration and also does not verify its objects;
- subsequent metadata writes succeed while the aggregate silently becomes stale.

A restart would not repair or report that state. Verification remains startup evidence, not continuous drift monitoring; a trigger changed after startup is intentionally outside this scope.

### Optional hardening design

Keep this invariant local to `pg_metadata_index.py`; there is only one application trigger in the repository, so adding a generic trigger-requirement hierarchy to `TableRequirement` would widen a shared interface for one use.

Add one private catalog query and verifier which proves:

1. the original trigger exists on `dlightrag_doc_metadata`;
2. it is enabled in the expected origin mode (`tgenabled = 'O'`);
3. it is an `AFTER ROW INSERT OR UPDATE OR DELETE` trigger (`tgtype = 29`);
4. it is bound to a zero-argument function named `dlightrag_sync_metadata_field_stats` returning `trigger`;
5. every currently attached metadata partition has the corresponding clone, linked through `tgparentid`, enabled with the same firing shape.

Call that verifier:

- after `verify_migrations(...)` in the reader branch; and
- after `apply_migrations(...)` in the writer branch of `PGMetadataIndex.initialize`.

One joined catalog query can prove the trigger/function binding; a separate function-existence query is unnecessary because the trigger stores the bound function OID. Do not hash `prosrc` or `pg_get_triggerdef`: that would turn legitimate migration edits into brittle startup hashes and exceed the repository's existing name/shape verification policy.

PostgreSQL documents `tgfoid`, `tgtype`, `tgenabled` and `tgparentid` in [`pg_trigger`](https://www.postgresql.org/docs/18/catalog-pg-trigger.html). The bit constants are defined by PostgreSQL's [`pg_trigger.h`](https://doxygen.postgresql.org/pg__trigger_8h_source.html): ROW=1, BEFORE=2, INSERT=4, DELETE=8, UPDATE=16, TRUNCATE=32 and INSTEAD=64, so this trigger's exact mask is `1 + 4 + 8 + 16 = 29`. PostgreSQL also guarantees that creating a row trigger on a partitioned parent clones it to existing and later attached partitions ([`CREATE TRIGGER`](https://www.postgresql.org/docs/18/sql-createtrigger.html)).

### Tests

If implemented, use real PostgreSQL tests to damage the schema and prove both startup roles fail loudly:

- drop the parent trigger;
- disable the parent trigger;
- optionally disable one attached child's clone to prove the all-partitions predicate.

The current reader damage matrix in `tests/integration/test_reader_role_pg.py:317-510` already establishes the expected fail-loud style. A writer-path assertion is also necessary; otherwise the common writer-only deployment would retain the silent hole.

### Complexity

One private SQL predicate, one small verifier, two calls, and focused integration cases. Runtime hot paths are unchanged. No configuration, telemetry, repair endpoint or background job is introduced.

## 2. Promotion does redundant work, but removing it is not a simplification

### Current behavior

During atomic cutover, promotion:

1. deletes the workspace's rows from each shared DEFAULT child;
2. the metadata child's cloned row trigger decrements aggregate counts once per deleted row;
3. attaches the pre-copied staging tables as dedicated partitions;
4. deletes and backfills that workspace's aggregate rows inside the same transaction.

See `src/dlightrag/adapters/postgres/corpus/promotion_worker.py:437-476` and `rebuild_metadata_field_stats_for_workspace` at `pg_metadata_index.py:724-731`.

For `D` documents with `F` present fields, the metadata delete invokes the trigger `D` times and performs work proportional to `D × F`; the recount then scans the same logical field presence again. This was established from the SQL shape, not a production benchmark. Promotion is threshold-gated and normally happens once per workspace, so this cost is outside the planner and ordinary-write hot paths.

### Alternatives rejected

- **`session_replication_role = replica`:** suppresses ordinary triggers for the whole session and also affects foreign-key triggers. PostgreSQL permits it only to a superuser or a role granted the relevant `SET` privilege. It is too broad and not portable to a normal managed-PostgreSQL application role ([runtime configuration](https://www.postgresql.org/docs/18/runtime-config-client.html#GUC-SESSION-REPLICATION-ROLE)).
- **Disable the cloned trigger with `ALTER TABLE`:** PostgreSQL takes `SHARE ROW EXCLUSIVE`, which conflicts with ordinary `ROW EXCLUSIVE` DML. On the shared DEFAULT child this would block other workspaces until cutover commit, contrary to the promotion lock design ([`ALTER TABLE`](https://www.postgresql.org/docs/18/sql-altertable.html#SQL-ALTERTABLE-DESC-DISABLE-ENABLE-TRIGGER)).
- **A transaction-local custom GUC checked by the trigger:** technically possible, but it creates a hidden bypass connecting promotion to trigger internals. It removes the recount's incidental repair behavior and makes future writes in that transaction depend on an ambient flag. This is a shallower interface than the current exact trigger plus explicit rebuild.
- **Trust decrements and omit the rebuild:** incorrect, because inserts into detached staging tables did not fire the parent trigger.
- **Move the rebuild outside the cutover transaction:** exposes an inconsistent planner schema and breaks rollback/crash atomicity.

### Decision

Deliberately retain the decrement-then-rebuild choreography. The redundancy is earned by a simple invariant: every ordinary row mutation fires the trigger, while the one physical relocation path explicitly reconciles after attachment. Both effects commit or roll back together.

The existing promotion integration test already proves aggregate equality before and after cutover and proves the cloned trigger handles post-promotion insert/delete (`tests/integration/test_promotion_worker_pg.py:540-661`). The only useful addition is to seed a custom metadata key so the promotion test covers the backfill's `jsonb_object_keys` leg, not only built-in fields.

## 3. Trigger concurrency: no production-interface fix is justified

The trigger orders keys by `(workspace, field_id)` within increment and decrement statements (`pg_metadata_index.py:193-218`). A theoretical deadlock exists if two raw SQL updates swap field presence in opposite directions: one locks field B while adding it and then wants A while removing it; the other does the inverse.

That operation is not expressible through `PGMetadataIndex`:

- built-in assignments preserve the old value with `COALESCE`;
- custom metadata uses JSONB `||`, which adds or overwrites keys but does not remove them;
- removal is exposed only as document `DELETE`/workspace `clear`.

See `_field_assignment`, `_UPSERT` and `_UPDATE` at `pg_metadata_index.py:448-513`. Supported updates therefore add presence only; their `OLD EXCEPT NEW` decrement set is empty. Inserts, additive updates and deletes each lock aggregate keys in the same ascending order.

A single signed-delta `INSERT ... ON CONFLICT` is not a valid simplification. A decrement candidate of `-1` violates `CHECK (document_count >= 0)` before it can safely represent a missing-row branch. This was confirmed against local PostgreSQL 18 with a temporary table: an existing conflicting key still raised the check violation for the `-1` candidate before the conflict update.

A conditional `MERGE`, advisory locks, or statement-level transition triggers could encode more cases, but all add a new locking or trigger interface to protect unsupported raw DML. Leave the current two-pass implementation. Record the additive-update invariant near the trigger or `_field_assignment` so a future replace/remove-field interface must revisit lock ordering.

## 4. Bulk/reset path inventory

Current production mutation paths are covered:

| Path | Mechanism | Aggregate behavior |
|---|---|---|
| `PGMetadataIndex.upsert` | row INSERT/UPDATE | trigger-maintained |
| `merge_custom_metadata` | additive UPDATE | trigger-maintained |
| `delete` / `clear` | row DELETE | trigger-maintained |
| promotion DEFAULT delete | row DELETE plus attach | trigger, then same-transaction rebuild |
| orphan cleanup | dynamically issued DELETE | trigger-maintained; direct stats cleanup is also in the orphan scan |
| normal workspace reset | `metadata_index.clear()` plus orphan cleanup | trigger-maintained |
| development reset | drops/recreates the complete `public` schema and ledger | migrations recreate/backfill both objects |

No runtime path truncates `dlightrag_doc_metadata`. PostgreSQL explicitly states that `TRUNCATE` does not fire `ON DELETE` triggers ([`TRUNCATE`](https://www.postgresql.org/docs/18/sql-truncate.html)); any future metadata TRUNCATE or direct partition-drop path must therefore own aggregate deletion/rebuild in the same transaction.

A second maintenance invariant concerns registry evolution: if a future filterable built-in field is added, applying only its dynamic column migration will not rerun the already-ledgered trigger/backfill migration. That revision must append a new migration which refreshes the trigger function and establishes aggregate counts for the new field. A concise comment in `_build_schema_migrations` is enough today; generating migration versions from registry state would make historical ledgers unstable.

No repair endpoint, scheduled reconciliation or drift telemetry is justified. Every supported mutation path is already closed at its source; startup object verification handles the remaining structural failure.

## 5. No further code-shape simplification

- Keep `_presence_rows` and `_field_stats_source` separate. Both derive columns from `_FILTERABLE_COLUMNS`, but one operates on `OLD`/`NEW` records and the other on a table relation with `doc_id` and a workspace predicate. A shared configurable builder would expose their differences as parameters rather than remove them.
- Keep the two-statement workspace rebuild. A data-modifying CTE is harder to reason about and saves one round trip inside a rare, scan-heavy promotion transaction.
- Keep the ledgered migration text unchanged. Replacing DROP+CREATE with `CREATE OR REPLACE TRIGGER` only changes fresh installations; existing databases never rerun that migration.
- Keep the row-level trigger. Statement-level transition triggers would require separate event triggers and more partition-specific behavior, enlarging the interface while weakening the proven path.
- Keep the aggregate table and 128-key bounded planner query unchanged.

By the deletion test, the current trigger and rebuild are deep: deleting either redistributes exactness and promotion knowledge into every writer. The proposed startup verifier is also deep despite being small: deleting it reintroduces two startup branches that cannot prove the same correctness object. The rejected abstractions merely move existing SQL differences behind wider parameters.

## Ranked recommendation

1. **Optional P2:** if out-of-band schema damage is in scope, add metadata-local catalog verification of the parent trigger, its bound function/firing shape, and all attached clones; invoke it after reader verification and writer migration application.
2. **Tests only with that hardening:** trigger dropped/disabled for both startup roles; one custom key through promotion.
3. **Document only:** additive-update lock-order assumption; future TRUNCATE/partition-drop parity; new filterable fields require a follow-up trigger/backfill migration.
4. **Deliberately leave:** promotion's rare redundant trigger work, two-pass trigger SQL, two-statement rebuild, separate SQL builders, and absence of background reconciliation/configuration.

The net result is smaller than the apparent risk list: there is no known trigger-function defect. One optional structural verification query would cover out-of-band damage; all other proposed refactors would increase interface surface or reduce robustness.

## Research checks performed

- Inventoried every repository reference and production DML path for `dlightrag_doc_metadata` with `rg`.
- Read the complete trigger/migration/rebuild and promotion cutover sources plus reset/orphan-cleanup paths.
- Cross-checked trigger cloning, catalog fields, trigger enable modes, lock level and TRUNCATE semantics against PostgreSQL 18 documentation.
- Executed a local PostgreSQL 18 transaction proving that a signed `-1` insert candidate violates the non-negative CHECK before the proposed conflict-update design can be used.

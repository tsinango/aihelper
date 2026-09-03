# Oracle migration notes

- The source D1 schema has no product, product-attribute, feature, or verified-fact rows. The Oracle counts are preserved as zero.
- `/api/v1/documents` upload returns `501` during this copy-only migration. The three existing ready PDFs are copied locally; the original OCR/ingestion pipeline is not silently replaced.
- Availability remains explicitly unavailable with `inventory_adapter_not_migrated` until a live inventory adapter is configured; it is never inferred from historical chat.
- Migration `003_mvp_rag_memory.sql` adds traceable case memory, scoped aliases, searchable learning examples, and embeddings. Migration `009_openrouter_vector_dimensions.sql` converts live vectors to 2048-dimensional Nemotron embeddings. Migration `010_review_provenance_async_embedding.sql` adds message relations, reusable knowledge evidence, and asynchronous embedding status. Historical Telegram answers remain recall/reviewer context only; they cannot directly answer customers. Publication still requires human review.
- The existing Oracle Caddy serves `notes.813328.xyz` on 80/443. Nginx therefore listens on `127.0.0.1:8080`; Caddy's new unmatched `:80` block forwards the migration app to Nginx without changing the notes route. A known production hostname is still needed before changing DNS or origin TLS.
- OpenRouter is the sole model provider. The service is fail-closed/degraded when the protected OpenRouter token is unavailable; transient errors receive finite retries and persistent failures never switch providers or models.

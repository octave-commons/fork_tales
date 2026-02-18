(契
  (id "契:ollama-embeddings-cost:127.0.0.1:11434:v1")
  (ver "1.0.0")
  (title "Ollama Embeddings Request Cost Contract")
  (parties
    (ctx 己 (entity "eta-mu-world.embedding-client"))
    (ctx 汝 (entity "ollama@127.0.0.1:11434"))
    (ctx 彼 (entity "receipt-ledger"))
    (ctx 世
      (resource "http:POST:/api/embeddings")
      (resource "http:POST:/api/embed")
      (resource "model:nomic-embed-text")
      (resource "gpu0")
      (resource "ram")))
  (scope
    (governs :embedding-request-cost-and-admission)
    (request-type :single-input)
    (endpoint "/api/embeddings")
    (endpoint-fallback "/api/embed")
    (model "nomic-embed-text"))
  (permits
    (allow :embed
      (when (and
              (<= input.bytes 32768)
              (<= inflight 2)
              (<= rate.rpm 60)
              (= request.model "nomic-embed-text"))))
    (allow :retry-once
      (when (and
              (= retry.count 0)
              (= retry.backoff.ms 250)
              (in error.class [:timeout :conn-reset]))))
    (allow :cache-read)
    (allow :cache-write))
  (forbids
    (deny :embed (when (> input.bytes 32768)))
    (deny :embed (when (> inflight 2)))
    (deny :embed (when (> rate.rpm 60)))
    (deny :embed (when (not (= request.model "nomic-embed-text"))))
    (deny :retry (when (> retry.count 0)))
    (deny :action-unknown))
  (owes
    (must (set request.idempotency-key input.sha256))
    (must (compute cost.cu
      (base 10)
      (per-kib 1)
      (rounding :ceil)))
    (must (emit-receipt
      (contract-id "契:ollama-embeddings-cost:127.0.0.1:11434:v1")
      (kind "cost.ollama.embedding")
      (fields [:ts :kind :origin :owner :dod :refs])))
    (must (fail-closed true)))
  (cost
    (unit "CU")
    (budget
      (max-cu-per-request 42)
      (max-cu-per-minute 2048))
    (limits
      (max-inflight 2)
      (request-timeout-ms 8000)
      (max-queue-wait-ms 1000))
    (accounting
      (debit-rule "cu = 10 + ceil(input.bytes/1024)")
      (window "60s")))
  (verify
    (check :schema
      (requires [id ver title parties scope permits forbids owes cost verify fails]))
    (check :allowlist
      (rule "requested action is explicitly permitted and not forbidden"))
    (check :budget
      (rule "computed cu <= max-cu-per-request and minute budget remaining"))
    (check :upstream
      (rule "try /api/embeddings first, then /api/embed; require http.status == 200 and embedding.length > 0"))
    (check :receipt
      (rule "receipt appended with contract-id and cu")))
  (fails
    (on :schema-fail
      (result :deny)
      (code "E_CONTRACT_SCHEMA")
      (retry false))
    (on :allowlist-fail
      (result :deny)
      (code "E_NOT_PERMITTED")
      (retry false))
    (on :budget-fail
      (result :deny)
      (code "E_COST_LIMIT")
      (retry-after-ms 60000))
    (on :upstream-fail
      (result :cache-only)
      (code "E_UPSTREAM_EMBED")
      (retry false))
    (on :verify-missing
      (result :deny)
      (code "E_UNVERIFIABLE")
      (retry false)))
  (notes
    (defaults
      (missing-field :deny)
      (missing-verify :deny)
      (unknown-action :deny))
    (refs ["promethean.receipts/v2" "signal:契"])))

;; ENSO-1 — Sacred Protocol Encoding for Promethean Shards
;; Purpose: single strict machine-checkable S-expression spec shards can implement.

(protocol ENSO-1

  (principles
    "deterministic envelopes"
    "small composable messages"
    "capability + privacy negotiation"
    "explicit flow-control + retry semantics"
    "single audit trail across tools + MCP + assets")

  ;; ============================================================
  ;; 1) CORE TYPES
  ;; ============================================================
  (types
    (type UUID :string)
    (type ISO8601 :string)

    (type Envelope
      (fields
        (id      UUID)
        (ts      ISO8601)
        (room    :string)
        (from    :string)
        (kind    :enum (event stream))
        (type    :string)
        (seq     :int?)
        (rel     (object
                   (replyTo UUID?)
                   (parents (list UUID)? )))
        (payload :any)
        (sig     :string?))
      (canonical-json
        "signature computed over canonical json of envelope without `sig`"))

    (type StreamFrame
      (fields
        (streamId UUID)
        (codec    :enum ("opus/48000/2" "pcm16le/16000/1" "text/utf8" "jsonl"))
        (seq      :int)
        (pts      :int)
        (eof      :bool?)
        (data     :bytes-or-string)))

    (type PrivacyProfile :enum (persistent pseudonymous ephemeral ghost))

    (type PrivacyRequest
      (fields
        (profile PrivacyProfile)
        (wantsE2E :bool?)
        (allowLogging :bool?)
        (allowTelemetry :bool?)))

    (type RetentionPolicy
      (fields
        (messages (object (defaultTTL :int) (maxTTL :int)))
        (assets   (object (defaultTTL :int) (maxTTL :int) (allowDerivations :bool)))
        (logs     (object (keepProtocolLogs :bool) (logTTL :int)))
        (roster   (object (keepPresenceHistory :bool)))
        (index    (object (allowSearch :bool) (indexTTL :int)))))

    (type HelloCaps
      (fields
        (proto :literal "ENSO-1")
        (agent (object (name :string) (version :string))?)
        (caps  (list :string))
        (privacy PrivacyRequest?)
        (cache  :any?)))

    (type ToolAdvertisement
      (fields
        (provider :enum (native mcp))
        (serverId :string?)
        (tools (list (object (name :string) (schema :any?))))
        (resources (list (object (name :string) (uri :string) (title :string?)))?)))

    (type ToolCall
      (fields
        (callId UUID)
        (provider :enum (native mcp))
        (serverId :string?)
        (name :string)
        (args :any)
        (ttlMs :int?)))

    (type ToolResult
      (fields
        (callId UUID)
        (ok :bool)
        (result :any?)
        (error :string?)
        (resources (list :any)?)))

    (type AssetPut
      (fields
        (name :string?)
        (mime :string)
        (bytes :int?)
        (cid :string?)
        (room :string?)
        (policy (object (public :bool?) (ttlSeconds :int?))?)))

    (type DataSourceId
      (fields
        (kind :enum (fs api db http mcp enso-asset other))
        (location :string)))

    (type Discoverability :enum (invisible discoverable visible hidden))

    (type Availability
      (one-of
        (object (mode :literal private))
        (object (mode :literal public))
        (object (mode :literal shared) (members (list :string)))
        (object (mode :literal conditional) (conditions (list :any)))))

    (type ContentPermissions
      (fields
        (readable :bool?) (changeable :bool?) (movable :bool?) (exchangeable :bool?)
        (sendable :bool?) (addable :bool?) (removable :bool?) (deletable :bool?)
        (saveable :bool?) (viewable :bool?)))

    (type ContextState :enum (active inactive standby pinned ignored)))

  ;; ============================================================
  ;; 2) EVENT CATALOG
  ;; ============================================================
  (events
    (family chat (types chat.msg))
    (family content (types content.post content.message content.retract content.burn))
    (family presence (types presence.join presence.part))
    (family capabilities (types caps.update))
    (family state (types state.patch))
    (family tooling (types tool.advertise tool.call tool.partial tool.result))
    (family voice (types voice.meta))
    (family flow (types flow.nack flow.pause flow.resume))
    (family stream-control (types stream.resume))
    (family assets (types asset.put asset.chunk asset.commit asset.ready asset.derive asset.derived asset.delete))
    (family cache (types cache.put cache.hit cache.miss cache.evict cache.partial cache.policy))
    (family privacy (types room.policy privacy.accepted consent.record approval.request approval.grant))
    (family context (types datasource.add datasource.update context.create context.add context.apply context.diff))
    (family mcp (types mcp.mount mcp.announce)))

  ;; ============================================================
  ;; 3) HANDSHAKE
  ;; ============================================================
  (handshake
    (hello
      (send (Envelope (kind event) (type hello) (payload HelloCaps)))
      (require "proto == ENSO-1" "caps declared" "privacy request optional"))

    (gateway-ack
      (must-send
        (Envelope (kind event) (type privacy.accepted) (payload :any))
        (Envelope (kind event) (type room.policy) (payload RetentionPolicy))))

    (capabilities
      (rules
        "capabilities are authoritative only when acknowledged by caps.update"
        "caps.update includes full caps list + granted/revoked + revision monotonic")))

  ;; ============================================================
  ;; 4) STREAMS + FLOW CONTROL
  ;; ============================================================
  (streams
    (rule "all stream frames are envelopes kind=stream with StreamFrame payload")
    (voice
      (frame-type voice.frame)
      (meta-event voice.meta)
      (vad-recommended true)
      (transcripts
        (codec "text/utf8")
        (link-back "use rel.parents to reference voice stream")))

    (flow-control
      (sequence
        (rule "seq monotonic per stream")
        (nack (event flow.nack (payload (streamId UUID) (missing (list :int)))))
        (pause (event flow.pause (payload (streamId UUID))))
        (resume (event flow.resume (payload (streamId UUID)))))
      (backpressure "senders MUST honor flow.pause")
      (degradation (state.patch "gateway emits room-level degraded indicators"))))

  ;; ============================================================
  ;; 5) TOOLS + MORGANNNA
  ;; ============================================================
  (tools
    (advertise (event tool.advertise (payload ToolAdvertisement)))

    (invoke
      (event tool.call (payload ToolCall))
      (partial (event tool.partial (payload ToolResult)))
      (result  (event tool.result  (payload ToolResult)))
      (ttl
        (rule
          "tool.call MAY include ttlMs"
          "gateway MUST timeout + emit tool.result ok=false error=timeout")))

    (morganna
      (eval-mode
        (flag-path "room.flags.eval")
        (requires
          (event act.rationale "justify each tool.call")
          (event act.intent "enumerate allowed actions"))
        (violation
          "concealing tool intent in eval mode is a protocol violation"))))

  ;; ============================================================
  ;; 6) MCP
  ;; ============================================================
  (mcp
    (caps "mcp.client" "mcp.server:<id>")

    (mount
      (event mcp.mount
        (payload
          (object
            (serverId :string)
            (transport (object (kind :string) (url :string)))
            (exposeTools :bool)
            (exposeResources (list :string))
            (labels (map :string :string)))))
      (gateway-behavior
        "gateway establishes JSON-RPC"
        "runs tools/list and resources/list"
        "mirrors as tool.advertise"))

    (announce
      (event mcp.announce (payload :any))
      (gateway-behavior "mirror as tool.advertise"))

    (tool-call
      (rule "tool.call provider=mcp routes via MCP tools/call")
      (streaming "MCP partials MAY be mapped to stream envelopes codec=text/utf8")))

  ;; ============================================================
  ;; 7) ASSETS
  ;; ============================================================
  (assets
    (upload
      (event asset.put    (payload AssetPut))
      (stream asset.chunk (payload StreamFrame))
      (event asset.commit (payload (object (cid :string?))))
      (event asset.ready  (payload (object (cid :string) (uri :string) (mime :string) (bytes :int)))))

    (messages
      (event content.post    (payload :any))
      (event content.message (payload :any))
      (rule "gateway emits content.message immediately; appends derived parts later"))

    (derivations
      (event asset.derive  (payload :any))
      (event asset.derived (payload :any))
      (sandbox
        (default "network disabled")
        (policy "room policy governs auto-derivations + MIME allowlist"))))

  ;; ============================================================
  ;; 8) CACHE
  ;; ============================================================
  (cache
    (rule "content addressed by cid; cache events coordinate across sessions")
    (resume (event stream.resume (payload (object (cid :string) (fromSeq :int?)))))
    (policy (event cache.policy (payload :any))))

  ;; ============================================================
  ;; 9) PRIVACY + RETENTION
  ;; ============================================================
  (privacy
    (profiles persistent pseudonymous ephemeral ghost)
    (effects
      (ghost
        (requires-e2e true)
        (forbid "server-side derivations")
        (forbid "indexing/search"))
      (ephemeral
        (ttl-short true)
        (derivations "optional; policy-bound"))
      (pseudonymous (logging "minimal"))
      (persistent  (indexing "allowed by policy")))

    (message-controls
      (fields expiresAt burnOnRead forbidIndex watermark)
      (rules
        "expiresAt <= room.messages.maxTTL"
        "burnOnRead requires content.burn receipts"))

    (consent
      (event consent.record (payload :any))
      (rule "export/index/retention actions should be receipt-backed")))

  ;; ============================================================
  ;; 10) CONTEXT
  ;; ============================================================
  (context
    (datasource
      (meta DataSourceId Discoverability Availability)
      (events datasource.add datasource.update))

    (graph
      (states active inactive standby pinned ignored)
      (entry (fields (id DataSourceId) (state ContextState) (overrides :any?) (permissions ContentPermissions?))))

    (events
      (context.create "create context container")
      (context.add    "add entry")
      (context.apply  "apply context to room")
      (context.diff   "summarize applied context changes"))

    (llm-view
      (rule "gateway computes LLM view listing active/standby/ignored + grants + parts")
      (approvals
        (soft-conditions
          (event approval.request)
          (event approval.grant)
          (rule "work continues only after approval.grant")))))

  ;; ============================================================
  ;; 11) SECURITY
  ;; ============================================================
  (security
    (auth "opaque tokens or mutual TLS")
    (signatures (algo ed25519) (rule "room policy may mandate signatures for roles/types"))
    (audit
      (rule "tool.call metadata may be logged in persistent rooms")
      (ghost-mode "suppress detailed logs; retain aggregates only")))

  ;; ============================================================
  ;; 12) SHARD ENCODING
  ;; ============================================================
  (shards
    (shard :router
      (owns envelope-validate capability-handshake policy-enforce room-state-crdt tool-dispatch mcp-bridge receipts audit-log)
      (must-emit room.policy privacy.accepted caps.update state.patch flow.pause flow.resume flow.nack tool.result)
      (must-verify (sig-policy) (ttl-policy) (privacy-policy) (retention-policy)))

    (shard :senses
      (owns stream-ingest vad-segmentation stream-registry-client nack-retry backpressure-honor)
      (must-emit voice.meta voice.frame transcript.frames)
      (must-honor flow.pause flow.resume))

    (shard :voice
      (owns tts-synthesis voice.playback codec-transcode)
      (must-consume voice.frame)
      (must-produce voice.frame (derived "text/utf8" "jsonl")))

    (shard :memory
      (owns cache-store retention-enforce index-policy consent-ledger)
      (must-honor forbidIndex burnOnRead privacy.profile)
      (must-provide cache.events vector-manifest deletion-receipts))

    (shard :brain
      (owns tool-planning act.rationale act.intent context-aware-reasoning)
      (must-honor eval-mode approvals privacy.policy)
      (must-emit (when eval-mode act.rationale act.intent))))

  ;; ============================================================
  ;; 13) REQUEST CONTRACTS
  ;; ============================================================
  (request-contracts
    (request :enso.hello
      (requires HelloCaps)
      (response (privacy.accepted room.policy))
      (reject-if (proto-mismatch true) (caps-missing true)))

    (request :enso.tool.call
      (requires ToolCall)
      (requires (when eval-mode act.rationale act.intent))
      (response (tool.partial* tool.result))
      (reject-if (ttl-missing? false)))

    (request :enso.stream.send
      (requires StreamFrame)
      (requires seq-monotonic)
      (flow-control (flow.nack flow.pause flow.resume)))

    (request :enso.asset.put
      (requires AssetPut)
      (flow (asset.chunk* asset.commit asset.ready))
      (reject-if (mime-disallowed true)))

    (request :enso.context.apply
      (requires (ctxId :string))
      (response (context.diff llm.view))
      (reject-if (approval-required-and-missing true)))

    (request :enso.mcp.mount
      (requires (serverId transport exposeTools exposeResources labels))
      (response (tool.advertise))
      (reject-if (acl-deny true))))

  ;; ============================================================
  ;; 14) ENSO <-> Π INTEROP
  ;; ============================================================
  (interop
    (rule "ENSO envelopes/events are the live-room substrate; Π is the snapshot substrate")
    (bridge
      (router->Π "export room policy, caps revisions, tool receipts, and context diffs into Π")
      (Π->router "seed router with promptdb + capability schema + shard keys")))

  ;; ============================================================
  ;; 15) OPEN QUESTIONS
  ;; ============================================================
  (open-questions
    (q :signature-mandate "Which event types MUST be signed in canonical rooms?")
    (q :crdt-choice "Which CRDT library/format is canonical for room state + context graph?")
    (q :voice-plane "When (if ever) do we switch voice to WebRTC while preserving envelope IDs?")
    (q :ghost-room-implementation "What is the minimal audit surface allowed under ghost profile?")
    (q :mcp-scope-manifests "Where do per-server allowlists live (repo/db/room policy), and how are they versioned?")
    (q :cache-authority "Is cache authoritative for replay, or advisory for latency only?")
    (q :derivation-workers "What sandbox runner is canonical (firejail/nsjail/docker) and what is the policy DSL?")
    (q :tool-trace "What exactly is captured in tool traces under each privacy profile?")))

)

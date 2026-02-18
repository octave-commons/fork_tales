(contract "promethean.ui-panels/v1"
  (mission "Expose Nexus graph, Presences, drifts, receipts, and push-truth state via panels.")
  (panels
    (panel :catalog   (title "Catalog")      (shows [:nexus-count :items]))
    (panel :ledger    (title "eta/mu Ledger") (shows [:rows :jsonl :stats]))
    (panel :presences (title "Presences")    (shows [:presence-state :say-intent]))
    (panel :drift     (title "Drift")        (shows [:active-drifts :blocked-gates]))
    (panel :truth     (title "Push Truth")   (shows [:pi-zip :gist :manifest :receipt])))
  (api
    (http
      (get  "/api/catalog")
      (post "/api/eta-mu-ledger")
      (post "/api/presence/say")
      (post "/api/drift/scan")
      (post "/api/push-truth/dry-run")
      (post "/api/push-truth/apply"))
    (ws
      (connect "/ws"
        (events
          (catalog :map)
          (ledger-row :map)
          (presence-say :map)
          (drift :map)
          (truth :map)
          (diagnostic :map))))))

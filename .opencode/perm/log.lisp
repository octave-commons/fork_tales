(perm/log.v1
  (version "1.0.0")
  (principal "Err")
  (events
    (perm/event.v1
      (id perm.e.000001)
      (ts 1763239800)
      (principal "Err")
      (type grant)
      (cap telemetry.read)
      (intensity 0.8)
      (scope
        (domains-allow ("*"))
        (methods-allow ("GET")))
      (purpose ("health-observation" "runtime-verification"))
      (ttl (hours 8760))
      (audit (log-level decision+reason)))

    (perm/event.v1
      (id perm.e.000002)
      (ts 1763239801)
      (principal "Err")
      (type deny)
      (cap web.crawl)
      (intensity 1.0)
      (scope
        (domains-allow ("*"))
        (methods-allow ("GET"))
        (robots (respect true)))
      (purpose ("default-deny-external-io"))
      (ttl (hours 999999))
      (audit (log-level decision+reason)))))

(contract "promethean.fork-tax-git/v1"
  (mission
    "Pay fork tax continuously by converting working drift into frequent commits and frequent pushes.
     Keep evidence explicit through receipts and presence-shaped repair guidance.")

  (scope
    (git-root ".")
    (receipt-log "receipts.log")
    (commands
      (audit "node contracts/contract_fork_tax_git_v1.mjs --audit")
      (cycle "node contracts/contract_fork_tax_git_v1.mjs --cycle --message \"<msg>\" --owner <owner> --dod \"<dod>\"")
      (cycle-push "node contracts/contract_fork_tax_git_v1.mjs --cycle --push --message \"<msg>\" --owner <owner> --dod \"<dod>\"")))

  (cadence
    (max-dirty-minutes 120)
    (max-unpushed-commits 5)
    (max-untracked-items 40)
    (checkpoint-rule
      "If branch is dirty past max-dirty-minutes, checkpoint commit is mandatory.")
    (push-rule
      "If ahead exceeds max-unpushed-commits, push is mandatory unless remote/upstream is missing; then emit drift."))

  (presences
    (presence fork-tax-canticle
      (facts ["commit age" "ahead/behind" "branch state"])
      (asks ["are we paying fork tax now?"])
      (repairs ["checkpoint commit" "push or declare gate block"]))

    (presence file-sentinel
      (facts ["tracked/staged/untracked counts" "dirty paths"])
      (asks ["what must be staged now?"])
      (repairs ["git add -u" "triage untracked files"]))

    (presence witness-thread
      (facts ["remote origin" "upstream tracking" "behind count"])
      (asks ["is lineage publishable?"])
      (repairs ["set remote origin" "set upstream" "reconcile behind commits"]))

    (presence keeper-of-receipts
      (facts ["receipts.log present" "decision receipt appended"])
      (asks ["is checkpoint evidence written?"])
      (repairs ["append :decision receipt for each cycle"]))

    (presence anchor-registry
      (facts ["owner" "dod" "commit message"])
      (asks ["is accountability explicit?"])
      (repairs ["require --owner" "require --dod" "require checkpoint message"])))

  (drift
    (kind :stale-dirty-tree
      (when (and (git.dirty? true) (> (git.minutes-since-commit) (cadence.max-dirty-minutes))))
      (severity 8)
      (repair "run cycle and commit checkpoint now"))

    (kind :unpushed-backlog
      (when (> (git.ahead-count) (cadence.max-unpushed-commits)))
      (severity 8)
      (repair "push current branch or declare push gate block"))

    (kind :missing-upstream-lineage
      (when (or (git.remote-origin-empty?) (git.upstream-empty?)))
      (severity 9)
      (repair "configure remote/upstream before claiming push paid"))

    (kind :untracked-fog
      (when (> (git.untracked-count) (cadence.max-untracked-items)))
      (severity 5)
      (repair "triage untracked files into commit/ignore buckets")))

  (ritual
    (order
      ["audit"
       "stage tracked changes"
       "append decision receipt"
       "commit checkpoint"
       "push if gate passes"])
    (gate
      (commit
        (requires [:owner :dod :receipt-log]))
      (push
        (requires [:remote-origin :upstream]))))

  (output
    (audit
      (record "promethean.fork-tax-git/v1.audit")
      (includes [:git :presences :say-intent :gate]))
    (cycle
      (record "promethean.fork-tax-git/v1.cycle")
      (includes [:actions :before :after :push]))
    (say-intent
      (facts :vector)
      (asks :vector)
      (repairs :vector)
      (constraints (no-new-facts true) (cite-refs true) (max-lines 8))))

  (refs
    ["contracts/contract_fork_tax_git_v1.mjs"
     ".opencode/promptdb/contracts/receipts.v2.contract.lisp"
     "receipts.log"]))

(contract "promethean.receipts/v2"
  (file (path "receipts.log") (append-only? true))
  (line-format
    (delimiter " | ")
    (required-keys [ts kind origin owner dod pi host manifest refs])
    (optional-keys [note tests decisions drift]))
  (kinds
    [:push-truth :artifact-hash :test-run :build :decision :drift :catalog
     :observation :field-impact :truth :refutation :adjudication]))

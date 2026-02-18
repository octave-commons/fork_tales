(protocol eta.v1
  (record eta/observation.v1
    (required
      (id :symbol)
      (ts :int)
      (source :enum (user file ipc sensor system))
      (payload :string))
    (optional
      (confidence :float)
      (entropy :float)
      (notes :string)))

  (record eta/field-impact.v1
    (required
      (observation-id :symbol)
      (field :enum (f1 f2 f3 f4 f5 f6 f7 f8))
      (vector :map)
      (magnitude :float))
    (optional
      (reason :string))))

from code.world_web.lith_ecs import LithECS


def test_lith_ecs_dsl():
    ecs = LithECS()

    dsl_source = """
    (entity {:in :sim :id :e/duck :type :agent})
    (attach {:in :sim :e :e/duck :c :World.Pos :v {:x 1.2 :y -3.4}})
    (attach {:in :sim :e :e/duck :c :Mind.Intent :v {:goal :inspect :target :e/node-123}})
    
    (system
      {:id :sys/panel-layout
       :reads  [:UI.Anchor :World.Pos :Presence.Attach :Presence.Pressure]
       :writes [:UI.PanelPose :UI.Visibility]
       :budget {:ms 2 :writes-per-tick 200}})
       
    (obs {:ctx :presence/witness_thread
          :about {:e :e/node-123}
          :signal {:kind :related :to :e/node-77}
          :p 0.62
          :time 105
          :source "embed:cosine"})
          
    (belief-policy
      {:id :bp/ui-salience
       :combine :ema
       :alpha 0.25
       :conflict :keep-both})
       
    (entity {:in :sim :id :p/witness_thread :type :presence})
    (attach {:in :sim :e :p/witness_thread :c :Presence.Attach
             :v [{:target :e/node-1 :w 0.9}
                 {:target :cluster/ritual :w 0.7}
                 {:target :region/eta-mu :w 0.5}]})
                 
    (entity {:in :sim :id :e/node-1 :type :node})
    (entity {:in :sim :id :e/node-2 :type :node})
    (attach {:in :sim :e :p/witness_thread :c :Presence.Attach
             :v [{:target :e/node-2 :w 0.4}]})
    """

    print("Executing DSL...")
    results = ecs.execute(dsl_source)
    print(f"Results: {results}")

    print("\nState Snapshot:")
    snapshot = ecs.get_snapshot()
    for d in snapshot["datoms"]:
        print(f"  {d}")

    print("\nQuerying panels...")
    # (q {:find [?p] :where [[?p :type :agent]]})
    q_res = ecs.query({"find": ["?p"], "where": [["?p", "type", ":agent"]]})
    print(f"Query Result (agents): {q_res}")

    print("\nCalculating presence overlap...")
    overlap = ecs.get_presence_overlap(":e/node-1", ":e/node-2")
    print(f"Overlap between e/node-1 and e/node-2: {overlap:.4f}")


if __name__ == "__main__":
    test_lith_ecs_dsl()

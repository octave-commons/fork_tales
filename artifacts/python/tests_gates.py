# python/tests_gates.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from eta_mu_lisp_datalog import load_mu_program, compile_prompt_observations

DEFAULT_GATES = open(".opencode/promptdb/rules/gates-of-truth.mu.sexp", "r", encoding="utf-8").read()

def run_case(name: str, obs_src: str):
    prog = load_mu_program(DEFAULT_GATES)
    for f in compile_prompt_observations(obs_src):
        prog.add_fact(f)
    prog.run()
    violations = prog.query("violation")
    print(f"\n=== {name} ===")
    if not violations:
        print("No violations ✅")
    else:
        for v in violations:
            print(v)

case1 = r'''
(observation.v2
  (id "obs:self:pass")
  (vantage (agent eta-mu.world-daemon))
  (subject (who (kind self) (id eta-mu.world-daemon)))
  (channel (kind internal))
  (claims
    (claim (id c:self:1) (scope state) (key "ws.connected") (val "true") (confidence 0.9))))
'''

case2 = r'''
(observation.v2
  (id "obs:self:fail")
  (vantage (agent eta-mu.world-daemon))
  (subject (who (kind self) (id eta-mu.world-daemon)))
  (claims
    (claim (id c:self:2) (scope state) (key "ws.connected") (val "true") (confidence 0.9))))
'''

case3 = r'''
(observation.v2
  (id "obs:other:fail")
  (vantage (agent eta-mu.world-daemon))
  (subject (who (kind other) (id user:err)))
  (claims
    (claim (id c:other:1) (scope state) (key "user.mood") (val "happy") (confidence 0.6))))
'''

case4 = r'''
(observation.v2
  (id "obs:other:pass")
  (vantage (agent eta-mu.world-daemon))
  (subject (who (kind other) (id ui:connection)))
  (channel (kind external))
  (claims
    (claim (id c:other:2) (scope state) (key "ui.connection.status") (val "connected") (confidence 0.95))))
'''

run_case("CASE 1: SELF + internal channel default evidence (expected PASS)", case1)
run_case("CASE 2: SELF + no evidence (expected FAIL)", case2)
run_case("CASE 3: OTHER + disallowed state key (expected FAIL)", case3)
run_case("CASE 4: OTHER + allowed key + external default evidence (expected PASS)", case4)

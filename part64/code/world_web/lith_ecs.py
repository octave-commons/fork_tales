from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class Datom:
    """Internal canonical form: (e a v ctx p t src)"""

    e: str  # Entity ID
    a: str  # Attribute / Component ID
    v: Any  # Value payload
    ctx: str = "世"  # Context (己/汝/彼/世/主 or presence ID)
    p: float = 1.0  # Probability / Confidence [0..1]
    t: int = 0  # Tick / Time
    src: str = ""  # Provenance / Source


def parse_sexp(source: str) -> List[Any]:
    """Simple S-expression parser supporting ( ) [ ] { }."""
    # Handle comments starting with ;
    source = re.sub(r";.*", "", source)
    # Tokenizer
    tokens = []
    raw_tokens = re.finditer(r"\(|\)|\{|\}|\[|\]|\"([^\"]*)\"|([^\s(){} [\]]+)", source)
    for m in raw_tokens:
        if m.group(1) is not None:
            tokens.append(f'"{m.group(1)}"')
        elif m.group(2) is not None:
            tokens.append(m.group(2))
        else:
            tokens.append(m.group(0))

    def parse_tokens(tokens_list: List[str]) -> Any:
        if not tokens_list:
            return None
        token = tokens_list.pop(0)
        if token in ("(", "{", "["):
            closing = ")" if token == "(" else "}" if token == "{" else "]"
            lst = [token]
            while tokens_list and tokens_list[0] != closing:
                val = parse_tokens(tokens_list)
                if val is not None:
                    lst.append(val)
            if tokens_list:
                tokens_list.pop(0)  # pop closing
            return lst
        elif token in (")", "}", "]"):
            return None
        elif token.startswith('"'):
            return token[1:-1]
        elif token.startswith(":"):
            return token  # Keyword
        else:
            try:
                f = float(token)
                if f.is_integer():
                    return int(f)
                return f
            except ValueError:
                return token  # Symbol

    results = []
    while tokens:
        form = parse_tokens(tokens)
        if form is not None:
            results.append(form)
    return results


def lisp_to_python(val: Any) -> Any:
    """Convert Lisp structures into Python maps/lists."""
    if not isinstance(val, list) or not val:
        return val

    head = val[0]
    if head == "{":
        # {:k v :k2 v2}
        res = {}
        for i in range(1, len(val) - 1, 2):
            k = str(val[i]).lstrip(":")
            v = lisp_to_python(val[i + 1])
            res[k] = v
        return res
    if head == "[":
        # [v1 v2 v3]
        return [lisp_to_python(x) for x in val[1:]]
    if head == "(":
        # (v1 v2 v3) -> list in python
        return [lisp_to_python(x) for x in val[1:]]

    return [lisp_to_python(x) for x in val]


class LithECS:
    def __init__(self):
        self.datoms: List[Datom] = []
        self.entities: Set[str] = set()
        self.systems: Dict[str, Dict[str, Any]] = {}
        self.rules: List[Dict[str, Any]] = []
        self.promises: List[Dict[str, Any]] = []
        self.belief_policies: Dict[str, Dict[str, Any]] = {}
        self.tick_count: int = 0
        self.lock = threading.Lock()

    def execute(self, source: str) -> List[Any]:
        """Execute Lith DSL source."""
        forms = parse_sexp(source)
        results = []
        for form in forms:
            if not isinstance(form, list) or len(form) < 2 or form[0] != "(":
                continue

            # form is ["(", "cmd", args...]
            head = form[1]
            args_raw = form[2] if len(form) > 2 else {}
            args = lisp_to_python(args_raw)
            if not isinstance(args, dict):
                args = {}

            if head == "entity":
                results.append(
                    self.entity(
                        str(args.get("in", "")),
                        str(args.get("id", "")),
                        str(args.get("type", "")),
                    )
                )
            elif head == "attach":
                results.append(
                    self.attach(
                        str(args.get("in", "")),
                        str(args.get("e", "")),
                        str(args.get("c", "")),
                        args.get("v"),
                    )
                )
            elif head == "obs":
                about = args.get("about", {})
                if not isinstance(about, dict):
                    about = {}
                results.append(
                    self.obs(
                        str(args.get("ctx", "")),
                        str(about.get("e", "")),
                        args.get("signal", {}),
                        float(args.get("p", 1.0)),
                        args.get("time"),
                        str(args.get("source", "")),
                    )
                )
            elif head == "system":
                results.append(
                    self.system(
                        str(args.get("id", "")),
                        args.get("reads", []),
                        args.get("writes", []),
                        args.get("budget", {}),
                    )
                )
            elif head == "promise":
                results.append(
                    self.promise(
                        str(args.get("id", "")),
                        str(args.get("by", "")),
                        str(args.get("in", "")),
                        args.get("when", {}),
                        args.get("guarantee", {}),
                        str(args.get("fallback", "")),
                    )
                )
            elif head == "rule":
                results.append(
                    self.rule(
                        str(args.get("id", "")),
                        str(args.get("in", "")),
                        args.get("when", {}),
                        args.get("then", []),
                    )
                )
            elif head == "belief-policy":
                results.append(
                    self.belief_policy(
                        str(args.get("id", "")),
                        str(args.get("combine", "max")),
                        float(args.get("alpha", 0.5)),
                        str(args.get("conflict", "keep-both")),
                    )
                )
            elif head == "q":
                results.append(self.query(args))
        return results

    def entity(self, sim_id: str, entity_id: str, type_id: str) -> str:
        with self.lock:
            self.entities.add(entity_id)
            self._add_datom_locked(
                Datom(
                    e=entity_id,
                    a="type",
                    v=type_id,
                    t=self.tick_count,
                    src="dsl:entity",
                )
            )
        return entity_id

    def attach(self, sim_id: str, entity_id: str, component_id: str, value: Any):
        with self.lock:
            self._add_datom_locked(
                Datom(
                    e=entity_id,
                    a=component_id,
                    v=value,
                    t=self.tick_count,
                    src="dsl:attach",
                )
            )

    def obs(
        self,
        ctx: str,
        about_e: str,
        signal: Dict[str, Any],
        p: float,
        time_tick: Optional[int] = None,
        source: str = "",
    ):
        t = time_tick if time_tick is not None else self.tick_count
        with self.lock:
            if isinstance(signal, dict):
                for k, v in signal.items():
                    self._add_datom_locked(
                        Datom(e=about_e, a=k, v=v, ctx=ctx, p=p, t=t, src=source)
                    )

    def system(
        self,
        system_id: str,
        reads: List[str],
        writes: List[str],
        budget: Dict[str, Any],
    ):
        with self.lock:
            self.systems[system_id] = {
                "reads": list(reads),
                "writes": list(writes),
                "budget": dict(budget),
            }

    def promise(
        self,
        promise_id: str,
        system_id: str,
        sim_id: str,
        when: Dict[str, Any],
        guarantee: Dict[str, Any],
        fallback: str,
    ):
        with self.lock:
            self.promises.append(
                {
                    "id": promise_id,
                    "by": system_id,
                    "in": sim_id,
                    "when": dict(when),
                    "guarantee": dict(guarantee),
                    "fallback": fallback,
                }
            )

    def rule(
        self,
        rule_id: str,
        sim_id: str,
        when: Dict[str, Any],
        then: List[Dict[str, Any]],
    ):
        with self.lock:
            self.rules.append(
                {"id": rule_id, "sim": sim_id, "when": dict(when), "then": list(then)}
            )

    def belief_policy(self, policy_id: str, combine: str, alpha: float, conflict: str):
        with self.lock:
            self.belief_policies[policy_id] = {
                "combine": combine,
                "alpha": alpha,
                "conflict": conflict,
            }

    def query(self, q_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simplified pattern matching query engine."""
        with self.lock:
            where = q_map.get("where", [])
            if not isinstance(where, list):
                return []

            # 1. Collect latest state per E-A
            latest_state = {}
            for d in self.datoms:
                latest_state[(d.e, d.a)] = d

            results = [{}]
            for pattern in where:
                if not isinstance(pattern, list) or len(pattern) < 3:
                    continue
                new_results = []
                for binding in results:
                    e_p = (
                        binding.get(pattern[0], pattern[0])
                        if isinstance(pattern[0], str) and pattern[0].startswith("?")
                        else pattern[0]
                    )
                    a_p = (
                        binding.get(pattern[1], pattern[1])
                        if isinstance(pattern[1], str) and pattern[1].startswith("?")
                        else pattern[1]
                    )
                    v_p = (
                        binding.get(pattern[2], pattern[2])
                        if isinstance(pattern[2], str) and pattern[2].startswith("?")
                        else pattern[2]
                    )

                    for (e, a), d in latest_state.items():
                        match_e = (e == e_p) or (
                            isinstance(pattern[0], str) and pattern[0].startswith("?")
                        )
                        match_a = (a == a_p) or (
                            isinstance(pattern[1], str) and pattern[1].startswith("?")
                        )
                        match_v = (d.v == v_p) or (
                            isinstance(pattern[2], str) and pattern[2].startswith("?")
                        )

                        if match_e and match_a and match_v:
                            nb = dict(binding)
                            if isinstance(pattern[0], str) and pattern[0].startswith(
                                "?"
                            ):
                                nb[pattern[0]] = e
                            if isinstance(pattern[1], str) and pattern[1].startswith(
                                "?"
                            ):
                                nb[pattern[1]] = a
                            if isinstance(pattern[2], str) and pattern[2].startswith(
                                "?"
                            ):
                                nb[pattern[2]] = d.v
                            new_results.append(nb)
                results = new_results

            find_vars = q_map.get("find")
            if isinstance(find_vars, list):
                return [{v: r.get(v) for v in find_vars} for r in results]
            return results

    def get_presence_overlap(self, e1: str, e2: str) -> float:
        with self.lock:
            sig1 = self._get_presence_signature_locked(e1)
            sig2 = self._get_presence_signature_locked(e2)
            return self._cosine_similarity(sig1, sig2)

    def _get_presence_signature_locked(self, entity_id: str) -> Dict[str, float]:
        weights = {}
        for d in self.datoms:
            # Match both :Presence.Attach and Presence.Attach
            if d.a.lstrip(":") == "Presence.Attach" and isinstance(d.v, list):
                for entry in d.v:
                    if isinstance(entry, dict) and entry.get("target") == entity_id:
                        weights[d.e] = float(entry.get("w", 0.0))
        return weights

    def _cosine_similarity(
        self, sig1: Dict[str, float], sig2: Dict[str, float]
    ) -> float:
        all_presences = set(sig1.keys()) | set(sig2.keys())
        if not all_presences:
            return 0.0
        dot = sum(sig1.get(p, 0.0) * sig2.get(p, 0.0) for p in all_presences)
        mag1 = math.sqrt(sum(v * v for v in sig1.values()))
        mag2 = math.sqrt(sum(v * v for v in sig2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def step(self):
        with self.lock:
            self.tick_count += 1
            self._reconcile_locked()
            self._audit_locked()

    def _add_datom_locked(self, d: Datom):
        self.datoms.append(d)
        if len(self.datoms) > 10000:
            self.datoms.pop(0)

    def _reconcile_locked(self):
        pass

    def _audit_locked(self):
        # Check active rules and fire them
        for rule in self.rules:
            findings = self.query(rule["when"])
            for finding in findings:
                # Fire 'then' observations
                # ... rule execution logic ...
                pass

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "tick": self.tick_count,
                "entities": list(self.entities),
                "datoms": [
                    {
                        "e": d.e,
                        "a": d.a,
                        "v": d.v,
                        "ctx": d.ctx,
                        "p": d.p,
                        "t": d.t,
                        "src": d.src,
                    }
                    for d in self.datoms[-100:]
                ],
            }

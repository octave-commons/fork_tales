# python/eta_mu_lisp_datalog.py
# Tiny s-expr parser + stratified datalog engine + (observation.v2 ...) compiler
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import hashlib
import time


# -------------------------
# S-EXPR PARSER
# -------------------------

@dataclass(frozen=True)
class Sym:
    name: str

Atom = Union[Sym, str, int, float]
SExpr = Union[Atom, List["SExpr"]]


def tokenize(src: str) -> Iterator[str]:
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == ";":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c in ("(", ")"):
            yield c
            i += 1
            continue
        if c == '"':
            i += 1
            buf = []
            while i < n:
                ch = src[i]
                if ch == "\\":
                    i += 1
                    if i >= n:
                        raise ValueError("unterminated string escape")
                    esc = src[i]
                    buf.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(esc, esc))
                    i += 1
                    continue
                if ch == '"':
                    i += 1
                    break
                buf.append(ch)
                i += 1
            else:
                raise ValueError("unterminated string")
            yield '"' + "".join(buf) + '"'
            continue

        j = i
        while j < n and (not src[j].isspace()) and src[j] not in ("(", ")", ";"):
            j += 1
        yield src[i:j]
        i = j


def parse_atom(tok: str) -> Atom:
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    try:
        if "." in tok or "e" in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        return Sym(tok)


def parse_all(src: str) -> List[SExpr]:
    toks = list(tokenize(src))
    pos = 0

    def parse_one() -> SExpr:
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected EOF")
        t = toks[pos]
        pos += 1
        if t == "(":
            out: List[SExpr] = []
            while True:
                if pos >= len(toks):
                    raise ValueError("unterminated list")
                if toks[pos] == ")":
                    pos += 1
                    return out
                out.append(parse_one())
        if t == ")":
            raise ValueError("unexpected ')")
        return parse_atom(t)

    forms: List[SExpr] = []
    while pos < len(toks):
        forms.append(parse_one())
    return forms


# -------------------------
# DATALOG CORE (STRATIFIED)
# -------------------------

@dataclass(frozen=True)
class Var:
    name: str


Term = Union[Var, Any]


@dataclass(frozen=True)
class Fact:
    pred: str
    args: Tuple[Any, ...]


@dataclass(frozen=True)
class Rule:
    head_pred: str
    head_terms: Tuple[Term, ...]
    body: SExpr


def is_sym(x: Any, name: Optional[str] = None) -> bool:
    return isinstance(x, Sym) and (name is None or x.name == name)


def is_var_symbol(s: Sym) -> bool:
    return (s.name[:1].isupper()) or s.name.startswith("?")


def to_term(x: SExpr) -> Term:
    if isinstance(x, Sym):
        return Var(x.name) if is_var_symbol(x) else x.name
    return x


def to_pred_call(expr: SExpr) -> Optional[Tuple[str, Tuple[Term, ...]]]:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], Sym):
        return None
    op = expr[0].name
    if op in ("and", "not", "exists", "rule", "fact", "pred", "sort", "sig", "module"):
        return None
    if op == "=":
        return None
    terms = tuple(to_term(a) for a in expr[1:])
    return op, terms


def bind_term(t: Term, value: Any, env: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(t, Var):
        cur = env.get(t.name)
        if cur is None:
            e2 = dict(env)
            e2[t.name] = value
            return e2
        return env if cur == value else None
    return env if t == value else None


def unify_terms(pattern: Tuple[Term, ...], datum: Tuple[Any, ...], env: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if len(pattern) != len(datum):
        return None
    cur = env
    for t, v in zip(pattern, datum):
        cur2 = bind_term(t, v, cur)
        if cur2 is None:
            return None
        cur = cur2
    return cur


def solve(expr: SExpr, env: Dict[str, Any], index: Dict[str, List[Fact]]) -> Iterator[Dict[str, Any]]:
    if expr is None:
        yield env
        return

    if isinstance(expr, list) and expr and is_sym(expr[0], "and"):
        envs: List[Dict[str, Any]] = [env]
        for sub in expr[1:]:
            nxt: List[Dict[str, Any]] = []
            for e in envs:
                nxt.extend(list(solve(sub, e, index)))
            envs = nxt
            if not envs:
                break
        yield from envs
        return

    if isinstance(expr, list) and expr and is_sym(expr[0], "not"):
        sub = expr[1] if len(expr) > 1 else None
        for _ in solve(sub, dict(env), index):
            return
        yield env
        return

    if isinstance(expr, list) and expr and is_sym(expr[0], "="):
        if len(expr) != 3:
            return
        a = to_term(expr[1])
        b = to_term(expr[2])
        av = env.get(a.name) if isinstance(a, Var) else a
        bv = env.get(b.name) if isinstance(b, Var) else b
        if av is not None and bv is not None:
            if av == bv:
                yield env
            return
        if av is None and bv is not None:
            e2 = bind_term(a, bv, env)
            if e2 is not None:
                yield e2
            return
        if av is not None and bv is None:
            e2 = bind_term(b, av, env)
            if e2 is not None:
                yield e2
            return
        yield env
        return

    if isinstance(expr, list) and expr and is_sym(expr[0], "exists"):
        if len(expr) != 3 or not isinstance(expr[1], list):
            return
        vars_list = [v.name for v in expr[1] if isinstance(v, Sym)]
        sub = expr[2]
        for e2 in solve(sub, dict(env), index):
            pruned = dict(e2)
            for vn in vars_list:
                pruned.pop(vn, None)
            yield pruned
            return
        return

    pc = to_pred_call(expr)
    if pc is None:
        return
    pred, terms = pc
    for f in index.get(pred, []):
        e2 = unify_terms(terms, f.args, env)
        if e2 is not None:
            yield e2


def rule_body_deps(expr: SExpr, *, neg: bool = False) -> List[Tuple[str, bool]]:
    out: List[Tuple[str, bool]] = []
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], Sym):
        return out
    op = expr[0].name
    if op == "and":
        for sub in expr[1:]:
            out.extend(rule_body_deps(sub, neg=neg))
    elif op == "not":
        if len(expr) > 1:
            out.extend(rule_body_deps(expr[1], neg=True))
    elif op == "exists":
        if len(expr) == 3:
            out.extend(rule_body_deps(expr[2], neg=neg))
    elif op == "=":
        pass
    else:
        out.append((op, neg))
    return out


def stratify(rules: List[Rule]) -> Dict[str, int]:
    preds = {r.head_pred for r in rules}
    s: Dict[str, int] = {p: 0 for p in preds}
    for _ in range(256):
        changed = False
        for r in rules:
            h = r.head_pred
            for p, is_neg in rule_body_deps(r.body):
                if p not in s:
                    continue
                need = s[p] + (1 if is_neg else 0)
                if s[h] < need:
                    s[h] = need
                    changed = True
        if not changed:
            break
    return s


class Program:
    def __init__(self) -> None:
        self.facts: Dict[str, List[Fact]] = {}
        self.fact_set: set[Fact] = set()
        self.rules: List[Rule] = []

    def add_fact(self, f: Fact) -> None:
        if f in self.fact_set:
            return
        self.fact_set.add(f)
        self.facts.setdefault(f.pred, []).append(f)

    def add_rule(self, r: Rule) -> None:
        self.rules.append(r)

    def run(self) -> None:
        strata = stratify(self.rules)
        rules_by_stratum: Dict[int, List[Rule]] = {}
        for r in self.rules:
            rules_by_stratum.setdefault(strata.get(r.head_pred, 0), []).append(r)

        for k in sorted(rules_by_stratum.keys()):
            rs = rules_by_stratum[k]
            for _ in range(1024):
                changed = False
                idx = self.facts
                for r in rs:
                    for env in solve(r.body, {}, idx):
                        args: List[Any] = []
                        ok = True
                        for t in r.head_terms:
                            if isinstance(t, Var):
                                v = env.get(t.name)
                                if v is None:
                                    ok = False
                                    break
                                args.append(v)
                            else:
                                args.append(t)
                        if not ok:
                            continue
                        nf = Fact(r.head_pred, tuple(args))
                        if nf not in self.fact_set:
                            self.add_fact(nf)
                            changed = True
                if not changed:
                    break

    def query(self, pred: str) -> List[Fact]:
        return list(self.facts.get(pred, []))


def compile_fact(form: SExpr) -> Optional[Fact]:
    if not (isinstance(form, list) and len(form) == 2 and is_sym(form[0], "fact")):
        return None
    inner = form[1]
    if not isinstance(inner, list) or not inner or not isinstance(inner[0], Sym):
        return None
    pred = inner[0].name
    args = [(a.name if isinstance(a, Sym) else a) for a in inner[1:]]
    return Fact(pred, tuple(args))


def compile_rule(form: SExpr) -> Optional[Rule]:
    if not (isinstance(form, list) and len(form) == 3 and is_sym(form[0], "rule")):
        return None
    head = form[1]
    body = form[2]
    if not isinstance(head, list) or not head or not isinstance(head[0], Sym):
        return None
    head_pred = head[0].name
    head_terms = tuple(to_term(x) for x in head[1:])
    return Rule(head_pred=head_pred, head_terms=head_terms, body=body)


def load_mu_program(src: str) -> Program:
    prog = Program()
    for form in parse_all(src):
        f = compile_fact(form)
        if f is not None:
            prog.add_fact(f)
            continue
        r = compile_rule(form)
        if r is not None:
            prog.add_rule(r)
            continue
    return prog


# -------------------------
# (observation.v2 ...) compiler (minimal)
# -------------------------

def _get_kv(lst: List[SExpr], key: str) -> Optional[List[SExpr]]:
    for x in lst:
        if isinstance(x, list) and x and is_sym(x[0], key):
            return x
    return None


def find_forms(tree: SExpr, head: str) -> List[List[SExpr]]:
    out: List[List[SExpr]] = []
    if isinstance(tree, list) and tree and isinstance(tree[0], Sym) and tree[0].name == head:
        out.append(tree)
    if isinstance(tree, list):
        for x in tree:
            out.extend(find_forms(x, head))
    return out


def compile_observation_v2(obs_form: List[SExpr]) -> List[Fact]:
    now = int(time.time())
    obs_id = f"obs:{hashlib.sha256(str(obs_form).encode()).hexdigest()[:12]}:{now}"
    for x in obs_form[1:]:
        if isinstance(x, list) and x and is_sym(x[0], "id") and len(x) >= 2:
            obs_id = x[1].name if isinstance(x[1], Sym) else str(x[1])

    facts: List[Fact] = [Fact("obs", (obs_id,))]

    v_agent = "unknown-agent"
    v = _get_kv(obs_form[1:], "vantage")
    if v:
        for it in v[1:]:
            if isinstance(it, list) and it and is_sym(it[0], "agent") and len(it) >= 2:
                v_agent = it[1].name if isinstance(it[1], Sym) else str(it[1])
                break
    facts.append(Fact("vantage", (obs_id, v_agent)))

    s_id = None
    subj = _get_kv(obs_form[1:], "subject")
    if subj:
        who = _get_kv(subj[1:], "who")
        if who:
            idf = _get_kv(who[1:], "id")
            if idf and len(idf) >= 2:
                s_id = idf[1].name if isinstance(idf[1], Sym) else str(idf[1])
    if s_id is None:
        s_id = v_agent
    facts.append(Fact("subject", (obs_id, s_id)))

    ch = _get_kv(obs_form[1:], "channel")
    channel_kind = None
    if ch:
        k = _get_kv(ch[1:], "kind")
        if k and len(k) >= 2:
            channel_kind = k[1].name if isinstance(k[1], Sym) else str(k[1])
    default_ev_kind = "internal" if channel_kind == "internal" else "external" if channel_kind == "external" else None

    claims = _get_kv(obs_form[1:], "claims")
    claim_items: List[List[SExpr]] = []
    if claims:
        for it in claims[1:]:
            if isinstance(it, list) and it and is_sym(it[0], "claim"):
                claim_items.append(it)

    for idx, cform in enumerate(claim_items, start=1):
        cid = f"c:{obs_id}:{idx}"
        for x in cform[1:]:
            if isinstance(x, list) and x and is_sym(x[0], "id") and len(x) >= 2:
                cid = x[1].name if isinstance(x[1], Sym) else str(x[1])

        facts.append(Fact("claim", (cid,)))
        facts.append(Fact("in-obs", (cid, obs_id)))

        sc = _get_kv(cform[1:], "scope")
        scope = (
            sc[1].name
            if sc and len(sc) >= 2 and isinstance(sc[1], Sym)
            else (str(sc[1]) if sc and len(sc) >= 2 else "perception")
        )
        facts.append(Fact("scope", (cid, scope)))

        kf = _get_kv(cform[1:], "key")
        vf = _get_kv(cform[1:], "val")
        tf = _get_kv(cform[1:], "text")
        key = (
            kf[1].name
            if kf and len(kf) >= 2 and isinstance(kf[1], Sym)
            else (str(kf[1]) if kf and len(kf) >= 2 else "text")
        )
        val = (
            vf[1].name
            if vf and len(vf) >= 2 and isinstance(vf[1], Sym)
            else (str(vf[1]) if vf and len(vf) >= 2 else (tf[1] if tf and len(tf) >= 2 else ""))
        )
        facts.append(Fact("key", (cid, str(key))))
        facts.append(Fact("val", (cid, str(val))))

        cf = _get_kv(cform[1:], "confidence")
        conf = float(cf[1]) if cf and len(cf) >= 2 and isinstance(cf[1], (int, float)) else 1.0
        facts.append(Fact("conf", (cid, conf)))

        evrefs = _get_kv(cform[1:], "evidence-refs")
        if evrefs and len(evrefs) >= 2:
            for j, ref in enumerate(evrefs[1:], start=1):
                evid = f"ev:{cid}:{j}"
                facts.append(Fact("ev", (evid,)))
                facts.append(Fact("ev-for", (evid, cid)))
                rid = ref.name if isinstance(ref, Sym) else str(ref)
                kind = "internal" if any(x in rid for x in ("tool", "trace", "telemetry", "state", "prompt")) else "external"
                facts.append(Fact("ev-kind", (evid, kind)))
        elif default_ev_kind is not None:
            evid = f"ev:{cid}:1"
            facts.append(Fact("ev", (evid,)))
            facts.append(Fact("ev-for", (evid, cid)))
            facts.append(Fact("ev-kind", (evid, default_ev_kind)))

    return facts


def compile_prompt_observations(src: str) -> List[Fact]:
    facts: List[Fact] = []
    for form in parse_all(src):
        for obs in find_forms(form, "observation.v2"):
            facts.extend(compile_observation_v2(obs))
    return facts

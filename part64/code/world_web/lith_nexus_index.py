from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LITH_NEXUS_ROOTS: tuple[str, ...] = (
    "contracts",
    ".opencode/knowledge",
    ".opencode/promptdb",
    ".opencode/protocol",
    "specs",
    "manifest.lith",
)
DEFAULT_LITH_NEXUS_INCLUDE_EXT: tuple[str, ...] = (".lith", ".lisp", ".md")
DEFAULT_LITH_NEXUS_IGNORE_GLOBS: tuple[str, ...] = (
    "**/node_modules/**",
    "**/dist/**",
    "**/.git/**",
    "**/__pycache__/**",
)
MARKDOWN_LITH_LANGS: set[str] = {"lith", "lisp", "sexp", "clj"}

LITH_HEAD_TO_KIND: dict[str, str] = {
    "packet": "packet",
    "contract": "contract",
    "契": "contract",
    "protocol": "protocol",
    "fact": "fact",
    "obs": "observation",
    "q": "question",
    "manifest": "spec",
}

REF_HEADS: set[str] = {
    "ref",
    "refs",
    "required-refs",
    "path",
    "paths",
    "uri",
    "uris",
    "depends",
    "depends_on",
    "depends-on",
    "target",
    "source",
    "file",
    "files",
    "contract",
    "packet",
    "resource",
    "resource_id",
    "packet_id",
    "contract_id",
    "proof_refs",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_ratio(seed: str, salt: int = 0) -> float:
    digest = hashlib.sha256(f"{seed}|{salt}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big")
    return raw / float(2**64 - 1)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quote_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _path_uri(rel_path: str) -> str:
    return f"library:/{rel_path}"


def _form_uri(rel_path: str, form_id: str) -> str:
    return f"lith://repo/{rel_path}#form={form_id}"


def _node_point(offset: int, line: int, column: int) -> dict[str, int]:
    return {"offset": int(offset), "line": int(line), "column": int(column)}


def _node_span(
    path: str,
    start_offset: int,
    start_line: int,
    start_column: int,
    end_offset: int,
    end_line: int,
    end_column: int,
) -> dict[str, Any]:
    return {
        "path": path,
        "start": _node_point(start_offset, start_line, start_column),
        "end": _node_point(end_offset, end_line, end_column),
    }


class LithSyntaxError(ValueError):
    def __init__(self, code: str, message: str, span: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.span = span


def _syntax_error(
    code: str,
    message: str,
    path: str,
    start: dict[str, int],
    end: dict[str, int] | None = None,
) -> LithSyntaxError:
    return LithSyntaxError(
        code,
        message,
        _node_span(
            path,
            int(start.get("offset", 0)),
            int(start.get("line", 1)),
            int(start.get("column", 1)),
            int((end or start).get("offset", 0)),
            int((end or start).get("line", 1)),
            int((end or start).get("column", 1)),
        ),
    )


def parse_lith(
    source: str,
    *,
    file_path: str,
    base_offset: int = 0,
    base_line: int = 1,
) -> list[dict[str, Any]]:
    index = 0
    line = base_line
    column = 1
    source_len = len(source)

    def point() -> dict[str, int]:
        return {"offset": base_offset + index, "line": line, "column": column}

    def advance() -> str:
        nonlocal index, line, column
        char = source[index] if index < source_len else ""
        index += 1
        if char == "\n":
            line += 1
            column = 1
        else:
            column += 1
        return char

    def peek() -> str:
        return source[index] if index < source_len else ""

    def skip_space() -> None:
        while index < source_len:
            char = peek()
            if char == ";":
                while index < source_len and peek() != "\n":
                    advance()
                continue
            if char in {" ", "\t", "\r", "\n", ","}:
                advance()
                continue
            break

    def parse_string() -> dict[str, Any]:
        start_idx = index
        start_point = point()
        advance()
        value = []
        while index < source_len:
            char = advance()
            if char == "\\":
                if index >= source_len:
                    raise _syntax_error(
                        "E_STRING_ESCAPE",
                        "unterminated string escape",
                        file_path,
                        start_point,
                        point(),
                    )
                escaped = advance()
                if escaped == "n":
                    value.append("\n")
                elif escaped == "r":
                    value.append("\r")
                elif escaped == "t":
                    value.append("\t")
                else:
                    value.append(escaped)
                continue
            if char == '"':
                end_point = point()
                return {
                    "type": "string",
                    "value": "".join(value),
                    "text": source[start_idx:index],
                    "span": _node_span(
                        file_path,
                        int(start_point["offset"]),
                        int(start_point["line"]),
                        int(start_point["column"]),
                        int(end_point["offset"]),
                        int(end_point["line"]),
                        int(end_point["column"]),
                    ),
                }
            value.append(char)
        raise _syntax_error(
            "E_STRING_UNTERM",
            "unterminated string",
            file_path,
            start_point,
            point(),
        )

    def parse_atom() -> dict[str, Any]:
        start_idx = index
        start_point = point()
        token_chars: list[str] = []
        while index < source_len:
            char = peek()
            if char in {"", "(", ")", "[", "]", ";"} or char.isspace() or char == ",":
                break
            token_chars.append(advance())
        raw = "".join(token_chars)
        if not raw:
            raise _syntax_error(
                "E_TOKEN", "unexpected token", file_path, start_point, point()
            )
        end_point = point()
        node_type = "symbol"
        value: Any = raw
        if raw.startswith(":"):
            node_type = "keyword"
        else:
            try:
                if any(marker in raw for marker in (".", "e", "E")):
                    value = float(raw)
                else:
                    value = int(raw)
                node_type = "number"
            except ValueError:
                value = raw
        return {
            "type": node_type,
            "value": value,
            "raw": raw,
            "text": source[start_idx:index],
            "span": _node_span(
                file_path,
                int(start_point["offset"]),
                int(start_point["line"]),
                int(start_point["column"]),
                int(end_point["offset"]),
                int(end_point["line"]),
                int(end_point["column"]),
            ),
        }

    def parse_sequence(closing: str, node_type: str) -> dict[str, Any]:
        start_idx = index
        start_point = point()
        advance()
        items: list[dict[str, Any]] = []
        while True:
            skip_space()
            if index >= source_len:
                raise _syntax_error(
                    "E_LIST_UNTERM",
                    f"unterminated {node_type}",
                    file_path,
                    start_point,
                    point(),
                )
            if peek() == closing:
                advance()
                end_point = point()
                return {
                    "type": node_type,
                    "items": items,
                    "text": source[start_idx:index],
                    "span": _node_span(
                        file_path,
                        int(start_point["offset"]),
                        int(start_point["line"]),
                        int(start_point["column"]),
                        int(end_point["offset"]),
                        int(end_point["line"]),
                        int(end_point["column"]),
                    ),
                }
            items.append(parse_form())

    def parse_form() -> dict[str, Any]:
        skip_space()
        if index >= source_len:
            raise _syntax_error("E_EOF", "unexpected end of input", file_path, point())
        char = peek()
        if char == "(":
            return parse_sequence(")", "list")
        if char == "[":
            return parse_sequence("]", "vector")
        if char == ")" or char == "]":
            raise _syntax_error(
                "E_CLOSE",
                f"unexpected '{char}'",
                file_path,
                point(),
                point(),
            )
        if char == '"':
            return parse_string()
        return parse_atom()

    forms: list[dict[str, Any]] = []
    while True:
        skip_space()
        if index >= source_len:
            break
        if forms and peek() in {")", "]"}:
            break
        forms.append(parse_form())
    return forms


def print_lith(node: dict[str, Any]) -> str:
    node_type = str(node.get("type", ""))
    if node_type == "list":
        return "(" + " ".join(print_lith(item) for item in node.get("items", [])) + ")"
    if node_type == "vector":
        return "[" + " ".join(print_lith(item) for item in node.get("items", [])) + "]"
    if node_type == "string":
        return _quote_string(str(node.get("value", "")))
    if node_type == "number":
        raw = str(node.get("raw", "") or "")
        if raw:
            return raw
        return str(node.get("value", 0))
    return str(node.get("raw", node.get("value", "")) or "")


def extract_markdown_lith_blocks(source: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = source.splitlines(keepends=True)
    offset = 0
    inside = False
    language = ""
    block_start_offset = 0
    block_start_line = 1
    buffer: list[str] = []

    for line_index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not inside and stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            if lang in MARKDOWN_LITH_LANGS:
                inside = True
                language = lang
                buffer = []
                block_start_offset = offset + len(line)
                block_start_line = line_index + 1
            offset += len(line)
            continue
        if inside and stripped.startswith("```"):
            blocks.append(
                {
                    "language": language,
                    "text": "".join(buffer),
                    "start_offset": block_start_offset,
                    "start_line": block_start_line,
                }
            )
            inside = False
            language = ""
            buffer = []
            offset += len(line)
            continue
        if inside:
            buffer.append(line)
        offset += len(line)
    return blocks


def _is_list(node: dict[str, Any]) -> bool:
    return str(node.get("type", "")) == "list"


def _is_vector(node: dict[str, Any]) -> bool:
    return str(node.get("type", "")) == "vector"


def _node_text_value(node: dict[str, Any] | None) -> str | None:
    if not isinstance(node, dict):
        return None
    node_type = str(node.get("type", ""))
    if node_type == "number":
        raw = str(node.get("raw", "") or "")
        return raw or str(node.get("value", 0))
    if node_type in {"string", "symbol", "keyword"}:
        return str(node.get("value", "") or "")
    return None


def _head_symbol(node: dict[str, Any]) -> str | None:
    if not _is_list(node):
        return None
    items = node.get("items", [])
    if not items:
        return None
    return _node_text_value(items[0])


def _child_form(node: dict[str, Any], head: str) -> dict[str, Any] | None:
    if not _is_list(node):
        return None
    for item in node.get("items", []):
        if _is_list(item) and _head_symbol(item) == head:
            return item
    return None


def _child_value(node: dict[str, Any], head: str) -> dict[str, Any] | None:
    child = _child_form(node, head)
    if not child:
        return None
    items = child.get("items", [])
    return items[1] if len(items) > 1 else None


def _vector_strings(node: dict[str, Any] | None) -> list[str]:
    if not isinstance(node, dict) or not (_is_vector(node) or _is_list(node)):
        return []
    rows: list[str] = []
    for item in node.get("items", []):
        value = _node_text_value(item)
        if value:
            rows.append(value)
    return rows


def _bool_value(node: dict[str, Any] | None, default: bool) -> bool:
    value = (_node_text_value(node) or "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return bool(default)


def load_lith_nexus_config(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "mcp.lith-nexus.config.lith"
    config: dict[str, Any] = {
        "roots": list(DEFAULT_LITH_NEXUS_ROOTS),
        "include_ext": list(DEFAULT_LITH_NEXUS_INCLUDE_EXT),
        "ignore_glob": list(DEFAULT_LITH_NEXUS_IGNORE_GLOBS),
        "mime": {"lith": "text/x-lith", "md": "text/markdown"},
        "writes": {
            "facts_dir": ".opencode/promptdb/facts",
            "inbox_dir": ".opencode/knowledge/inbox",
            "allow_secret_writes": False,
        },
        "index": {"use_git_ls_files": True, "watch": False},
        "config_path": str(config_path),
    }
    if not config_path.exists() or not config_path.is_file():
        return config
    try:
        forms = parse_lith(config_path.read_text("utf-8"), file_path=str(config_path))
    except Exception:
        return config
    root = next(
        (
            form
            for form in forms
            if _is_list(form) and (_head_symbol(form) or "") == "config"
        ),
        None,
    )
    if not root:
        return config
    roots = _vector_strings(_child_value(root, "roots"))
    include_ext = _vector_strings(_child_value(root, "include_ext"))
    ignore_glob = _vector_strings(_child_value(root, "ignore_glob"))
    if roots:
        config["roots"] = roots
    if include_ext:
        config["include_ext"] = include_ext
    if ignore_glob:
        config["ignore_glob"] = ignore_glob
    mime_form = _child_form(root, "mime")
    if mime_form:
        lith_mime = _node_text_value(_child_value(mime_form, "lith"))
        md_mime = _node_text_value(_child_value(mime_form, "md"))
        if lith_mime:
            config["mime"]["lith"] = lith_mime
        if md_mime:
            config["mime"]["md"] = md_mime
    writes_form = _child_form(root, "writes")
    if writes_form:
        facts_dir = _node_text_value(_child_value(writes_form, "facts_dir"))
        inbox_dir = _node_text_value(_child_value(writes_form, "inbox_dir"))
        if facts_dir:
            config["writes"]["facts_dir"] = facts_dir
        if inbox_dir:
            config["writes"]["inbox_dir"] = inbox_dir
        config["writes"]["allow_secret_writes"] = _bool_value(
            _child_value(writes_form, "allow_secret_writes"),
            False,
        )
    index_form = _child_form(root, "index")
    if index_form:
        config["index"]["use_git_ls_files"] = _bool_value(
            _child_value(index_form, "use_git_ls_files"),
            True,
        )
        config["index"]["watch"] = _bool_value(_child_value(index_form, "watch"), False)
    return config


def _normalize_rel_path(path_value: str) -> str:
    text = str(path_value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _source_bucket(rel_path: str) -> str:
    if rel_path == "manifest.lith":
        return "manifest"
    if rel_path.startswith(".opencode/promptdb/"):
        return "promptdb"
    if rel_path.startswith(".opencode/protocol/"):
        return "protocol"
    if rel_path.startswith("contracts/"):
        return "contracts"
    if rel_path.startswith("specs/"):
        return "specs"
    return "repo"


def _is_ignored(rel_path: str, ignore_globs: list[str]) -> bool:
    clean = _normalize_rel_path(rel_path)
    for pattern in ignore_globs:
        if fnmatch.fnmatch(clean, pattern):
            return True
    return False


def _within_roots(rel_path: str, roots: list[str]) -> bool:
    clean = _normalize_rel_path(rel_path)
    for root in roots:
        normalized_root = _normalize_rel_path(root)
        if not normalized_root:
            continue
        if clean == normalized_root:
            return True
        if clean.startswith(normalized_root.rstrip("/") + "/"):
            return True
    return False


def _collect_git_paths(repo_root: Path, roots: list[str]) -> list[Path]:
    command = ["git", "-C", str(repo_root), "ls-files", "-co", "--exclude-standard"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    paths: list[Path] = []
    for raw_line in proc.stdout.splitlines():
        rel_path = _normalize_rel_path(raw_line)
        if not rel_path or not _within_roots(rel_path, roots):
            continue
        abs_path = repo_root / rel_path
        if abs_path.exists() and abs_path.is_file():
            paths.append(abs_path)
    return paths


def _collect_submodule_git_paths(repo_root: Path, roots: list[str]) -> list[Path]:
    command = ["git", "-C", str(repo_root), "submodule", "status", "--recursive"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return []
    if proc.returncode != 0:
        return []

    paths: list[Path] = []
    for raw_line in proc.stdout.splitlines():
        line = str(raw_line or "").rstrip()
        if not line:
            continue
        normalized = line.lstrip(" +-U")
        parts = normalized.split()
        if len(parts) < 2:
            continue
        submodule_rel = _normalize_rel_path(parts[1])
        if not submodule_rel:
            continue
        submodule_abs = repo_root / submodule_rel
        if not submodule_abs.exists() or not submodule_abs.is_dir():
            continue
        try:
            sub_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(submodule_abs),
                    "ls-files",
                    "-co",
                    "--exclude-standard",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            continue
        if sub_proc.returncode != 0:
            continue
        for sub_line in sub_proc.stdout.splitlines():
            child_rel = _normalize_rel_path(sub_line)
            if not child_rel:
                continue
            rel_path = _normalize_rel_path(f"{submodule_rel}/{child_rel}")
            if not rel_path or not _within_roots(rel_path, roots):
                continue
            abs_path = repo_root / rel_path
            if abs_path.exists() and abs_path.is_file():
                paths.append(abs_path)
    return paths


def _collect_scan_paths(repo_root: Path, roots: list[str]) -> list[Path]:
    rows: list[Path] = []
    for root in roots:
        target = repo_root / root
        if target.is_file():
            rows.append(target)
            continue
        if not target.exists() or not target.is_dir():
            continue
        rows.extend(path for path in target.rglob("*") if path.is_file())
    return rows


def _collect_index_paths(repo_root: Path, config: dict[str, Any]) -> list[Path]:
    roots = [str(item) for item in config.get("roots", []) if str(item).strip()]
    include_ext = {
        str(item).lower() for item in config.get("include_ext", []) if str(item).strip()
    }
    ignore_globs = [
        str(item) for item in config.get("ignore_glob", []) if str(item).strip()
    ]
    use_git = bool(((config.get("index") or {}).get("use_git_ls_files", True)))
    candidates = _collect_git_paths(repo_root, roots) if use_git else []
    if use_git:
        candidates.extend(_collect_submodule_git_paths(repo_root, roots))
    if not candidates:
        candidates = _collect_scan_paths(repo_root, roots)
    deduped: dict[Path, Path] = {}
    for path in candidates:
        abs_path = path.resolve()
        try:
            rel_path = _normalize_rel_path(str(abs_path.relative_to(repo_root)))
        except ValueError:
            continue
        if include_ext and abs_path.suffix.lower() not in include_ext:
            continue
        if _is_ignored(rel_path, ignore_globs):
            continue
        deduped[abs_path] = abs_path
    return [deduped[key] for key in sorted(deduped, key=lambda item: str(item))]


def _walk_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [node]
    for child in (
        node.get("items", []) if isinstance(node.get("items", []), list) else []
    ):
        if isinstance(child, dict):
            rows.extend(_walk_nodes(child))
    return rows


def _extract_tags(form: dict[str, Any]) -> list[str]:
    tags = _vector_strings(_child_value(form, "tags"))
    return [tag for tag in tags if tag]


def _extract_explicit_id(form: dict[str, Any]) -> str | None:
    value = _node_text_value(_child_value(form, "id"))
    if value:
        return value
    head = _head_symbol(form) or ""
    items = form.get("items", [])
    if head in {"contract", "契"} and len(items) > 1:
        return _node_text_value(items[1])
    return None


def _extract_title(form: dict[str, Any], declared_kind: str, fallback: str) -> str:
    title = _node_text_value(_child_value(form, "title"))
    if title:
        return title
    head = _head_symbol(form) or ""
    items = form.get("items", [])
    if head in {"contract", "契"} and len(items) > 1:
        contract_name = _node_text_value(items[1])
        if contract_name:
            return contract_name
    prompt = _node_text_value(_child_value(form, "claim")) or _node_text_value(
        _child_value(form, "ask")
    )
    if prompt:
        return prompt[:160]
    return fallback or declared_kind


def _looks_repo_path(value: str) -> bool:
    clean = _normalize_rel_path(value)
    return (
        any(
            clean.endswith(ext)
            for ext in (
                ".lisp",
                ".lith",
                ".md",
                ".json",
                ".txt",
                ".contract",
                ".packet",
            )
        )
        or "/" in clean
    )


def _looks_explicit_ref(value: str) -> bool:
    clean = str(value or "").strip()
    if not clean or " " in clean or "://" in clean:
        return False
    return ":" in clean


def _collect_reference_values(form: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    def visit(node: dict[str, Any], parent_head: str | None = None) -> None:
        if _is_list(node):
            head = _head_symbol(node)
            items = node.get("items", [])
            capture_children = bool(head and head in REF_HEADS)
            for child_index, child in enumerate(items[1:] if head else items):
                if not isinstance(child, dict):
                    continue
                if capture_children:
                    value = _node_text_value(child)
                    if value:
                        refs.append(value)
                        continue
                visit(child, head or parent_head)
            return
        if _is_vector(node):
            for child in node.get("items", []):
                if isinstance(child, dict):
                    visit(child, parent_head)
            return
        value = _node_text_value(node)
        if not value:
            return
        if str(parent_head or "") in REF_HEADS:
            refs.append(value)
            return
        if value.startswith("http://") or value.startswith("https://"):
            refs.append(value)
            return
        if _looks_repo_path(value) or _looks_explicit_ref(value):
            refs.append(value)

    visit(form)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in refs:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _form_summary(
    *,
    rel_path: str,
    file_sha256: str,
    form: dict[str, Any],
    source_kind: str,
    include_text: bool,
    spec_node_id: str | None = None,
) -> dict[str, Any]:
    canonical = print_lith(form)
    form_sha256 = _sha256_text(canonical)
    explicit_id = _extract_explicit_id(form)
    head = _head_symbol(form) or "form"
    declared_kind = LITH_HEAD_TO_KIND.get(head, "form")
    stable_seed = f"{rel_path}:{form_sha256}"
    form_id = f"lith:form:{_sha256_text(stable_seed)[:24]}"
    title = _extract_title(form, declared_kind, explicit_id or head)
    tags = _extract_tags(form)
    refs = _collect_reference_values(form)
    span = dict(form.get("span", {}))
    summary = {
        "id": form_id,
        "stable_id": explicit_id or form_id,
        "explicit_id": explicit_id,
        "head": head,
        "kind": declared_kind,
        "title": title,
        "path": rel_path,
        "file_sha256": file_sha256,
        "form_sha256": form_sha256,
        "canonical": canonical if include_text else "",
        "tags": tags,
        "refs": refs,
        "span": span,
        "uri": _form_uri(rel_path, form_id),
        "source_kind": source_kind,
        "spec_node_id": spec_node_id,
    }
    if include_text:
        summary["text"] = str(form.get("text", ""))
    return summary


def _node_coordinates(seed: str, ring: int = 0) -> tuple[float, float]:
    angle = _stable_ratio(seed, ring) * math.tau
    radius = 0.18 + (_stable_ratio(seed, ring + 1) * 0.28)
    x = 0.5 + math.cos(angle) * radius
    y = 0.5 + math.sin(angle) * radius
    return round(_clamp01(x), 4), round(_clamp01(y), 4)


def _append_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    source: str,
    target: str,
    kind: str,
    weight: float,
    provenance: dict[str, Any],
) -> None:
    key = (source, target, kind)
    if not source or not target or key in seen:
        return
    seen.add(key)
    edges.append(
        {
            "id": f"lith:edge:{_sha256_text(f'{source}|{target}|{kind}')[:24]}",
            "source": source,
            "target": target,
            "kind": kind,
            "weight": round(_clamp01(weight), 4),
            "provenance": dict(provenance),
        }
    )


def collect_lith_nexus_index(
    repo_root: Path,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve()
    config = load_lith_nexus_config(root)
    files = _collect_index_paths(root, config)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[str, str, str]] = set()
    errors: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    tag_nodes_by_tag: dict[str, str] = {}
    file_nodes_by_path: dict[str, str] = {}
    semantic_nodes_by_explicit_id: dict[str, str] = {}
    pending_refs: list[dict[str, Any]] = []
    node_ids_seen: set[str] = set()

    def append_node(
        node: dict[str, Any], *, duplicate_error: str | None = None
    ) -> bool:
        node_id = str(node.get("id", "") or "")
        if not node_id:
            return False
        if node_id in node_ids_seen:
            if duplicate_error:
                errors.append(
                    {"path": duplicate_error, "error": f"duplicate_node_id:{node_id}"}
                )
            return False
        node_ids_seen.add(node_id)
        nodes.append(node)
        return True

    def ensure_file_node(
        rel_path: str, *, file_sha256: str | None = None, label: str | None = None
    ) -> str:
        normalized = _normalize_rel_path(rel_path)
        if normalized in file_nodes_by_path:
            return file_nodes_by_path[normalized]
        file_node_id = f"lith:file:{_sha256_text(normalized)[:24]}"
        x, y = _node_coordinates(normalized, 2)
        append_node(
            {
                "id": file_node_id,
                "kind": "file",
                "label": label or Path(normalized).name or normalized,
                "title": label or Path(normalized).name or normalized,
                "path": normalized,
                "source_uri": _path_uri(normalized),
                "x": x,
                "y": y,
                "confidence": 1.0,
                "provenance": {
                    "path": normalized,
                    "source_uri": _path_uri(normalized),
                    "sha256": file_sha256 or "",
                },
                "extension": {
                    "path": normalized,
                    "source_kind": _source_bucket(normalized),
                },
            }
        )
        file_nodes_by_path[normalized] = file_node_id
        return file_node_id

    def ensure_tag_node(tag: str) -> str:
        if tag in tag_nodes_by_tag:
            return tag_nodes_by_tag[tag]
        tag_node_id = f"lith:tag:{_sha256_text(tag)[:24]}"
        x, y = _node_coordinates(tag, 5)
        append_node(
            {
                "id": tag_node_id,
                "kind": "tag",
                "label": tag,
                "title": tag,
                "x": x,
                "y": y,
                "confidence": 1.0,
                "provenance": {"tag": tag},
                "extension": {"tag": tag},
            }
        )
        tag_nodes_by_tag[tag] = tag_node_id
        return tag_node_id

    for abs_path in files:
        rel_path = _normalize_rel_path(str(abs_path.relative_to(root)))
        source_kind = _source_bucket(rel_path)
        try:
            source = abs_path.read_text("utf-8")
        except Exception as exc:
            errors.append({"path": rel_path, "error": f"read_error:{exc}"})
            continue
        file_sha256 = _sha256_text(source)
        file_node_id = ensure_file_node(rel_path, file_sha256=file_sha256)
        file_row = {
            "path": rel_path,
            "sha256": file_sha256,
            "source_kind": source_kind,
            "form_count": 0,
            "packet_count": 0,
            "contract_count": 0,
            "fact_count": 0,
            "spec_node_id": "",
        }

        spec_node_id: str | None = None
        if abs_path.suffix.lower() == ".md":
            spec_node_id = f"lith:spec:{_sha256_text(rel_path)[:24]}"
            sx, sy = _node_coordinates(rel_path, 7)
            append_node(
                {
                    "id": spec_node_id,
                    "kind": "spec",
                    "label": Path(rel_path).stem,
                    "title": Path(rel_path).stem,
                    "path": rel_path,
                    "source_uri": _path_uri(rel_path),
                    "x": sx,
                    "y": sy,
                    "confidence": 1.0,
                    "provenance": {
                        "path": rel_path,
                        "source_uri": _path_uri(rel_path),
                        "sha256": file_sha256,
                    },
                    "extension": {
                        "path": rel_path,
                        "source_kind": source_kind,
                        "mime_type": config.get("mime", {}).get("md", "text/markdown"),
                    },
                }
            )
            _append_edge(
                edges,
                edge_seen,
                source=file_node_id,
                target=spec_node_id,
                kind="declares",
                weight=1.0,
                provenance={"path": rel_path, "sha256": file_sha256},
            )
            file_row["spec_node_id"] = spec_node_id

        try:
            blocks = (
                extract_markdown_lith_blocks(source)
                if abs_path.suffix.lower() == ".md"
                else [
                    {
                        "language": abs_path.suffix.lower().lstrip("."),
                        "text": source,
                        "start_offset": 0,
                        "start_line": 1,
                    }
                ]
            )
            parsed_forms: list[dict[str, Any]] = []
            for block in blocks:
                parsed_forms.extend(
                    parse_lith(
                        str(block.get("text", "")),
                        file_path=rel_path,
                        base_offset=int(block.get("start_offset", 0)),
                        base_line=int(block.get("start_line", 1)),
                    )
                )
        except Exception as exc:
            errors.append({"path": rel_path, "error": f"parse_error:{exc}"})
            file_rows.append(file_row)
            continue

        for form in parsed_forms:
            summary = _form_summary(
                rel_path=rel_path,
                file_sha256=file_sha256,
                form=form,
                source_kind=source_kind,
                include_text=include_text,
                spec_node_id=spec_node_id,
            )
            form_rows.append(summary)
            file_row["form_count"] = int(file_row["form_count"]) + 1
            declared_kind = str(summary.get("kind", "form") or "form")
            title = str(summary.get("title", "") or declared_kind)
            form_id = str(summary.get("id", ""))
            form_sha256 = str(summary.get("form_sha256", ""))
            explicit_id = str(summary.get("explicit_id", "") or "")
            x, y = _node_coordinates(form_id, 11)
            form_node = {
                "id": form_id,
                "kind": "form",
                "label": title,
                "title": title,
                "path": rel_path,
                "source_uri": str(summary.get("uri", "")),
                "x": x,
                "y": y,
                "confidence": 1.0,
                "provenance": {
                    "path": rel_path,
                    "source_uri": str(summary.get("uri", "")),
                    "sha256": form_sha256,
                    "span": dict(summary.get("span", {})),
                },
                "extension": {
                    "head": str(summary.get("head", "")),
                    "path": rel_path,
                    "title": title,
                    "explicit_id": explicit_id,
                    "form_sha256": form_sha256,
                    "source_kind": source_kind,
                    "tags": list(summary.get("tags", [])),
                    "refs": list(summary.get("refs", [])),
                    "canonical": str(summary.get("canonical", "") or ""),
                },
            }
            if include_text:
                form_node["extension"]["text"] = str(summary.get("text", "") or "")
            append_node(form_node, duplicate_error=rel_path)

            parent_id = spec_node_id or file_node_id
            _append_edge(
                edges,
                edge_seen,
                source=parent_id,
                target=form_id,
                kind="contains",
                weight=1.0,
                provenance={"path": rel_path, "sha256": form_sha256},
            )

            semantic_node_id = (
                explicit_id
                or f"lith:{declared_kind}:{_sha256_text(f'{rel_path}:{form_sha256}')[:24]}"
            )
            if declared_kind != "form":
                sx, sy = _node_coordinates(semantic_node_id, 13)
                semantic_node = {
                    "id": semantic_node_id,
                    "kind": declared_kind,
                    "label": title,
                    "title": title,
                    "path": rel_path,
                    "source_uri": str(summary.get("uri", "")),
                    "x": sx,
                    "y": sy,
                    "confidence": 1.0,
                    "provenance": {
                        "path": rel_path,
                        "source_uri": str(summary.get("uri", "")),
                        "sha256": form_sha256,
                        "span": dict(summary.get("span", {})),
                    },
                    "extension": {
                        "head": str(summary.get("head", "")),
                        "path": rel_path,
                        "title": title,
                        "explicit_id": explicit_id,
                        "form_id": form_id,
                        "form_sha256": form_sha256,
                        "source_kind": source_kind,
                        "tags": list(summary.get("tags", [])),
                        "refs": list(summary.get("refs", [])),
                        "canonical": str(summary.get("canonical", "") or ""),
                    },
                }
                if include_text:
                    semantic_node["extension"]["text"] = str(
                        summary.get("text", "") or ""
                    )
                if not append_node(semantic_node, duplicate_error=rel_path):
                    continue
                if explicit_id:
                    semantic_nodes_by_explicit_id[explicit_id] = semantic_node_id
                _append_edge(
                    edges,
                    edge_seen,
                    source=form_id,
                    target=semantic_node_id,
                    kind="declares",
                    weight=1.0,
                    provenance={"path": rel_path, "sha256": form_sha256},
                )
                _append_edge(
                    edges,
                    edge_seen,
                    source=semantic_node_id,
                    target=form_id,
                    kind="derived_from",
                    weight=1.0,
                    provenance={"path": rel_path, "sha256": form_sha256},
                )
                if declared_kind == "packet":
                    file_row["packet_count"] = int(file_row["packet_count"]) + 1
                    packet_rows.append(
                        {
                            "node_key": explicit_id or semantic_node_id,
                            "path": rel_path,
                            "id": explicit_id,
                            "v": _node_text_value(_child_value(form, "v")) or "",
                            "kind": _node_text_value(_child_value(form, "kind"))
                            or ":packet",
                            "title": title,
                            "tags": list(summary.get("tags", [])),
                            "routing": {
                                "target": _node_text_value(
                                    _child_value(
                                        _child_form(form, "routing") or {}, "target"
                                    )
                                ),
                                "handler": _node_text_value(
                                    _child_value(
                                        _child_form(form, "routing") or {}, "handler"
                                    )
                                ),
                                "mode": _node_text_value(
                                    _child_value(
                                        _child_form(form, "routing") or {}, "mode"
                                    )
                                ),
                            },
                        }
                    )
                elif declared_kind == "contract":
                    file_row["contract_count"] = int(file_row["contract_count"]) + 1
                    contract_rows.append(
                        {
                            "node_key": explicit_id or semantic_node_id,
                            "path": rel_path,
                            "id": explicit_id,
                            "v": _node_text_value(_child_value(form, "v")) or "",
                            "kind": ":contract",
                            "title": title,
                            "tags": list(summary.get("tags", [])) or [":contract"],
                            "routing": {"target": None, "handler": None, "mode": None},
                        }
                    )
                elif declared_kind in {"fact", "observation", "question"}:
                    file_row["fact_count"] = int(file_row["fact_count"]) + 1
                    fact_rows.append(
                        {
                            "id": explicit_id or semantic_node_id,
                            "kind": declared_kind,
                            "title": title,
                            "path": rel_path,
                            "tags": list(summary.get("tags", [])),
                        }
                    )

                for tag in summary.get("tags", []):
                    tag_id = ensure_tag_node(str(tag))
                    _append_edge(
                        edges,
                        edge_seen,
                        source=semantic_node_id,
                        target=tag_id,
                        kind="tagged",
                        weight=0.9,
                        provenance={"path": rel_path, "sha256": form_sha256},
                    )
            else:
                for tag in summary.get("tags", []):
                    tag_id = ensure_tag_node(str(tag))
                    _append_edge(
                        edges,
                        edge_seen,
                        source=form_id,
                        target=tag_id,
                        kind="tagged",
                        weight=0.9,
                        provenance={"path": rel_path, "sha256": form_sha256},
                    )

            for ref_value in summary.get("refs", []):
                pending_refs.append(
                    {
                        "source": semantic_node_id
                        if declared_kind != "form"
                        else form_id,
                        "source_kind": declared_kind,
                        "path": rel_path,
                        "sha256": form_sha256,
                        "value": str(ref_value),
                    }
                )

        file_rows.append(file_row)

    for ref in pending_refs:
        value = str(ref.get("value", "") or "").strip()
        if not value:
            continue
        source_id = str(ref.get("source", "") or "")
        source_kind = str(ref.get("source_kind", "") or "")
        path_value = str(ref.get("path", "") or "")
        ref_sha256 = str(ref.get("sha256", "") or "")
        edge_kind = "references"
        target_id = ""
        if value.startswith("http://") or value.startswith("https://"):
            target_id = f"lith:resource:{_sha256_text(value)[:24]}"
            if target_id not in node_ids_seen:
                x, y = _node_coordinates(target_id, 17)
                append_node(
                    {
                        "id": target_id,
                        "kind": "resource",
                        "label": value,
                        "title": value,
                        "source_uri": value,
                        "x": x,
                        "y": y,
                        "confidence": 0.8,
                        "provenance": {"source_uri": value},
                        "extension": {"url": value},
                    }
                )
        elif _looks_repo_path(value):
            candidate_path = _normalize_rel_path(value)
            candidate_abs = root / candidate_path
            try:
                candidate_is_file = candidate_abs.exists() and candidate_abs.is_file()
            except OSError as exc:
                errors.append(
                    {
                        "kind": "invalid_ref_path",
                        "path": path_value,
                        "value": value[:240],
                        "error": exc.__class__.__name__,
                    }
                )
                candidate_is_file = False
            if candidate_is_file:
                target_id = ensure_file_node(candidate_path)
                if source_kind in {"packet", "contract", "protocol", "spec"}:
                    edge_kind = "depends_on"
        elif _looks_explicit_ref(value):
            target_id = semantic_nodes_by_explicit_id.get(value, "")
        if not target_id:
            continue
        _append_edge(
            edges,
            edge_seen,
            source=source_id,
            target=target_id,
            kind=edge_kind,
            weight=0.72,
            provenance={"path": path_value, "sha256": ref_sha256},
        )

    role_counts: dict[str, int] = {}
    for node in nodes:
        role = str(node.get("kind", "unknown") or "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "unknown") or "unknown")
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1

    return {
        "record": "eta-mu.lith-nexus-index.v1",
        "schema_version": "lith.nexus.index.v1",
        "generated_at": _now_iso(),
        "repo_root": str(root),
        "config": {
            "roots": list(config.get("roots", [])),
            "include_ext": list(config.get("include_ext", [])),
            "ignore_glob": list(config.get("ignore_glob", [])),
            "mime": dict(config.get("mime", {})),
            "writes": dict(config.get("writes", {})),
            "index": dict(config.get("index", {})),
        },
        "files": file_rows,
        "forms": form_rows,
        "packets": packet_rows,
        "contracts": contract_rows,
        "facts": fact_rows,
        "nodes": nodes,
        "edges": edges,
        "errors": errors,
        "stats": {
            "file_count": len(file_rows),
            "form_count": len(form_rows),
            "packet_count": len(packet_rows),
            "contract_count": len(contract_rows),
            "fact_count": len(fact_rows),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "role_counts": role_counts,
            "edge_kind_counts": edge_kind_counts,
        },
        "joins": {
            "by_explicit_id": dict(sorted(semantic_nodes_by_explicit_id.items())),
            "by_path": dict(sorted(file_nodes_by_path.items())),
        },
    }


def merge_lith_nexus_into_logical_graph(
    *,
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
    file_path_to_node: dict[str, str],
    tag_token_to_logical: dict[str, str],
    lith_nexus: dict[str, Any] | None,
    normalize_path_for_file_id: Any,
    safe_float: Any,
    clamp01: Any,
) -> None:
    nexus_payload = lith_nexus if isinstance(lith_nexus, dict) else {}
    lith_nodes_raw = nexus_payload.get("nodes", [])
    if not isinstance(lith_nodes_raw, list):
        lith_nodes_raw = []
    lith_edges_raw = nexus_payload.get("edges", [])
    if not isinstance(lith_edges_raw, list):
        lith_edges_raw = []

    logical_node_ids: set[str] = {
        str(node.get("id", "")).strip()
        for node in graph_nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }
    logical_edge_keys: set[tuple[str, str, str]] = {
        (
            str(edge.get("source", "")).strip(),
            str(edge.get("target", "")).strip(),
            str(edge.get("kind", "")).strip().lower(),
        )
        for edge in graph_edges
        if isinstance(edge, dict)
        and str(edge.get("source", "")).strip()
        and str(edge.get("target", "")).strip()
    }
    lith_node_id_map: dict[str, str] = {}

    for lith_node in lith_nodes_raw:
        if not isinstance(lith_node, dict):
            continue
        source_node_id = str(lith_node.get("id", "")).strip()
        if not source_node_id:
            continue
        node_kind = str(lith_node.get("kind", "")).strip().lower()
        provenance_payload = (
            lith_node.get("provenance", {})
            if isinstance(lith_node.get("provenance", {}), dict)
            else {}
        )
        normalized_path = normalize_path_for_file_id(
            str(lith_node.get("path") or provenance_payload.get("path") or "")
        )
        if (
            node_kind == "file"
            and normalized_path
            and normalized_path in file_path_to_node
        ):
            lith_node_id_map[source_node_id] = file_path_to_node[normalized_path]
            continue
        if node_kind == "tag":
            extension_payload = (
                lith_node.get("extension", {})
                if isinstance(lith_node.get("extension", {}), dict)
                else {}
            )
            raw_tag = str(
                extension_payload.get("tag") or lith_node.get("label", "")
            ).strip()
            normalized_tag = re.sub(r"\s+", "_", raw_tag.lower())
            normalized_tag = re.sub(r"[^a-z0-9_]+", "", normalized_tag).strip("_")
            if normalized_tag and normalized_tag in tag_token_to_logical:
                lith_node_id_map[source_node_id] = tag_token_to_logical[normalized_tag]
                continue

        target_node_id = source_node_id
        if target_node_id in logical_node_ids:
            target_node_id = (
                "logical:lith:"
                + hashlib.sha256(target_node_id.encode("utf-8")).hexdigest()[:24]
            )
        if target_node_id in logical_node_ids:
            continue

        merged_node = dict(lith_node)
        merged_node["id"] = target_node_id
        merged_node["kind"] = node_kind or str(merged_node.get("kind", "logical"))
        merged_node["label"] = str(
            merged_node.get("label") or merged_node.get("title") or target_node_id
        )
        merged_node["x"] = round(clamp01(safe_float(merged_node.get("x", 0.5), 0.5)), 4)
        merged_node["y"] = round(clamp01(safe_float(merged_node.get("y", 0.5), 0.5)), 4)
        merged_node["confidence"] = round(
            clamp01(safe_float(merged_node.get("confidence", 1.0), 1.0)), 4
        )
        merged_provenance = dict(provenance_payload)
        merged_provenance.setdefault("origin_graph", "lith_nexus")
        if normalized_path:
            merged_provenance.setdefault("path", normalized_path)
        merged_node["provenance"] = merged_provenance
        graph_nodes.append(merged_node)
        logical_node_ids.add(target_node_id)
        lith_node_id_map[source_node_id] = target_node_id

    for lith_edge in lith_edges_raw:
        if not isinstance(lith_edge, dict):
            continue
        source_id = lith_node_id_map.get(
            str(lith_edge.get("source", "")).strip(),
            str(lith_edge.get("source", "")).strip(),
        )
        target_id = lith_node_id_map.get(
            str(lith_edge.get("target", "")).strip(),
            str(lith_edge.get("target", "")).strip(),
        )
        kind = str(lith_edge.get("kind", "relates")).strip().lower() or "relates"
        if not source_id or not target_id or source_id == target_id:
            continue
        if source_id not in logical_node_ids or target_id not in logical_node_ids:
            continue
        edge_key = (source_id, target_id, kind)
        if edge_key in logical_edge_keys:
            continue
        logical_edge_keys.add(edge_key)
        edge_id = str(lith_edge.get("id", "")).strip()
        if not edge_id:
            edge_id = (
                "logical:lith-edge:"
                + hashlib.sha256(
                    f"{source_id}|{target_id}|{kind}".encode("utf-8")
                ).hexdigest()[:20]
            )
        merged_edge = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "kind": kind,
            "weight": round(clamp01(safe_float(lith_edge.get("weight", 0.7), 0.7)), 4),
        }
        if isinstance(lith_edge.get("provenance"), dict):
            merged_edge["provenance"] = dict(lith_edge.get("provenance", {}))
        graph_edges.append(merged_edge)


def build_promptdb_snapshot_from_lith_index(index: dict[str, Any]) -> dict[str, Any]:
    files = [row for row in index.get("files", []) if isinstance(row, dict)]
    packets = [row for row in index.get("packets", []) if isinstance(row, dict)]
    contracts = [row for row in index.get("contracts", []) if isinstance(row, dict)]
    errors = [row for row in index.get("errors", []) if isinstance(row, dict)]
    promptdb_files = [
        row
        for row in files
        if str(row.get("path", "")).startswith(".opencode/promptdb/")
    ]
    promptdb_packets = [
        row
        for row in packets
        if str(row.get("path", "")).startswith(".opencode/promptdb/")
    ]
    promptdb_contracts = [
        row
        for row in contracts
        if str(row.get("path", "")).startswith(".opencode/promptdb/")
    ]
    promptdb_errors = [
        row
        for row in errors
        if str(row.get("path", "")).startswith(".opencode/promptdb/")
    ]
    promptdb_root = Path(str(index.get("repo_root", ""))) / ".opencode" / "promptdb"
    return {
        "root": str(promptdb_root) if promptdb_root.exists() else "",
        "packet_count": len(promptdb_packets),
        "contract_count": len(promptdb_contracts),
        "file_count": len(promptdb_files),
        "packets": promptdb_packets,
        "contracts": promptdb_contracts,
        "errors": promptdb_errors,
    }

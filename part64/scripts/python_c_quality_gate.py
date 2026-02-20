#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import lizard
except ModuleNotFoundError as exc:
    raise SystemExit(
        "lizard is required for complexity checks. Install with: python -m pip install lizard"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Budget:
    cognitive_complexity: int
    cyclomatic_complexity: int
    file_lines: int


@dataclass(frozen=True)
class Issue:
    severity: str
    language: str
    metric: str
    actual: int
    limit: int
    location: str


BUDGETS: dict[str, dict[str, Budget]] = {
    "python": {
        "warning": Budget(
            cognitive_complexity=45,
            cyclomatic_complexity=80,
            file_lines=2500,
        ),
        "error": Budget(
            cognitive_complexity=150,
            cyclomatic_complexity=380,
            file_lines=9000,
        ),
    },
    "c": {
        "warning": Budget(
            cognitive_complexity=4,
            cyclomatic_complexity=40,
            file_lines=900,
        ),
        "error": Budget(
            cognitive_complexity=10,
            cyclomatic_complexity=120,
            file_lines=2000,
        ),
    },
}


def _collect_files() -> tuple[list[Path], list[Path]]:
    python_roots = [ROOT / "part64" / "code", ROOT / "part64" / "scripts"]
    c_roots = [ROOT / "part64" / "code"]

    py_files: list[Path] = []
    for base in python_roots:
        py_files.extend(sorted(base.rglob("*.py")))

    c_files: list[Path] = []
    for base in c_roots:
        c_files.extend(sorted(base.rglob("*.c")))

    return py_files, c_files


def _load_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _severity_for(actual: int, warning: int, error: int, mode: str) -> str | None:
    if mode == "warn":
        if actual > warning:
            return "warning"
        return None
    if actual > error:
        return "error"
    if actual > warning:
        return "warning"
    return None


def _append_metric_issue(
    issues: list[Issue],
    *,
    mode: str,
    language: str,
    metric: str,
    actual: int,
    location: str,
) -> None:
    budgets = BUDGETS[language]
    warning_limit = getattr(budgets["warning"], metric)
    error_limit = getattr(budgets["error"], metric)
    severity = _severity_for(actual, warning_limit, error_limit, mode)
    if severity is None:
        return
    limit = error_limit if severity == "error" else warning_limit
    issues.append(
        Issue(
            severity=severity,
            language=language,
            metric=metric,
            actual=actual,
            limit=limit,
            location=location,
        )
    )


def _python_syntax_issues(files: list[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in files:
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            ast.parse(source)
        except SyntaxError as error:
            location = f"{path.relative_to(ROOT)}:{error.lineno or 1}"
            issues.append(
                Issue(
                    severity="error",
                    language="python",
                    metric="syntax",
                    actual=1,
                    limit=0,
                    location=f"{location} {error.msg}",
                )
            )
    return issues


def _c_syntax_issues(files: list[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in files:
        completed = subprocess.run(
            [
                "cc",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-fsyntax-only",
                str(path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        warning_lines = [
            line.strip()
            for line in completed.stderr.splitlines()
            if ": warning:" in line
        ]
        for warning_line in warning_lines:
            normalized_warning = warning_line.replace(f"{ROOT}/", "")
            issues.append(
                Issue(
                    severity="warning",
                    language="c",
                    metric="compiler_warning",
                    actual=1,
                    limit=0,
                    location=f"{path.relative_to(ROOT)} {normalized_warning}",
                )
            )
        if completed.returncode == 0:
            continue
        first_line = (
            completed.stderr.strip().splitlines()[0]
            if completed.stderr.strip()
            else "syntax check failed"
        )
        issues.append(
            Issue(
                severity="error",
                language="c",
                metric="syntax",
                actual=completed.returncode,
                limit=0,
                location=f"{path.relative_to(ROOT)} {first_line}",
            )
        )
    return issues


def _complexity_issues(files: list[Path], language: str, mode: str) -> list[Issue]:
    issues: list[Issue] = []
    for path in files:
        file_lines = _load_line_count(path)
        _append_metric_issue(
            issues,
            mode=mode,
            language=language,
            metric="file_lines",
            actual=file_lines,
            location=str(path.relative_to(ROOT)),
        )

        extensions = lizard.get_extensions(["NS"])
        result = list(lizard.analyze([str(path)], exts=extensions))
        if not result:
            continue
        file_analysis = result[0]

        for function in file_analysis.function_list:
            cyclomatic = int(function.cyclomatic_complexity)
            cognitive = int(getattr(function, "max_nested_structures", 0))
            location = f"{path.relative_to(ROOT)}:{function.start_line} {function.name}"
            _append_metric_issue(
                issues,
                mode=mode,
                language=language,
                metric="cyclomatic_complexity",
                actual=cyclomatic,
                location=location,
            )
            _append_metric_issue(
                issues,
                mode=mode,
                language=language,
                metric="cognitive_complexity",
                actual=cognitive,
                location=location,
            )
    return issues


def _render_issue(issue: Issue) -> str:
    if issue.metric == "syntax":
        return f"[{issue.severity}] {issue.language} syntax: {issue.location}"
    return (
        f"[{issue.severity}] {issue.language} {issue.metric}: "
        f"{issue.actual} > {issue.limit} at {issue.location}"
    )


def _print_budget_summary() -> None:
    print("Python/C quality budgets")
    for language, budget_set in BUDGETS.items():
        warn = budget_set["warning"]
        err = budget_set["error"]
        print(
            f"- {language}: warning(cognitive>{warn.cognitive_complexity}, cyclomatic>{warn.cyclomatic_complexity}, lines>{warn.file_lines}) "
            f"error(cognitive>{err.cognitive_complexity}, cyclomatic>{err.cyclomatic_complexity}, lines>{err.file_lines})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python/C quality gate")
    parser.add_argument(
        "--mode",
        choices=["warn", "strict"],
        default="warn",
        help="warn prints warning-tier violations; strict also enforces error-tier violations",
    )
    args = parser.parse_args()

    py_files, c_files = _collect_files()

    issues: list[Issue] = []
    issues.extend(_python_syntax_issues(py_files))
    issues.extend(_c_syntax_issues(c_files))
    issues.extend(_complexity_issues(py_files, "python", args.mode))
    issues.extend(_complexity_issues(c_files, "c", args.mode))

    _print_budget_summary()
    if not issues:
        print("No violations found.")
        return 0

    issues.sort(
        key=lambda issue: (
            0 if issue.severity == "error" else 1,
            issue.language,
            issue.metric,
            issue.location,
        )
    )
    for issue in issues:
        print(_render_issue(issue))

    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    error_count = sum(1 for issue in issues if issue.severity == "error")
    print(f"Total: {error_count} errors, {warning_count} warnings")

    if error_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

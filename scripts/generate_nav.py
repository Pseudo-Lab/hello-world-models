"""
frontmatter의 domain과 year를 기반으로 mkdocs.yml의 nav를 자동 생성하는 스크립트.

사용법: python scripts/generate_nav.py

mkdocs.yml에는 PyYAML의 safe_load가 읽지 못하는 태그(!!python/name:...)가 있으므로,
파일 전체를 다시 쓰지 않고 nav 블록만 찾아 교체한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "docs" / "review"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

DOMAIN_LABELS = {
    "overview": "Overview",
    "model-based-rl": "Model-Based RL",
    "sequence-based-world-models": "Sequence-Based World Models",
    "predictive-world-models": "Predictive World Models",
    "generative-world-models": "Generative World Models",
}

DOMAIN_ORDER = list(DOMAIN_LABELS.keys())


def parse_frontmatter(filepath: Path) -> dict | None:
    text = filepath.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    return yaml.safe_load(match.group(1))


def generate_nav() -> list:
    docs_by_domain: dict[str, list] = {d: [] for d in DOMAIN_ORDER}

    for md_file in sorted(REVIEW_DIR.glob("*.md")):
        meta = parse_frontmatter(md_file)
        if not meta:
            continue

        domain = meta.get("domain", "")
        title = meta.get("title", md_file.stem)
        year = meta.get("year", 9999)
        rel_path = f"review/{md_file.name}"

        if domain in docs_by_domain:
            docs_by_domain[domain].append((year, title, rel_path))

    nav = [{"Home": "index.md"}]

    for domain_key in DOMAIN_ORDER:
        entries = docs_by_domain[domain_key]
        if not entries:
            continue
        entries.sort(key=lambda x: x[0])
        section = []
        for year, title, path in entries:
            section.append({f"{title} ({year})": path})
        nav.append({DOMAIN_LABELS[domain_key]: section})

    return nav


def update_mkdocs_yml(nav: list) -> None:
    text = MKDOCS_YML.read_text(encoding="utf-8")

    # 최상위 nav: 키부터 다음 최상위 키(또는 파일 끝) 직전까지가 nav 블록이다
    match = re.search(r"^nav:[^\n]*\n(?:(?:[ \t-].*)?\n)*", text, re.MULTILINE)
    if not match:
        raise SystemExit("mkdocs.yml에서 최상위 nav: 블록을 찾지 못했습니다.")

    nav_yaml = yaml.dump(
        {"nav": nav}, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    MKDOCS_YML.write_text(text[: match.start()] + nav_yaml + text[match.end() :], encoding="utf-8")


if __name__ == "__main__":
    nav = generate_nav()
    update_mkdocs_yml(nav)
    print("nav updated successfully.")

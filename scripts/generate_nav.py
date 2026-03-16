"""
frontmatter의 domain과 year를 기반으로 mkdocs.yml의 nav를 자동 생성하는 스크립트.

사용법: python scripts/generate_nav.py
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "docs" / "review"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

DOMAIN_LABELS = {
    "latent-world-models": "Latent World Models",
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
    with open(MKDOCS_YML, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["nav"] = nav

    with open(MKDOCS_YML, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    nav = generate_nav()
    update_mkdocs_yml(nav)
    print("nav updated successfully.")

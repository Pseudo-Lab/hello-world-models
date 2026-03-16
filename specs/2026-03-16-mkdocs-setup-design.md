# MkDocs Material 스터디 사이트 설계

## 개요
- **사이트 제목:** Hello, World! — World Models Study
- **도구:** MkDocs Material + mkdocs-awesome-pages-plugin
- **배포:** GitHub Actions → GitHub Pages (main push 시 자동)
- **nav 자동화:** Python 스크립트가 frontmatter 기반으로 도메인별 그룹 + 연도순 정렬하여 `mkdocs.yml` nav 갱신

## 카테고리
1. **Latent World Models** — World Models, PlaNet, Dreamer, DreamerV2, MuZero, DreamerV3
2. **Sequence-Based World Models** — Trajectory Transformer, Decision Transformer
3. **Predictive World Models** — CPC, V-JEPA, V-JEPA 2
4. **Generative World Models** — Genie, Cosmos

## 디렉토리 구조
```
hello-world-models/
├── mkdocs.yml
├── requirements.txt
├── scripts/
│   └── generate_nav.py
├── docs/
│   ├── index.md
│   └── review/
│       ├── world-models.md
│       ├── planet.md
│       ├── dreamer.md
│       ├── dreamer-v2.md
│       ├── muzero.md
│       ├── dreamer-v3.md
│       ├── trajectory-transformer.md
│       ├── decision-transformer.md
│       ├── CPC.md
│       ├── v-jepa.md
│       ├── v-jepa-2.md
│       ├── genie.md
│       └── cosmos.md
├── .github/
│   └── workflows/
│       └── deploy.yml
└── .gitignore
```

## 문서 템플릿
```markdown
---
title: "논문 제목"
year: 2020
venue: NeurIPS
domain: latent-world-models
---

# 논문 제목

```{admonition} Information
- **Paper:** [링크](링크)
- **Presenter:** 발표자
- **Last updated:** 날짜
```

## 내용
```

## 자동화
- `scripts/generate_nav.py`: frontmatter의 `domain`별 그룹 + `year`순 정렬 → `mkdocs.yml` nav 갱신
- `.github/workflows/deploy.yml`: main push → mkdocs build → gh-pages 배포

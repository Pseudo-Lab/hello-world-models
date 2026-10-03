# MkDocs 사용 가이드

리뷰 사이트(https://pseudo-lab.github.io/hello-world-models/)에 논문 리뷰 페이지를 추가하는 방법입니다.

## 1. 환경 설정

```bash
git clone https://github.com/Pseudo-Lab/hello-world-models.git
cd hello-world-models
pip install -r requirements.txt
```

## 2. 로컬에서 사이트 확인

```bash
mkdocs serve
```
http://127.0.0.1:8000 에서 확인 가능. 파일 수정 시 자동 반영.

## 3. 리뷰 문서 작성

`docs/review/` 폴더에 마크다운 파일을 추가합니다.

### 3-1. Frontmatter 작성

파일 최상단에 아래 형식의 frontmatter를 작성합니다. nav 자동 생성에 사용됩니다.

```yaml
---
title: "논문 제목"
year: 2024
venue: NeurIPS
domain: latent-world-models
---
```

`domain`은 아래 중 하나를 사용합니다:
- `latent-world-models`
- `sequence-based-world-models`
- `predictive-world-models`
- `generative-world-models`

### 3-2. Information 섹션 작성

```markdown
!!! info "Information"
    - **Title:** 논문 제목
    - **Venue:** NeurIPS 2024
    - **Paper:** [arXiv](https://arxiv.org/abs/xxxx.xxxxx)
    - **Project:** [Project Page](링크)  ← 있을 때만
    - **Code:** [GitHub](링크)           ← 있을 때만
    - **Presenter:** 발표자
    - **Last updated:** 2026-03-16
```

### 3-3. 이미지 삽입

이미지는 리뷰 파일마다 폴더를 따로 만들어 저장합니다. 폴더 이름은 리뷰 파일 이름과 같게 합니다.

```
docs/
├── review/
│   └── dreamer-v2.md
└── assets/
    └── dreamer-v2/
        ├── 01_overall_architecture.png
        └── 02_rssm_components.png
```

- 폴더 이름: 리뷰 파일 이름에서 `.md`를 뺀 이름 (`dreamer-v2.md` → `docs/assets/dreamer-v2/`)
- 파일 이름: 공백 없이 영문 소문자, 숫자, `_`, `-`만 사용합니다. 본문에 나오는 순서대로 `01_`, `02_` 같은 번호를 붙이면 관리하기 쉽습니다.
- `Pasted image 2026....png`처럼 붙여 넣기로 생긴 이름은 내용이 드러나는 이름으로 바꿔서 올립니다.

리뷰 문서에서는 아래 형식으로 삽입합니다.

**기본 이미지:**
```markdown
![이미지 설명](../assets/dreamer-v2/01_overall_architecture.png)
```

**크기 조절:**
```markdown
![이미지 설명](../assets/dreamer-v2/01_overall_architecture.png){ width="600" }
```

**캡션 포함 이미지:**
```markdown
<figure markdown="span">
  ![이미지 설명](../assets/dreamer-v2/01_overall_architecture.png){ width="600" }
  <figcaption>Figure 1. 캡션 내용 (source: 출처)</figcaption>
</figure>
```

### 3-4. 수식 작성

인라인 수식:
```markdown
$E = mc^2$
```

블록 수식:
```markdown
$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$
```

### 3-5. 접기/펼치기 (Details)

```markdown
??? note "클릭하여 펼치기"
    숨겨진 내용이 여기에 표시됩니다.
```

기본 펼침 상태:
```markdown
???+ note "클릭하여 접기"
    기본으로 펼쳐진 상태입니다.
```

## 4. Nav 자동 생성

문서 추가 후 아래 스크립트를 실행하면 `mkdocs.yml`의 nav가 도메인별 > 연도순으로 자동 갱신됩니다.

```bash
python scripts/generate_nav.py
```

## 5. 배포

`main` 브랜치에 push하면 GitHub Actions가 자동으로 사이트를 빌드하고 GitHub Pages에 배포합니다.

```bash
git add .
git commit -m "Add: 논문 리뷰 추가"
git push
```

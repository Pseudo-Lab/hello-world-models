# Marp 발표자료 가이드

[Marp](https://marp.app/)는 마크다운으로 슬라이드를 만드는 도구입니다.

## 1. 설치

**VS Code 확장 (추천)**

1. VS Code 좌측 Extensions 탭 (`Ctrl+Shift+X` / `Cmd+Shift+X`)
2. "Marp for VS Code" 검색 → 설치

**CLI (선택)**

```bash
# npm
npm install -g @marp-team/marp-cli

# Homebrew (macOS)
brew install marp-cli
```

## 2. 기본 사용법

마크다운 파일을 만들고, 맨 위에 frontmatter를 작성합니다. `---`로 슬라이드를 구분합니다.

```markdown
---
marp: true
paginate: true
---

# 첫 번째 슬라이드

내용을 작성합니다.

---

# 두 번째 슬라이드

- 항목 1
- 항목 2
```

## 3. 미리보기

**VS Code**

`marp: true`가 포함된 `.md` 파일을 열고 우측 상단의 미리보기 버튼 (아이콘: 돋보기가 있는 문서)을 클릭하면 슬라이드 미리보기가 표시됩니다.

**CLI**

```bash
marp slides.md --preview
```

## 4. 기본 테마

frontmatter에서 `theme`을 지정합니다. Marp에 내장된 테마는 3가지입니다.

```yaml
---
marp: true
theme: default   # 또는 gaia, uncover
---
```

| 테마 | 특징 |
|------|------|
| `default` | 깔끔한 기본 스타일 |
| `gaia` | 따뜻한 톤, `lead` 클래스로 중앙 정렬 가능 |
| `uncover` | 모던하고 미니멀, 내용 중앙 정렬 |

## 5. 주요 문법

### Directive (슬라이드 설정)

frontmatter에서 전체 슬라이드에 적용하거나, 슬라이드 내에서 HTML 주석으로 개별 적용합니다.

```markdown
---
marp: true
paginate: true          # 전체: 페이지 번호
header: "Hello, World Models!"  # 전체: 상단 헤더
footer: "2026"          # 전체: 하단 푸터
---

# 슬라이드 1

---

<!-- _backgroundColor: #f0f0f0 -->
<!-- _paginate: false -->
# 이 슬라이드만 배경색 변경, 페이지 번호 숨김
```

`_` 접두사가 붙으면 해당 슬라이드에만 적용됩니다.

### 이미지 크기 조절

```markdown
![width:500px](image.png)
![w:300 h:200](image.png)
```

### 배경 이미지

```markdown
![bg](background.jpg)          <!-- 전체 배경 -->
![bg left:40%](side-image.jpg) <!-- 좌측 40%에 이미지, 우측에 내용 -->
![bg right](side-image.jpg)    <!-- 우측에 이미지 -->
```

### 수식 (KaTeX)

frontmatter에 `math: katex`를 추가합니다.

```markdown
인라인: $E = mc^2$

블록:
$$
\mathcal{L} = \mathbb{E}[\| \epsilon - \epsilon_\theta \|^2]
$$
```

## 6. PDF 추출

### VS Code

1. Marp 마크다운 파일을 연 상태에서 Command Palette 열기 (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. `Marp: Export Slide Deck` 입력 → 선택
3. 저장 대화상자에서 파일 형식을 **PDF**로 선택 → 저장

PPTX, HTML, PNG, JPEG 형식도 같은 방법으로 추출할 수 있습니다.

### CLI

```bash
# PDF
marp slides.md --pdf

# PPTX
marp slides.md --pptx

# 파일명 지정
marp slides.md -o output.pdf
```

---

## 7. 커스텀 테마 (review 테마)

이 폴더에 스터디 공용 테마와 샘플이 준비되어 있습니다.

- `review.css` — 스터디 공용 슬라이드 테마
- `sample.md` — 기본 구조가 잡힌 샘플 발표자료

### 커스텀 테마 적용하기

Marp는 커스텀 CSS를 자동으로 찾지 못하므로, 아래 방법 중 하나로 등록해야 합니다.

**방법 A) VS Code 설정 (추천)**

프로젝트 루트의 `.vscode/settings.json.example`을 복사해서 사용합니다.

```bash
cp .vscode/settings.json.example .vscode/settings.json
```

또는 직접 `.vscode/settings.json`을 만들고 아래 내용을 추가합니다.

```json
{
  "markdown.marp.themes": [
    "./templates/marp/review.css"
  ]
}
```

등록 후 VS Code를 재시작하면 어디서든 `theme: review`로 사용할 수 있습니다.

**방법 B) CLI `--theme` 옵션**

```bash
marp --theme templates/marp/review.css slides.md --preview
marp --theme templates/marp/review.css slides.md --pdf
```

> **참고:** 커스텀 CSS는 같은 폴더에 있어도 자동으로 인식되지 않습니다. 반드시 위 방법 중 하나로 경로를 등록해야 합니다.

### Frontmatter 예시

```yaml
---
marp: true
title: "논문 제목 — 논문 리뷰"
paginate: true
theme: review
math: katex
header: "Hello, World Models!"
---
```

### review 테마 기능 요약

| 기능 | 사용법 |
|------|--------|
| 글씨 크기 조절 | `<!-- _class: small -->`, `xsmall`, `xxsmall` |
| 2단 레이아웃 | `<!-- _class: two-col -->` + `<div class="columns">` |
| 이미지 크기 | `![w50 center](image.png)` (w10~w100) |
| 리스트 이어쓰기 | `<!-- _class: l2 -->` (2레벨부터 시작) |

자세한 사용 예시는 `sample.md`를 참고하세요.

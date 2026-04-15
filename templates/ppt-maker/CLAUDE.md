# CLAUDE.md — ppt-maker

논문 리뷰 발표 PPT를 **end-to-end**로 제작하는 템플릿입니다.
HTML/CSS로 슬라이드 삽화를 디자인하고, Playwright로 3×DPI PNG를 캡처한 뒤,
**python-pptx로 PPT 파일을 직접 빌드**합니다.

> **⚠️ Marp CLI는 사용하지 않습니다.**
> 마크다운 입력 **문법**은 Marp 스타일을 그대로 파싱(익숙함을 위해)하지만,
> 실제 빌드는 `build_pptx.py` (python-pptx)가 처리합니다. 수식은 matplotlib가
> PNG로 렌더해서 삽입하고, 폰트/색상은 스크립트 상단 상수로 제어합니다.
> Marp CLI 기반이 아니라 python-pptx 기반이라, 한글 폰트·수식·레이아웃을
> 자유롭게 제어할 수 있는 것이 장점입니다.

이 파일은 **Claude Code 자동 로드 지침서**입니다.
사용자가 "논문 X 리뷰 PPT 만들어줘" / "슬라이드 만들어줘" / "PPT 빌드해줘" 같은
요청을 하면 아래 원칙을 따라 작업합니다.

---

## 전체 워크플로우

```
  [1] 논문 요약 · 슬라이드 구조 설계       (대화로)
          ↓
  [2] 슬라이드별 HTML 작성               (visual/*.html)
          ↓
  [3] Playwright 캡처                    (visual/v3/*.png)
          ↓
  [4] Marp 문법 마크다운 작성             (lecture.md)
          ↓
  [5] python build_pptx.py lecture.md   (lecture.pptx 생성)
          ↓
  [6] PowerPoint에서 미세 조정           (사용자)
```

---

## Step 1 — 슬라이드 구조 설계

사용자가 논문 또는 섹션을 지정하면:
- 핵심 주장 / 주요 방법 / 실험 / 의의 로 분해
- 각 파트를 **1~6 슬라이드** 분량으로 쪼개 제안
- 사용자 승인 후 다음 단계 진행

---

## Step 2 — HTML 슬라이드 삽화 (`visual/`)

> 📖 **상세 규칙은 [`DESIGN_GUIDE.md`](DESIGN_GUIDE.md) 참조** — 918줄의 실전 디자인
> 가이드 (컬러 팔레트, 폰트 위계, 컴포넌트 CSS, 레이아웃 비율, 차트 가이드, 버전 관리 등).
> Claude는 슬라이드 작성 전 이 문서를 반드시 함께 읽을 것.

### 디자인 시스템

`visual/_common_style.css` 참조. **반드시 따라야 할 규칙**:

| 항목 | 값 |
|---|---|
| 캔버스 (`.slide-visual`) | 1280px 너비 고정 |
| 본문 최소 폰트 | **20px** (PPT 축소 시 가독성) |
| 헤더 | 24-30px |
| 수식 폰트 | `Times New Roman` serif |
| 변수·코드 폰트 | `JetBrains Mono` |
| 한국어 줄바꿈 | `word-break: keep-all` (이미 CSS 포함) |
| 하이픈 단어 | `<span style="white-space:nowrap">straight-through</span>` |

### 색상 팔레트 (의미 구분)

| 변수 | 색 | 쓰임새 예 |
|---|---|---|
| `--wine` | `#8a1538` | 주 강조 / Actor |
| `--blue` | `#1a5276` | 2차 강조 / Critic |
| `--teal` | `#0e6655` | Environment / Transition |
| `--orange` | `#d35400` | Reward / Loss |
| `--purple` | `#7a4e9c` | RSSM / h |
| `--green` | `#1e8449` | 긍정 / 환경 |

### 예시 자산 (복제해서 수정할 것 — 처음부터 만들지 말 것)

**독립 HTML 예시 6종** — 실제 발표 PPT에서 채택된 슬라이드 기반:

| 패턴 | 파일 | 언제 쓰나 |
|---|---|---|
| 6-카드 그리드 | `visual/example_1_grid_components.html` | 모델 컴포넌트·구성 요소 나열 |
| 논문 figure + 범례 | `visual/example_2_figure_with_legend.html` | 논문 다이어그램 + 한글 설명 |
| 2열 역할 비교 | `visual/example_3_two_column_roles.html` | 두 단계/역할 나란히 비교 |
| 간단 개념 비교 | `visual/example_4_simple_concept.html` | 2-3 항목 최소 비교 |
| 단계별 흐름 (비유) | `visual/example_5_staged_flow.html` | Stage 1 → Stage 2 비유 카드 |
| 수식 + 안정화 카드 | `visual/example_6_formula.html` | 수식 정의 + 부가 설명 |

**PNG-only 참고 자료** — `visual/v3/reference/` (HTML 소스는 없지만 패턴 참고용):

| 파일 | 패턴 |
|---|---|
| `ref_compare_two_cards.png` | 큰 2열 비교 카드 (좌우 풍부한 내용) |
| `ref_three_step_flow.png` | 3단계 flow (STEP 1→2→3) |
| `ref_analogy_cards.png` | 비유·유추 카드 레이아웃 |
| `ref_infographic_numbers.png` | 숫자 강조 인포그래픽 |
| `ref_mechanism_diagram.png` | 메커니즘·수식 박스 다이어그램 |

참고 PNG는 "이런 패턴도 가능하다" 보여주는 용도. 필요하면 Claude에게 PNG 보여주고
"이런 스타일로 HTML 만들어줘" 요청하면 재현해줌.

### HTML 작성 규칙

- 각 HTML은 **한 슬라이드 한 장** 기준
- `<link rel="stylesheet" href="_common_style.css">` 로 공통 스타일 참조
- 루트 요소는 `<div class="slide-visual">` — Playwright가 이 요소만 캡처해 여백 자동 트림
- 슬라이드 내부는 카드(`.card`), 그리드(`display: grid`), SVG 등으로 조합

---

## Step 3 — Playwright 캡처

### 환경 준비 (없으면)

```bash
pip install playwright
playwright install chromium
```

### 캡처 스크립트 (필요시 즉석 생성)

```python
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).parent
JOBS = [
    ("visual/my_slide.html", "visual/v3/my_slide.png"),
]

async def main():
    out_dir = HERE / "visual" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1400, "height": 1200},
            device_scale_factor=3,   # 필수: 3×DPI
        )
        for src, dst in JOBS:
            await page.goto((HERE / src).as_uri(), wait_until="networkidle")
            await page.wait_for_timeout(2000)    # 폰트 로드 대기
            sv = await page.query_selector(".slide-visual")
            await sv.screenshot(path=str(HERE / dst))
            print(f"  ✓ {dst}")
        await browser.close()

asyncio.run(main())
```

**핵심 설정**:
- `device_scale_factor=3` — PPT 확대 시에도 또렷
- `wait_until="networkidle"` + `wait_for_timeout(2000)` — Google Fonts 로드 대기
- `.slide-visual` 요소만 캡처 — 여백 자동 트림

### 품질 체크 (매 캡처 후 자동 점검)

- [ ] 본문 폰트 ≥ 20px
- [ ] 카드 밖으로 텍스트 넘침 없음
- [ ] 한국어 단어·조사 중간 끊김 없음
- [ ] 하이픈 영단어 한 줄 유지
- [ ] 색상 팔레트 일관
- [ ] 슬라이드 간 시각적 연속성

문제 발견 시 HTML 수정 → 재캡처 반복.

---

## Step 4 — 마크다운 입력 (`lecture.md`)

> Marp CLI 없이 `build_pptx.py`가 직접 파싱합니다. Marp 스타일 **문법**만 재활용.

### 기본 문법

```markdown
---
marp: true
theme: review
paginate: true
math: katex
---

# 제목

본문 또는 이미지

![w80 center](visual/v3/slide_1.png)

---

# 다음 슬라이드

- bullet
- **볼드**

$$ L(\xi) = \mathbb{E}[(v - V^\lambda)^2] $$
```

- 슬라이드 구분: `---`
- 이미지: `![w<너비> center](path)` — 너비·정렬 지정
- 수식: `$$...$$` — build_pptx.py가 matplotlib로 PNG 렌더

`sample_lecture.md`를 참조하여 구조 파악.

---

## Step 5 — PPT 빌드 (`build_pptx.py` · python-pptx 기반)

### 실행

```bash
python build_pptx.py lecture.md
# → lecture.pptx 생성
```

### 스크립트가 해주는 것

- 마크다운 파싱 (Marp 문법 인식)
- Freesentation 한글 폰트 적용 (OOXML `latin/ea/cs` 모두 설정)
- `$$...$$` 수식을 matplotlib로 PNG 렌더 후 삽입
- Wine 팔레트 (`--wine #8a1538` 등) 기본 적용
- Bullet (OOXML 정식)·표·이미지·인용·코드 블록 처리

### 커스터마이즈

`build_pptx.py` 상단 상수만 조정:
- `FONT_MAIN = "Freesentation"` → 다른 폰트
- `WINE`, `DARK`, `GRAY` 등 `RGBColor` 값
- 슬라이드 크기는 `Presentation()` 인스턴스 설정

---

## 버전 관리 원칙

- PNG는 **덮어쓰지 않고** `_v1.png`, `_v2.png`로 보존
- HTML 대폭 수정 시 `_v2.html`로 분기
- `visual/v3/` 가 최종 제출용 PNG 폴더

---

## Claude Code 사용자를 위한 팁

사용자는 이런 식으로 요청하면 됩니다:

- "DreamerV3 논문의 Behavior Learning 섹션을 슬라이드 6장으로 만들어줘"
- "방금 만든 슬라이드 중 3번 것 카드가 밖으로 넘쳤어. 세로 키워줘"
- "lecture.md로 PPT 빌드해줘"
- "슬라이드 품질 체크해줘"

Claude는 이 CLAUDE.md를 자동 로드하므로 **워크플로우·디자인 규칙·품질 체크를
스스로 적용**합니다. 사용자에게는 중간 결과(PNG 미리보기, 최종 PPT 경로)만
전달하면 됩니다.

---

## 폴더 구조 요약

```
ppt-maker/
├── CLAUDE.md                      # 이 파일
├── visual/                        # HTML 삽화 + PNG
│   ├── _common_style.css          # 디자인 시스템
│   ├── example_compare.html       # 2열 비교 예시
│   ├── example_timeline.html      # SVG 체인 예시
│   ├── example_formula.html       # 수식 카드 예시
│   └── v3/                        # 최종 PNG 폴더
│       └── example_*.png
├── build_pptx.py                  # Markdown → PPT 빌더
├── sample_lecture.md              # 마크다운 입력 예시
└── requirements.txt               # Python 의존성
```

---

## 빠른 시작 (사용자가 이 템플릿을 처음 쓸 때)

1. 이 폴더 통째로 본인 프로젝트에 복사: `cp -r ppt-maker my_paper`
2. `cd my_paper` 후 Claude Code 실행
3. Claude에게 요청: *"XX 논문 리뷰 PPT 만들어줘"*
4. 결과 확인, 필요하면 미세 조정 요청 반복

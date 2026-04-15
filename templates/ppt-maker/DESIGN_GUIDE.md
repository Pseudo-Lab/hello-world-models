# 비주얼 에셋 제작 상세 가이드 (Design Guide)

> **이 문서는 ppt-maker 템플릿의 상세 레퍼런스**입니다.
> 짧은 요약은 `CLAUDE.md`를, 세세한 CSS 규칙·컴포넌트 패턴·폰트 위계·버전 관리
> 노하우는 이 문서를 참고하세요.
>
> 예시 HTML은 `visual/example_compare.html`, `example_timeline.html`,
> `example_formula.html` 세 가지 패턴을 제공합니다. 새 슬라이드를 만들 때 이들 중
> 가까운 패턴을 **복제해서 수정**하세요. 팔레트·폰트 규칙은 `visual/_common_style.css`에
> 고정돼 있어 그대로 상속하면 됩니다.

---

## 0. 핵심 원칙

### 기존 데이터 절대 수정 금지

> **어떤 상황에서도 기존에 생성된 HTML 파일이나 캡처 폴더를 덮어쓰지 않는다.**

- 수정 요청이 오면 → **새 버전의 HTML**을 생성 (예: `slide_v1.html` → `slide_v2.html`)
- 캡처도 → **새 버전의 PNG**로 저장 (예: `slide_v1.png` → `slide_v2.png`)
- 기존 버전은 항상 원본 그대로 보존 — 롤백·비교 가능
- PPT도 마찬가지: 새 파일명(날짜+버전)으로 생성, 기존 `.pptx`는 건드리지 않음

**왜?** 이전 버전을 참고하거나 롤백할 수 있어야 하고, 작업 이력이 추적 가능해야 함.

### 참조 레퍼런스

새 슬라이드를 작성할 때는 **이 템플릿의 예시 HTML을 먼저 읽고 스타일/구조를 따를 것.**
처음부터 만들지 말고 가까운 패턴을 복제해서 수정.

**독립 HTML 예시 6종** (실제 발표에서 채택된 슬라이드 기반):

| 파일 | 패턴 | 언제 쓰나 |
|---|---|---|
| `visual/example_1_grid_components.html` | 6-카드 그리드 | 모델 컴포넌트·구성 요소 |
| `visual/example_2_figure_with_legend.html` | 논문 figure + 범례 | 논문 다이어그램 + 한글 설명 |
| `visual/example_3_two_column_roles.html` | 2열 역할 비교 | 두 단계/역할 나란히 |
| `visual/example_4_simple_concept.html` | 간단 개념 비교 | 최소 비교 · 핵심 짚기 |
| `visual/example_5_staged_flow.html` | 단계별 흐름 (비유) | Stage 흐름 + 비유 |
| `visual/example_6_formula.html` | 수식 + 안정화 카드 | 수식 정의 + 부가 설명 |
| `visual/_common_style.css` | 공통 팔레트·폰트·베이스 | 예시 중 일부에서 `<link>`로 참조 |

완성된 PNG 미리보기: `visual/v3/example_*.png`.

**PNG-only 참고 자료** (`visual/v3/ref_*.png`) — HTML 소스 없음, 시각적 패턴 참고용:

| 파일 | 패턴 |
|---|---|
| `ref_compare_two_cards.png` | 큰 2열 비교 카드 |
| `ref_three_step_flow.png` | 3단계 flow (STEP 1→2→3) |
| `ref_analogy_cards.png` | 비유·유추 카드 |
| `ref_infographic_numbers.png` | 숫자 강조 인포그래픽 |
| `ref_mechanism_diagram.png` | 메커니즘·수식 박스 |

필요하면 Claude에게 참고 PNG 보여주고 "이 스타일로 HTML 만들어줘" 요청.

---

## 1. 디자인 시스템

### 컬러 팔레트

`visual/_common_style.css`의 `:root` 변수로 정의됨. 역할별로 색을 분리해 사용.

| 변수 | HEX | 용도 |
|------|------|------|
| `--wine` | `#8a1538` | 주 강조, 중요 수식, 핵심 개념 |
| `--wine-bg` | `#fdf2f5` | wine 카드 배경 |
| `--blue` | `#1a5276` | 2차 강조, 모델/구조 |
| `--blue-bg` | `#eaf2f8` | blue 카드 배경 |
| `--teal` | `#0e6655` | 3차, 환경·변수 |
| `--teal-bg` | `#e8f6f3` | teal 카드 배경 |
| `--orange` | `#d35400` | Loss, 경고, 보상 |
| `--orange-bg` | `#fdf4e3` | orange 카드 배경 |
| `--purple` | `#7a4e9c` | 정규화·보조 |
| `--purple-bg` | `#f1ebf7` | purple 카드 배경 |
| `--green` | `#1e8449` | 긍정·정답·최종 |
| `--green-bg` | `#e8f5ec` | green 카드 배경 |
| `--dark` | `#111` | 본문 |
| `--gray` | `#555` | 보조 텍스트 |
| `--gray-light` | `#999` | 구분선·비활성 |

**규칙**:
- 역할별 색 배정을 일관되게 유지 (한 논문 리뷰 안에서 같은 개념은 같은 색)
- 장점·긍정 = 틸 또는 그린, 단점·경고 = 와인 또는 레드
- 화살표(흐름 표시): `var(--gray)` + 볼드
- **프로젝트에 맞게 수정 가능** — 색 의미만 일관되면 팔레트 자체는 바꿔도 됨

### 텍스트 스타일

- **최소 폰트 크기: 20px** — 모든 텍스트는 캡처 후 20px 이상이어야 함 (PPT 축소 시 가독성)
- **⚠️ font-weight 누락 금지** — 모든 CSS 클래스와 인라인 스타일에 `font-weight`를 반드시 명시. 누락하면 기본값 400 Regular로 렌더되어 톤이 어긋남.

### font-weight 체계

| 무게 | 용도 |
|------|------|
| 400 (Regular) | 거의 사용 안 함 |
| 500 (Medium) | 부연 설명 |
| 600 (SemiBold) | 본문 보조·라벨·캡션 |
| 700 (Bold) | 카드 제목·강조 |
| 800 (ExtraBold) | 핵심 수식·대제목 |

### 텍스트 크기 위계 (슬라이드 기준)

| 요소 | 크기 | 무게 |
|------|------|------|
| 대제목 (슬라이드 상단) | 28-32px | 800 |
| 중제목 (카드 헤더) | 24-28px | 700-800 |
| 본문 | 20-24px | 600 |
| 부연 설명 | 18-20px | 500-600 |
| 수식 (`Times New Roman` serif) | 22-28px | 500 |
| 코드·변수 (`JetBrains Mono`) | 18-22px | 700 |

### 폰트

공통 스타일에서 import되는 Google Fonts:

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
```

- **기본**: `Inter` (한글 포함 전체)
- **변수·코드**: `JetBrains Mono`
- **수식**: `Times New Roman` serif (수식 박스 내부만)

한국어 브랜드 폰트(Freesentation, Pretendard 등)를 쓰고 싶으면 `_common_style.css`에
`@font-face`를 추가하고 `body { font-family: '...', Inter, sans-serif }` 로 override.

### 한국어 줄바꿈 규칙

```css
body, .slide-visual, .slide-visual * {
  word-break: keep-all;
  overflow-wrap: break-word;
  hyphens: manual;
}
```

`_common_style.css`에 이미 포함. 그래도 끊기는 경우:

- **하이픈 단어** (예: `straight-through`, `state-of-the-art`) →
  `<span style="white-space:nowrap;">straight-through</span>` 로 감쌈
- **의미 단위가 억지로 분리** → `<br>`로 명시적 줄바꿈 지정
- **괄호 부연** (예: "(30개 차원)") → 앞 텍스트에 `<br>` 붙이고 괄호를 새 줄로

---

## 2. HTML 비주얼 제작

### 캔버스 기본

- `.slide-visual` 너비 **1280px 고정**
- 배경 흰색, 패딩 `26px 30px` (가변)
- 내부는 `display: flex; flex-direction: column; gap: 16px` 세로 스택

### 레이아웃 원칙

- **비율에 맞추되 억지로 채우지 않기** — 여백도 디자인 요소
- 카드가 너무 넓으면 2열로, 너무 좁으면 1열로
- 수직 정렬: 중요 요소는 상단, 부연은 하단
- 여러 카드를 배치할 때 **균등 그리드** 사용 (`grid-template-columns: 1fr 1fr`)

### 콘텐츠 비율

- **수식 + 개념 설명** 패턴 → 수식 50-60% · 설명 40-50%
- **비교 카드 2열** → 좌 50% · 우 50%
- **타임라인 + 설명** → SVG 65-70% · 텍스트 30-35%
- **3카드 그리드** → 각 33%

### 카드 공통 스타일

```css
.card {
  padding: 14-18px;
  border-radius: 10-14px;
  border: 2-3px solid;   /* 색 있는 카드 */
  background: var(--*-bg); /* 같은 계열 연한 배경 */
  display: flex;
  flex-direction: column;
  gap: 8-12px;
}
```

---

## 3. 공통 컴포넌트 CSS 패턴

아래 패턴은 `visual/example_*.html`에 실제 구현되어 있음. 필요 시 해당 파일을 열어
HTML+CSS를 그대로 복사해 시작.

### (1) 2열 비교 카드 — `example_compare.html`

개념 A vs B를 나란히 놓는 레이아웃. 좌우 카드 대조, 중앙에 `→` 또는 `vs` 구분자.

```html
<div class="compare">
  <div class="world A">
    <div class="world-hdr"><span class="ic">🌍</span><div class="txt">A</div></div>
    <div class="comp-list">
      <div class="comp-item"><span class="lbl">S</span><div class="body">...</div></div>
      ...
    </div>
  </div>
  <div class="arrow-mid">→</div>
  <div class="world B"> ... </div>
</div>
```

### (2) SVG 체인·타임라인 — `example_timeline.html`

시퀀스나 rollout 표현. `viewBox` 좌표계로 노드·화살표 정밀 배치.

```html
<svg viewBox="0 0 1280 340" style="width:100%; height:340px;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="7" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L8,3.5 L0,7 z" fill="#7a4e9c"/>
    </marker>
  </defs>
  <circle cx="315" cy="185" r="36" fill="#f1ebf7" stroke="#7a4e9c" stroke-width="3"/>
  <text x="315" y="193" text-anchor="middle" font-size="24" font-weight="800">...</text>
  <line x1="351" y1="185" x2="433" y2="185" stroke="#7a4e9c" stroke-width="2.5" marker-end="url(#arr)"/>
</svg>
```

**SVG 팁**:
- viewBox 좌표계 기준 폰트 20-28px가 적절 (캡처 3×DPI 로 충분히 또렷)
- 화살표는 `<marker>` 재사용
- 노드 원: `circle r=30-40`, 박스: `rect width=80-120 height=50-60`
- 텍스트: `text-anchor="middle"`, `font-weight="800"`로 또렷하게

### (3) 수식 박스 — `example_formula.html`

수식 정의 + 안정화/해설 카드 2열 레이아웃.

```html
<div class="fbox lam">
  <div class="lbl">🎯 target V<sup>λ</sup> — λ-return</div>
  <div class="eq" style="font-family:'Times New Roman', serif;">
    V<sup>λ</sup><sub>t</sub> = r̂<sub>t</sub> + γ̂<sub>t</sub> · [ (1−λ)·v(ẑ<sub>t+1</sub>) + λ·V<sup>λ</sup><sub>t+1</sub> ]
  </div>
</div>
```

### (4) Proscons (장단점) 카드

```html
<div class="proscons">
  <div class="pros">
    <div class="hdr">✅ 장점</div>
    <ul><li>...</li></ul>
  </div>
  <div class="cons">
    <div class="hdr">⚠️ 단점</div>
    <ul><li>...</li></ul>
  </div>
</div>
```

색: 장점 = `var(--teal)` 또는 `var(--green)`, 단점 = `var(--wine)` 또는 `var(--orange)`.

### (5) 비유/메타포 카드

비유를 전달할 때는 큰 이모지 + 카드. 왼쪽은 실제 개념, 오른쪽은 비유.

```html
<div class="analogy">
  <div class="real">실제 개념</div>
  <div class="mid">↔</div>
  <div class="metaphor">🎮 게임으로 비유하면...</div>
</div>
```

### (6) 배지·칩·라벨

```html
<span class="badge" style="
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 14px; font-weight: 800;
  background: var(--wine); color: white;
">V2</span>

<span class="chip" style="
  padding: 5px 12px; border-radius: 8px;
  border: 2px solid var(--wine); color: var(--wine);
  font-family: 'JetBrains Mono', monospace; font-weight: 700;
">ρ = 1</span>
```

---

## 4. 차트·다이어그램 (matplotlib)

데이터 시각화나 복잡한 다이어그램은 HTML보다 matplotlib이 편할 때가 있음. 규칙:

- `dpi=200`, `bbox_inches='tight'`, `facecolor='#FFFFFF'`
- `figsize` 가이드: 단일 차트 `(9.6, 4.4)`, 복합 `(10, 5.5)` 이상
- 색상: CSS 팔레트와 동일한 HEX 사용
- 폰트: Inter / sans-serif, 기본 크기 14-18pt
- 모든 축 라벨·제목·범례에 `fontweight='bold'` 권장

```python
import matplotlib.pyplot as plt

WINE = "#8a1538"
BLUE = "#1a5276"
DARK = "#1F1F1F"
GRAY = "#555555"

fig, ax = plt.subplots(figsize=(9.6, 4.4))
# ... plotting ...
fig.savefig("visual/v3/my_chart.png", dpi=200, bbox_inches='tight', facecolor='#FFFFFF')
```

---

## 5. 파일 명명 규칙

### HTML 슬라이드

```
{주제 약어}_{순번}_{설명}.html
```

예시:
- `bl_1a_imagination_mdp.html` — "Behavior Learning 1a: Imagination MDP"
- `bl_1b_rollout.html`
- `bl_2a_critic_nstep.html`

버전 분기 시:
- `bl_2b_critic_loss.html` → `bl_2b_critic_loss_v2.html` (대폭 수정)
- 소수 CSS 수정은 동일 파일에서

### HTML 템플릿 구조

```html
<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>슬라이드 이름</title>
<link rel="stylesheet" href="_common_style.css">
<style>
  /* 이 슬라이드만의 개별 스타일 */
</style></head><body>

<h1 class="slide-title">슬라이드 제목 (옵션)</h1>
<div class="slide-visual">
  <!-- 콘텐츠 -->
</div>

</body></html>
```

### 캡처 PNG

HTML과 같은 이름 + `.png`. `visual/v3/` 폴더에 저장.
- `bl_1a_imagination_mdp.html` → `visual/v3/bl_1a_imagination_mdp.png`
- 버전 분기: `visual/v3/bl_2b_critic_loss_v1.png`, `_v2.png`

---

## 6. 캡처 (Playwright)

### 도구

- **Playwright chromium** — Chrome 기반 headless 브라우저
- 설치: `pip install playwright && playwright install chromium`

### 캡처 설정

필수:
- `device_scale_factor=3` — 3×DPI 고해상도
- `viewport = {"width": 1400, "height": 1200}` (여유 있게)
- `wait_until="networkidle"` + `wait_for_timeout(2000)` — 폰트 로드 대기
- `page.query_selector(".slide-visual")` 로 해당 요소만 캡처 → 여백 자동 트림

### 캡처 스크립트 (일괄)

```python
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).parent

JOBS = [
    ("visual/slide_1.html", "visual/v3/slide_1.png"),
    ("visual/slide_2.html", "visual/v3/slide_2.png"),
]

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1400, "height": 1200},
            device_scale_factor=3,
        )
        for src, dst in JOBS:
            (HERE / dst).parent.mkdir(parents=True, exist_ok=True)
            await page.goto((HERE / src).as_uri(), wait_until="networkidle")
            await page.wait_for_timeout(2000)
            sv = await page.query_selector(".slide-visual")
            await sv.screenshot(path=str(HERE / dst))
            print(f"  ✓ {dst}")
        await browser.close()

asyncio.run(main())
```

---

## 7. PPT 빌드 (`build_pptx.py`)

### 입력 마크다운 문법

Marp 스타일 문법을 그대로 쓰지만, **Marp CLI는 사용하지 않음** — python-pptx가 파싱.

- 슬라이드 구분: `---`
- 제목: `# 제목`, `## 소제목`
- bullet: `- 항목`, `  - 하위 항목` (2-space 들여쓰기)
- 이미지: `![w80 center](path)` — 너비·정렬 지정
- 수식: `$$ ... $$` (블록), `$ ... $` (인라인) — matplotlib가 PNG로 렌더
- 표: 파이프 문법
- 코드: ```` ```python ... ``` ````

### 빌드

```bash
python build_pptx.py lecture.md
# → lecture.pptx 생성
```

### 스크립트 내부가 해주는 것

- 마크다운 파싱 (Marp 스타일)
- 한글 폰트 적용 (OOXML `latin/ea/cs` 태그 모두 설정 — 기본 `Freesentation`)
- `$$...$$` 수식을 matplotlib로 PNG 렌더 후 삽입
- 팔레트 (`WINE`, `DARK`, `GRAY` 등) 적용
- Bullet (OOXML 정식) · 표 · 이미지 · 인용 · 코드 블록 처리

### 커스터마이즈

`build_pptx.py` 상단 상수로 조정:

```python
FONT_MAIN = "Freesentation"   # 다른 폰트로 교체 가능 (Pretendard 등)
WINE = RGBColor(0x8A, 0x15, 0x38)
DARK = RGBColor(0x11, 0x11, 0x11)
GRAY = RGBColor(0x55, 0x55, 0x55)
```

---

## 8. 버전 관리

### 규칙

- **PNG**: 덮어쓰지 않고 `_v1.png`, `_v2.png`로 보존
- **HTML**: 대폭 수정 시 `_v2.html`로 분기, 사소한 CSS 조정은 동일 파일
- **PPT**: 새 빌드는 타임스탬프 또는 버전 추가 (`lecture_260418_v2.pptx`)

### 분기 시점

**새 버전(`_v2`)으로 분기할 때:**
- 레이아웃 구조 자체를 바꿀 때
- 슬라이드가 PPT에 이미 삽입된 후 디자인 변경
- 비교·롤백이 필요한 실험적 수정

**동일 파일 수정해도 되는 경우:**
- 오타 수정
- 색상·폰트 크기 미세 조정
- 텍스트 한두 줄 변경

---

## 9. 품질 체크리스트

슬라이드 완성 전 확인:

- [ ] 본문 폰트 **≥ 20px**
- [ ] 모든 CSS 클래스에 `font-weight` 명시
- [ ] 카드 밖으로 텍스트 넘침 없음
- [ ] 한국어 단어·조사 중간 끊김 없음
- [ ] 하이픈 영단어 (`straight-through` 등) 한 줄 유지
- [ ] 색상 팔레트 일관 (한 논문 리뷰 안에서 같은 개념은 같은 색)
- [ ] 수식 폰트 `Times New Roman` serif 통일
- [ ] 슬라이드 간 시각적 연속성 (레이아웃·색·폰트)
- [ ] 3×DPI로 캡처 (`device_scale_factor=3`)
- [ ] `visual/v3/` 폴더에 저장 (덮어쓰기 없이)

문제 발견 시 HTML 수정 → 재캡처 반복.

---

## 10. 정리 한 장

```
논문 읽기 + 요약
    ↓
슬라이드 구조 설계 (대화)
    ↓
example_*.html 복제 → 수정 → 새 HTML
    ↓
Playwright 캡처 → visual/v3/*.png
    ↓
lecture.md 작성 (Marp 문법)
    ↓
python build_pptx.py lecture.md → lecture.pptx
    ↓
PowerPoint 미세 조정 → _v2.pptx
```

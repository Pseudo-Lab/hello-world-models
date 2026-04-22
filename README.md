# Hello, World! — World Models Study

본 프로젝트는 [가짜연구소(PseudoLab)](https://www.linkedin.com/company/pseudolab/)에서 진행하는 [World Models 스터디](https://pseudo-lab.com/projects/ee36e216-4876-49c4-899e-636e9b7e20a8)입니다. World Models 관련 논문을 리뷰하고 스터디 자료를 정리하는 공간입니다.

참여 방법: 매주 수요일 오후 9시, 가짜연구소 Discord ROOM-AK로 입장!

## Contributors
- 유정화 [Jeonghwa Yoo] | [Github](https://github.com/jeongHwarr) | [Linkedin](https://www.linkedin.com/in/jeonghwa-yoo-8403a716b/) |
- 김주연 [Juyeon Kim] | [Github](https://github.com/JYeonKim) | [Linkedin](https://www.linkedin.com/in/ju-yeon-kim) |
- 장보아 [Boa Jang] | [Github](https://github.com/Jang-Boa) | [Linkedin](https://www.linkedin.com/in/boa-jang-93a72918a) |
- 이재호 [Jaeho Lee] | [Github](https://github.com/ORE24) | [Linkedin](https://www.linkedin.com/in/jae-ho-lee-82b418303) |
- 김현수
- 유지형
- 정민지
  
## Reviewed Papers
| idx | Date | Presenter | Paper | Resources |
| :--: | :--: | :--: | :--: | :--: |
| 0 | 2026-03-18 | 유정화 | Introduction & Overview| [YouTube ](https://youtu.be/F_Vd7-3JUc8)/ Review |
| 1 | 2026-03-25 | 장보아 | [World Models](https://arxiv.org/abs/1803.10122) (NeurIPS 2018) | [YouTube](https://youtu.be/hzyZPWn2goI) / [Review](https://pseudo-lab.github.io/hello-world-models/review/world-models/) |
| 2 | 2026-04-01 | 이재호 | [PlaNet](https://arxiv.org/abs/1811.04551) (ICML 2019) | [YouTube](https://www.youtube.com/watch?v=qVEcdMkFZc4) / [Review](https://pseudo-lab.github.io/hello-world-models/review/planet/) |
| 3 | 2026-04-08 | 김주연 | [Dreamer](https://arxiv.org/abs/1912.01603) (ICLR 2020) | [YouTube](https://youtu.be/e_WeefKrxB8) / [Review](https://pseudo-lab.github.io/hello-world-models/review/dreamer/) |
| 4 | 2026-04-15 | 유정화 | [DreamerV2](https://arxiv.org/abs/2010.02193) (ICLR 2021) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/dreamer-v2/) |
| 5 | 2026-04-22 | 이재호 | [MuZero](https://arxiv.org/abs/1911.08265) (Nature 2020) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/muzero/) |
| 6 | 2026-05-06 | 김주연 | [DreamerV3](https://arxiv.org/abs/2301.04104) (ICLR 2023) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/dreamer-v3/) |
| 7 | 2026-05-13 | | [Trajectory Transformer](https://arxiv.org/abs/2106.02039) (NeurIPS 2021) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/trajectory-transformer/) |
| 8 | 2026-05-20 | 김현수 | [Decision Transformer](https://arxiv.org/abs/2106.01345) (NeurIPS 2021) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/decision-transformer/) |
| 9 | 2026-06-03 | | [CPC](https://arxiv.org/abs/1807.03748) (arXiv 2018) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/CPC/) |
| 10 | 2026-06-10 | 유지형 | [V-JEPA](https://arxiv.org/abs/2404.08471) (arXiv 2023) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/v-jepa/) |
| 11 | 2026-06-17 | 김주연 | [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) (arXiv 2025) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/v-jepa-2/) |
| 12 | 2026-06-24 | 정민지 | [Genie](https://arxiv.org/abs/2402.15391) (ICML 2024) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/genie/) |
| 13 | 2026-07-01 | 유지형 | [Cosmos](https://arxiv.org/abs/2501.03575) (arXiv 2025) | YouTube / [Review](https://pseudo-lab.github.io/hello-world-models/review/cosmos/) |

## MkDocs 사용 가이드

### 1. 환경 설정

```bash
git clone https://github.com/Pseudo-Lab/hello-world-models.git
cd hello-world-models
pip install -r requirements.txt
```

### 2. 로컬에서 사이트 확인

```bash
mkdocs serve
```
http://127.0.0.1:8000 에서 확인 가능. 파일 수정 시 자동 반영.

### 3. 리뷰 문서 작성

`docs/review/` 폴더에 마크다운 파일을 추가합니다.

#### 3-1. Frontmatter 작성

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

#### 3-2. Information 섹션 작성

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

#### 3-3. 이미지 삽입

이미지는 `docs/assets/` 폴더에 저장하고 아래 형식으로 삽입합니다.

**기본 이미지:**
```markdown
![이미지 설명](../assets/이미지파일.png)
```

**크기 조절:**
```markdown
![이미지 설명](../assets/이미지파일.png){ width="600" }
```

**캡션 포함 이미지:**
```markdown
<figure markdown="span">
  ![이미지 설명](../assets/이미지파일.png){ width="600" }
  <figcaption>Figure 1. 캡션 내용 (source: 출처)</figcaption>
</figure>
```

#### 3-4. 수식 작성

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

#### 3-5. 접기/펼치기 (Details)

```markdown
??? note "클릭하여 펼치기"
    숨겨진 내용이 여기에 표시됩니다.
```

기본 펼침 상태:
```markdown
???+ note "클릭하여 접기"
    기본으로 펼쳐진 상태입니다.
```

### 4. Nav 자동 생성

문서 추가 후 아래 스크립트를 실행하면 `mkdocs.yml`의 nav가 도메인별 > 연도순으로 자동 갱신됩니다.

```bash
python scripts/generate_nav.py
```

### 5. 배포

`main` 브랜치에 push하면 GitHub Actions가 자동으로 사이트를 빌드하고 GitHub Pages에 배포합니다.

```bash
git add .
git commit -m "Add: 논문 리뷰 추가"
git push
```

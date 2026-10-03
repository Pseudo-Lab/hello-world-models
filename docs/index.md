# Hello, World! — World Models Study

World Models 관련 논문을 리뷰하고 스터디 자료를 정리하는 공간입니다.

## Categories

- **Model-Based RL** — 월드 모델 안에서 계획하거나 행동을 학습하는 강화학습 (World Models, PlaNet, Dreamer, MuZero, DreamerV2, DreamerV3)
- **Sequence-Based World Models** — 시퀀스 모델링 기반 접근 (Decision Transformer, Trajectory Transformer)
- **Predictive World Models** — 예측 기반 표현 학습 (V-JEPA, V-JEPA 2)
- **Generative World Models** — 생성 기반 월드 모델 (GAIA-1, Genie, Cosmos)

## How to Contribute

1. `docs/review/` 폴더에 논문 리뷰 마크다운 파일 추가
2. frontmatter에 `title`, `year`, `venue`, `domain` 작성
3. `python scripts/generate_nav.py` 실행하여 nav 자동 갱신
4. Push하면 GitHub Actions가 자동으로 사이트 빌드 및 배포

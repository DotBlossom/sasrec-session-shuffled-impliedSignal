
#  상품 추천 모델 개요 
> **LLM 기반 데이터 구조화 및 고도화된 SASRec을 활용한 2단계 추천 시스템 연구**

본 프로젝트는 SK 쉴더스 루키즈 최종 프로젝트에서 설계한 **AI 모델과 데이터 파이프라인의 범용성 및 성능을 검증**하기 위한 연구 프로젝트입니다. 대규모 패션 데이터셋(H&M)의 비정형 데이터를 LLM으로 구조화하고, 이를 2단계 개인화 추천 모델(Retrieval & Reranking)에 최적화하여 구현하였습니다.


> [!NOTE]  
> 아래에서 설명하는 모델은 초기 바닐라 모델(단순 seq rec)이며, 모델의 거시적인 구조 및 데이터 구조의 추론 효용성을 증명하기 위해, seq 모델 내부나 item tower 모델의 내부는 핵심 로직만 차용한 채 가장 간략한 모델 구조로 구현되었습니다.
---

##  프로젝트 개요 (Overview)
* **연구 목적**: 비정형 상품 설명의 의미론적 속성 추출 및 추천 파이프라인의 도메인 범용성/성능 검증
* **데이터셋**: H&M Personalized Fashion Recommendations (Item: 70k / Interaction: 15M)
* **핵심 성과**: 
    * seq 모델 베이스라인 대비 **Recall 평균 50~60% 향상**
    * 연산 최적화를 통해 **VRAM 사용량 80% 절감** (RTX 2070 Super 8GB 기준)
    
---

## 시스템 아키텍처 (Architecture)

1. **전처리 및 Item Tower**: 상품/유저 로그 정제 및 Pre-trained 모델을 통한 상품 벡터 추출 후 VectorDB 저장
2. **User Retrieval (1차 후보 추출)**: SASRec 기반 개인화 추천 및 비딥러닝 방식(인기 아이템/그래프 기반)을 혼합하여 300~500개 후보군 선정 (*비딥러닝 구현 예정*)
3. **Reranking (최종 순위 도출)**: 1차 후보군에 LLM 구조화 피처를 **Cross-feature**로 적용하여 최종 추천 순위 도출 (*구현 예정*)

---

##  Item Metadata Hybrid Structure
아이템 임베딩의 정교함을 위해 데이터를 성격에 따라 **STD(Standard)**와 **RE(Reinforced)** 두 가지 레이어로 구조화하였습니다.

### 1. STD (Standard Features) : 데이터의 앵커(Anchor)
* **정의**: 속성값의 도메인이 한정적인 정형 카테고리 정보입니다.
* **역할**: 임베딩 공간에서 아이템이 위치할 기본적인 좌표를 고정하는 앵커 역할을 수행합니다.
* **주요 필드**: `prod_name`, `product_type_name`, `section_name` 등

### 2. RE (Reinforced Features) : 데이터의 해상도(Resolution)
* **정의**: 비정형 텍스트를 LLM을 통해 9가지 의미 단위 필드로 슬라이싱/정제한 데이터입니다.
* **역할**: 단순 카테고리로 구분하기 어려운 유사 상품 간의 미세한 디테일을 보완하여 임베딩 해상도를 높입니다.
* **주요 필드**: `MAT`(소재), `CAT`(세부 분류), `DET`(디자인 디테일) 등 9종

> **성능 지표**: 앵커의 안정성(유사도 0.92)을 유지하면서 유사 상품 간 변별력 **105% 증가**, 카테고리 간 경계 **119% 명확화** 달성

---

##  핵심 기술 스펙 (Technical Specs)

### 🔹 LLM 기반 데이터 구조화 및 피처 확장
* **의미론적 속성 추출**: 비정형 텍스트를 패션 도메인 특화 9가지 필드로 구조화하여 Transformer 연산 효율 극대화
* **피처 활용도**: 고품질 피처를 Retrieval 단계를 넘어 Reranker의 학습 데이터로 연계하여 파이프라인 전반의 사용성 확대

### 🔹 SASRec 기반 시퀀스 추천 모델 고도화
* **대조 학습(Contrastive Learning) 최적화**: 세션 마지막 벡터들을 활용한 Loss 연산 구조 도입으로 아이템 공간 학습 효율 증대
* **다중 의도 파악**: 세션 내 랜덤 벡터 주입을 통해 유저의 복합적이고 우연한 니즈 반영
* **단순 암기 방지(Shuffling)**: 세션 내 아이템 순서를 무작위로 섞어 본질적인 연관성 학습 유도
* **맥락 분리 피처 주입**: 이전 세션과의 구매액 편차 등 유저의 상태 변화를 반영하는 데이터 주입

---

##  상세 수행 내역 (Implementation Details)

### 1. 전처리 및 고해상도 Item Tower
* **FastAPI 기반 파이프라인**: 원시 데이터를 복합 피처로 변환하는 모듈 설계 및 Airflow 연동 API 최적화
* **데이터 융합**: LLM 파싱 필드를 기본 카테고리(Anchor) 피처와 융합하여 정교한 임베딩 공간 구축
* **주요 코드 파일**:
    * `item_tower.py`: 아이템 메타데이터 필드 임베딩 및 TransformerEncoder 인코딩. SimCSE 구조 차용 및 Unsupervised 대조학습 구현
    * `tower_code/v3/FeatureProcessor_v3`: 유저-상품 시퀀스 및 피처 데이터 취합 및 캐싱
    * `load_aligned_pretrained_embeddings`: 정렬된 사전학습 벡터 생성 및 DataLoader 인스턴스화

### 2. Seq 모델 학습 최적화 및 경량화
* **모델 튜닝**: Time-decay 가중치 및 확률적 셔플링 적용으로 지표 대폭 개선 (합산 Recall@k 60% 향상)
* **학습 자원 절감**: 마지막 세션 예측 벡터와 랜덤 벡터(15%)만 손실 계산에 사용하여 **VRAM 80% 절감**
* **데이터 정제**: S-BERT 유사도 통계 분석을 통해 실제 대체재가 오답(False Negative)으로 처리되는 현상 방지
* **주요 코드 파일**:
    * `SASRecDataset_v3`: 동일 세션 아이템 간 확률적 셔플링 및 Unsupervised 학습용 데이터 리턴 구현
    * `SASRecUserTower_v3`: 데이터 분석 기반 성능 기여 피처 선별 및 유저 Seq 모델 구성
    * `inbatch_corrected_logq_loss_with_hard_neg_margin`: LogQ Correction 및 시간 감쇠가 적용된 대조 학습 손실 함수

---

##  주요 성과 및 최종 지표 (Results)
* **데이터 Adapting 전략 입증**: 정형(STD)과 비정형(RE) 데이터를 '의미론적 필드'로 융합하여 임베딩 성능 향상
* **범용적 도메인 이식성 확보**: 신규 데이터를 '카테고리+비정형' 구조로 환원 적용 가능한 파이프라인의 높은 범용성 증명
* **인프라 효율화**: 연산 최적화를 통해 로컬 환경(RTX 2070 Super 8GB)에서 대규모 데이터 학습 가능
* **최종 지표**:
    * Hard Negative Mining 적용 시 평균 5% 추가 성능 향상 확인
    * 최종 Seq-only 모델 Recall@20 / Recall@100 / Recall@500 지표 확보 완료

---
<br>
---

## 📊 Item Representation Evaluation Report

주제: 의미론적 필드(Semantic Field) 기반 텍스트 메타데이터 융합 전략의 유효성 검증

1. 평가 개요 (Overview)

본 평가는 추천 시스템의 Item Tower(Transformer + MLP 아키텍처)가 아이템의 메타데이터를 임베딩 공간에 얼마나 효과적으로 투영하는지 검증하기 위해 수행되었습니다.

동일한 사전 학습(Pre-trained) 체크포인트를 통제 변인으로 두고, **Case 1(STD: 카테고리 기반 기본 뼈대)**과 **Case 2(STD + RE: 도메인 최소 단위로 분할된 텍스트 메타데이터 추가)**의 임베딩 출력값을 비교하여 데이터 구조 설계의 기하학적 우수성을 수치적으로 증명했습니다.

2. 핵심 가설 검증 결과 (Hypothesis Testing)

가설 1. 앵커 정렬도 (Anchor Alignment & Structural Stability)

검증 지표: 동일 아이템의 Case 1과 Case 2 간 평균 코사인 유사도

측정 결과: 0.9255

분석: 자연어 기반의 방대한 텍스트 모달리티(RE)가 추가 주입되었음에도, 임베딩 벡터가 기존 카테고리(STD)가 형성한 기하학적 중심축(Anchor)에서 크게 벗어나지 않았습니다. 이는 Transformer 모델이 기존의 핵심 정체성을 파괴하지 않고(No Catastrophic Interference), 부가 텍스트를 정교한 수식어(Modifier)로만 융합해 내는 데 최적화되어 있음을 증명합니다.

가설 2. 공간 해상도 (Intra/Inter Cluster Variance)

검증 지표: 카테고리 기준 군집 내 분산(Intra) 및 군집 간 거리(Inter)

측정 결과:

군집 내 분산: 0.0446 → 0.0918 (약 105% 증가)

군집 간 거리: 0.0916 → 0.2014 (약 119% 증가)

분석: 텍스트 메타데이터의 주입으로 같은 카테고리 내 아이템 간의 미세한 차이(소재, 핏 등)가 벡터 공간에 반영되며 내부 해상도(Intra-variance)가 2배 이상 확장되었습니다. 동시에 군집 간 거리(Inter-variance)도 비례하여 확장(비율 약 0.45 ∼ 0.48 유지)됨으로써, 카테고리 간의 거시적 분류 경계가 무너지지 않고 오히려 더 명확해지는 이상적인 공간 구조를 달성했습니다.

가설 3. 정보 기여도 및 디스인탱글먼트 (Ablation Variance)

검증 지표: Case 2 전체 분산(0.1998) 대비 특정 RE 필드 제거 시의 분산 하락폭

측정 결과:

최고 기여: [DET] (디테일/외형) → 기여도 +0.0049


분석: 단일 텍스트 필드([DET])가 128차원 전체 공간 분산의 약 2.45%를 지배적으로 확장시켰습니다. 반면, 형태학적 본질과 무관한 색상([COL]) 차원은 임베딩 구조에 영향을 주지 못하도록 억제되었습니다. 이는 설계된 데이터 분할 전략이 모델로 하여금 패션 도메인의 구조적 중요도를 스스로 인지하고 노이즈를 분리(Disentanglement)하도록 유도했음을 보여줍니다.

3. 결론 (Conclusion)

분석 결과, 하나의 플랫(Flat)한 텍스트로 메타데이터를 주입하는 대신 의미론적 필드(Semantic Field) 단위로 쪼개어 STD 앵커와 결합한 데이터 구조 설계는 Transformer 인코더에 매우 특화된 방식임이 확인되었습니다. 이 아키텍처는 기본 아이템 속성의 안정성을 유지하면서도, 텍스트 모달리티를 상호 간섭 없이 융합하여 임베딩 공간의 표현력과 해상도를 극대화하는 데 성공했습니다.
<br>
<br>
<br>

# 2차 프로젝트: 모델 세부 구조 advanced 개발 및 과정
---




## Core Architecture & Implementation Details

제공되는 추천 모델(`SASRecUserTower_v4`)은 단순히 피처를 이어 붙이는(Concat) 방식을 탈피하여, 각 모달리티의 고유한 기하학적 특성을 끝까지 보존하는 정교한 아키텍처를 채택했습니다.

### 1. Decoupled Transformer Layer (비침습적 평행 인코딩)
* **문제 인식:** ID 임베딩과 텍스트 임베딩을 조기 융합(Early Fusion)할 경우 발생하는 노이즈 간섭 현상 방지.
* **구현 기법:** * ID 스트림과 Prompt(Semantic) 스트림이 각자의 Query, Key, Value를 독립적으로 투영.
  * 어텐션 맵(Attention Map) 계산 시에만 두 시그널을 `attn_id + alpha * attn_p` 형태로 결합(Learnable Alpha 적용).
  * 연산 결과는 다시 각자의 스트림으로 분리 주입(`Strictly Non-invasive`)되어 정보의 순수성을 레이어 끝까지 보존.

### 2. Element-wise 3-Way Stream Fusion Gate (차원 단위 독립 융합)
* **구현 기법:** Softmax 기반의 제로섬(Zero-sum) 융합 병목을 타파하기 위해 **Sigmoid 기반의 SENet 구조** 도입.
* **Adaptive Gating:** 128차원의 각 차원(Element)별로 독립적인 스위치(0~1)를 부여. ID가 가진 강력한 CF 시그널의 파이를 깎지 않고, Prompt의 유용한 미세 디테일만 부스터(Additive Fusion) 형태로 얹어주는 마이크로 컨트롤 달성.
* **Prior Bias Injection:** sigmoid 초기화 시 ID(60%), Prompt(30%), Static(30%) 개방률을 Logit 역산으로 세팅하여, 학습 초기 모달리티 붕괴 방지.

### 3. DCN-v2 기반 Static Feature Crossing
* 유저 프로필 및 컨텍스트 정보(Age, Channel, Recency 등)를 처리하기 위해 초경량 **Deep & Cross Network (DCN-V2)** 도입. 
* 2-Layer 교차 연산을 통해 이산형/연속형 피처 간의 고차원 상호작용(Feature Crossing)을 효과적으로 추출하여 융합 게이트에 전달.

---

## 🛡️ Advanced Contrastive Learning & HNM

단순한 In-batch Negative를 넘어, 모델의 변별력을 극대화하기 위해 **Semantic-Denoised Global HNM** 파이프라인을 구축했습니다.

* **S-BERT Semantic Shield (가짜 오답 마스킹):**
  * 대조 학습 시, S-bert 기반 유사도 통계를 구한 후, 분포에 기반하여 정답 아이템과 S-BERT 코사인 유사도가 0.90 ~0.99 이상인 아이템(사실상 동일한 대체재)을 사전에 판별.
  
* **User-Anchored HNM with Curriculum Learning:**
  * 현재 유저의 예측 벡터(User Embedding)를 기준으로 전체 아이템 공간(Global Space)을 스캔하여 Top-K Hard Negative 채굴.

* **LogQ Correction:** 아이템의 인기도 편향(Popularity Bias)을 수학적으로 상쇄.

---

## ⚙️ Optimization & Sequence Engineering

* **Time-Decay Weighting:** 구매 시점(`interaction_dates`)을 기준으로, 최근 상호작용에 더 높은 그래디언트 가중치(반감기 적용)를 부여하여 최신 트렌드 반영.
* **Session Boundary & Stochastic Sampling:** 전체 시퀀스가 아닌 '세션의 마지막 액션'과 '15%의 확률적 무작위 액션'에 대해서만 Loss를 연산. 추천 성능의 하락 없이 VRAM 사용량을 80% 절감하는 동시에 데이터 셔플링 효과(단순 암기 방지) 달성.

---




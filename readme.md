
---

# 🚀 LLM-Enhanced Semantic Sequential Recommendation System

본 프로젝트는 대규모 이커머스 패션 데이터(H&M, 70k+ Items, 15M+ Interactions)를 기반으로 구축된 **개인화 시퀀스 추천 파이프라인(Two-stage)**입니다. 

LLM을 통해 추출된 비정형 텍스트 속성(9가지 의미론적 필드)을 S-BERT로 벡터화하고, 이를 아이템의 협업 필터링(CF) 시그널과 결합하기 위해 **Decoupled 3-Way User Tower** 아키텍처를 직접 설계 및 구현했습니다.

## 📌 Project Overview
* **목적:** 범용적인 도메인 이식성을 갖춘 고성능 하이브리드 추천 모델 파이프라인 검증
* **핵심 성과:** * 텍스트(Prompt)와 ID 시그널의 오버스무딩(Over-smoothing) 문제를 해결한 평행 어텐션 구조 도입.
  * HNM(Hard Negative Mining) 과정에서 발생하는 공간 붕괴(Catastrophic Forgetting)를 방어하는 **Semantic Shield** 알고리즘 구현.


---

## 🏗️ Core Architecture & Implementation Details

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



📊 Item Representation Evaluation Report

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
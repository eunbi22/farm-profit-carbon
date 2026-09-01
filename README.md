# 🌾 Farm Profit & Carbon Simulator

### FarmMap 기반 논벼 필지의 탄소감축 잠재량과 농가 수익성을 함께 분석하는 데이터 기반 시뮬레이터

> **From FarmMap Parcels to Carbon & Profit Decisions**

Farm Profit & Carbon Simulator는 해남군 논벼 재배 필지를 대상으로  
**FarmMap 공간정보 + IPCC Tier 1 배출량 산정 + XGBoost 수확량 예측**을 결합하여,

**① 탄소감축 잠재량 → ② 탄소가치 → ③ 쌀 판매수익 → ④ 통합 수익성**

을 필지 단위로 분석하는 프로젝트입니다.

---

## ✦ Why?

농가의 저탄소 농업 전환은 단순히  
"탄소를 얼마나 줄일 수 있는가?"의 문제가 아닙니다.

실제 의사결정에서는 동시에 질문해야 합니다.

> **수확량은 어떻게 되는가?**  
> **탄소배출은 얼마나 줄어드는가?**  
> **탄소감축의 경제적 가치는 얼마인가?**  
> **결국 농가에게 더 유리한 선택인가?**

본 프로젝트는 이 문제를 **하나의 데이터 파이프라인으로 연결**하는 것을 목표로 합니다.

---

# 🎯 Core Idea

```text
              FarmMap
        Paddy Parcel Data
               │
               ▼
      ┌─────────────────┐
      │   Data Pipeline │
      └────────┬────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Yield Model       IPCC Tier 1
    XGBoost           CH₄ Model
       │                │
       ▼                ▼
   Rice Revenue     CO₂eq Reduction
       │                │
       └───────┬────────┘
               ▼
       Carbon Revenue
               │
               ▼
      ┌─────────────────┐
      │ Profit Simulator│
      └─────────────────┘
               │
               ▼
       Rice Only vs.
       Rice + Carbon

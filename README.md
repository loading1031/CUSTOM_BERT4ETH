## 🧠 BERT4ETH 계량 모델 – 신규 주소 학습 문제 개선

### 📌 개요
- **프로젝트명:** 머신러닝을 이용한 블록체인 이상탐지 
- **목표:** 기존 BERT4ETH 모델이 신규 주소 학습 시 재학습이 필요한 한계를 극복하고, 신규 주소에 대한 추론 성능 향상
- **기간:** 6월 24일 ~

---

### 🎯 문제 정의
- 기존 BERT4ETH는 주소(`address`)를 직접 학습하여, 학습되지 않은 신규 주소는 추론 불가능
- 신규 주소 발생 시 모델 재학습 필요 → 운영 비용 증가, 실시간성 저하

---

### 💡 솔루션 – NEW BERT 핵심 아이디어
1. **`address` → `identity`** 재정의
   - 동일/다른 주소 여부만 학습 (`identity token`)
   - 전체 주소를 샘플링하지 않아도 loss 계산 가능
2. **2단계 학습 구조**
   - **1차:** Identity 마스킹(Masked Language Model)  
   - **2차:** 각 주소 시퀀스를 가깝게 학습하는 Cluster 단계
3. **추론 시 신규 주소 재학습 불필요**
   - Pretrained 모델 + RandomForest로 빠른 분류

---

### 🛠 주요 기능 및 구현
- Ethereum 거래 데이터(EOA 기준) 전처리
- MLM + Cluster Pretrain 구조 적용
- PyTorch 기반 BERT 변형 모델 구현
- 피싱 계좌 탐지용 MLP, RandomForest 분류기 실험
- 신규 주소 추론 성능 검증
- 데이터 시각화 및 confusion matrix 분석

---

### 📊 결과
결과 링크: https://github.com/loading1031/CUSTOM_BERT4ETH/tree/main/outputs/confusion_matrix
- **MLM 단계:** 기존 BERT 대비 Macro Avg 성능 향상
- **Cluster 단계:** Precision 향상, Recall 일부 감소 (과적합 주의)
- **확장 테스트:** Normal/Phishing 외 Tornado, Dean 데이터 포함 시에도 신규 주소 추론 가능성 확인
- 추천 조합: **1차 학습 BERT + RandomForest**

---

### 🖥 기술 스택
- **Frameworks:** PyTorch, Scikit-learn  
- **Languages:** Python  
- **Infra:** CUDA GPU 환경  
- **Tools:** Matplotlib, Pandas, NumPy

---

### 📂 데이터셋 구조
- Normal / Phishing 계좌별 in/out 거래 데이터
- 거래 시퀀스 최대 길이: 100  
- 주요 컬럼: `block_number`, `from_address`, `to_address`, `value`, `block_timestamp`

---

### 🔗 참고
- BERTP 학습 파일: [OneDrive 링크](https://gachonunivackr-my.sharepoint.com/:f:/g/personal/yoonsh1004z_o365_gachon_ac_kr/ErCHuxl0sYJFhxLQjJpNjYcB5Nw50UPdYPBNL-MWZMF8yQ?e=5Xjjyf)

# 모델 기초 구성

## 1. 모델 알고리즘

### $\begin{aligned} \psi &= 0.25 \times \text{XGBoost} + 0.25 \times \text{LightGBM} + 0.25 \times \text{RandomForest} \\& + \ 0.15 \times \text{GradientBoosting} + 0.1 \times \text{Deep Learning} \end{aligned}$

## 2. Risk Score

### $\begin{aligned} \delta &= \alpha \times \text{sales} + \beta \times \text{customer} + \gamma \times \text{market} \end{aligned}$

- where
    - $\alpha$ : 0.4(매출 가중치)
    - $\beta$ : 0.3(고객 가중치)
    - $\gamma$ : 0.3(시장 가중치)

# 모델 상세 수식

## 1. 전처리

- 구간형 칼럼 수치화
    
    $$
    \text{bin2rank}(b) = \begin{cases}
    0.05 & \text{if} \ b= 10\% \ \text{이하} \ \\ 0.175 & \text{if} \ b= 10-25\% \\ 0.375 & \text{if} \ b= 25-50\% \\ 0.625 & \text{if} \ b= 50-75\% \\ 0.875 & \text{if} \ b= 75-90\% \\ 0.95 & \text{if} \ b= 90\% \ \text{초과}
    \end{cases}
    $$
    
    - $r^{SAA}_{i, t}, r^{CNT}_{i, t}, r^{CUS}_{i, t}, r^{AOV}_{i, t}, r^{CXL}_{i, t}$에 대해 수행
- 순위비율 0-100 기준 → 백분율 [0, 1] 변환
    
    $$
    p^{ind\_rank}_{i, t} = \frac{\text{M12\_SME\_RY\_SAA\_PCE\_RT}_{i, t}}{100}, \\ p^{bzn\_rank}_{i, t} = \frac{\text{M12\_SME\_BZN\_SAA\_PCE\_RT}_{i, t}}{100}
    $$
    
    - 값이 작을수록 리스크 큼(상대 순위 낮음)
- [0, 1] 표준화
    
    $$
    nz(x) = \min(1, \max(0, x))
    $$
    

## 2. 변화/이상치 탐지

- 계절성 및 점프 확인용 MAD 사용
    
    $$
    MoM(r_{i ,t}) = r_{i, t} - r_{i, t - 1} \\ z_{i, t}(r) = \frac{r_{i, t} - \text{median}(r_{i, t - 12:t - 1})}{\sqrt{2} \cdot MAD(r_{i, t - 12:t - 1}) + \epsilon}
    $$
    
- 매출 급락 → 음의 급락 변화 → ReLU_ 사용
    
    $$
    \text{relu\_}(x) = \max(0, -x)
    $$
    

## 3. Sales Risk

- 사용 컬럼
    - RC_M1_SAA
    - RC_M1_TO_UE_CT
    - RC_M1_AC_NP_AT
    - APV_CE_RAT
    - DLV_SAA_RAT
    - M12_SME_RY_SAA_PCE_RT
    - M12_SME_BZN_SAA_PCE_RT

### 1. 랭크 및 변화량

$$
\begin{aligned} & r^{SAA}_{i, t} = bin2rank(\text{RC\_M1\_SAA}_{i, t}), \\ & r^{CNT}_{i, t} = bin2rank(\text{RC\_M1\_TO\_UE\_CT}_{i, t}), \\ & r^{AOV}_{i, t} = bin2rank(\text{RC\_M1\_AV\_NP\_AT}_{i, t}), \\ & r^{CXL}_{i, t} = bin2rank(\text{APV\_CE\_RAT}_{i, t}) \end{aligned} \\ \Delta r^{SAA}_{i, t} = MoM(r^{SAA}_{i, t}), \Delta r^{CNT}_{i, t} = MoM(r^{CNT}_{i, t}), \Delta r^{AOV}_{i, t} = MoM(r^{AOV}_{i, t})
$$

### 2. 급락/이상치 점수

$$
s^{drop}_{i, t} = nz(\frac{\text{relu\_}(\Delta r^{SAA}_{i, t}) + \text{relu\_}(\Delta r^{CNT}_{i, t})}{2}) \\ s^{aov\_risk}_{i, t} = nz(\text{relu\_}(z_{i, t}(r^{AOV}))) \\ s^{cxl}_{i, t} = nz(\max(r^{CXL}_{i, t}, \sigma(z_{i, t}(r^{CXL}))))
$$

### 3. 동종/상권 약세

$$
s^{peer}_{i, t} = nz(\frac{(1 - p^{ind\_rank}_{i, t}) + (1 - p^{bzn\_rank}_{i, t})}{2})
$$

### 4. 배달 의존도

$$
s^{dlv}_{i, t} = nz(\frac{\text{DLV\_SAA\_RAT}_{i, t}}{100}) \\ s^{dlv\_jump}_{i, t} = nz(\text{relu\_}(-MoM(s^{dlv}_{i, t})) + \text{relu\_}(-z_{i, t}(s^{dlv})))
$$

### 5. 최종 매출 위험도

$$
Sales\_Risk_{i, t} = 0.35 s^{drop}_{i, t} + 0.15 s^{aov\_risk}_{i, t} + 0.2 s^{cxl}_{i, t} + 0.2 s^{peer}_{i, t} + 0.1 s^{dlv}_{i, t} \cdot s^{dlv\_jump}_{i, t}
$$

## 4. Customer Risk

- 사용 컬럼
    - RC_M1_UE_CUS_CN
    - MCT_UE_CLN_REU_RAT
    - MCT_UE_CLN_NEW_RAT
    - 남/녀 연령비중(데이터셋3의 M12_*_RAT)
    - RC_M1_SHC_RSD/WP/FLP_UE_CLN_RAT

### 1. 고객수/충성도 급락

$$
r^{CUS}_{i, t} = bin2rank(\text{RC\_M1\_UE\_CUS\_CN}_{i, t}) \\ s^{cus\_drop}_{i, t} = nz(\text{relu\_}(MoM(r^{CUS}_{i, t})) + \text{relu\_}(z_{i, t}(r^{CUS})))
$$

- 재방문 및 신규 비율 → 0-1 정규화
    
    $$
    q^{reu}_{i, t} = nz(\frac{\text{MCT\_UE\_CLN\_REU\_RAT}_{i, t}}{100}), \\ q^{new}_{i, t} = nz(\frac{\text{MCT\_UE\_CLN\_NEW\_RAT}_{i, t}}{100}) \\ s^{loyal}_{i, t} = nz(\text{relu\_}(MoM(q^{reu}_{i, t}))), s^{acq}_{i, t} = nz(\text{relu\_}(MoM(q^{new}_{i, t})))
    $$
    

### 2. 연령 및 유형 집중도

- 연령/성별 분포 $w_k$ → 헤르핀달-허시만 지수(HHI) 측정
    
    $$
    \text{HHI}_{i, t} = \sum^{12}_{k = 1} w^2_{k, i, t}, \sum w_{k, i, t} = 1
    $$
    
- 유형(거주/직장/유동) → HHI 측정
    
    $$
    \text{HHI}^{type}_{i, t} = \sum^{3}_{m = 1} v^2_{m, i, t}
    $$
    
- 과밀 → 리스크
    
    $$
    s^{mix}_{i, t} = nz(\frac{\text{HHI}_{i, t} - \text{median}(\text{HHI}_{i, t - 12:t - 1})}{\text{MAD}(\text{HHI}_{i, t - 12:t - 1}) + \epsilon})_{+} \\ s^{type}_{i, t} = nz(\frac{\text{HHI}^{type}_{i, t} - \text{median}(\text{HHI}^{type}_{i, t - 12:t - 1})}{\text{MAD}(\text{HHI}^{type}_{i, t - 12:t - 1}) + \epsilon})_{+} \\ (\cdot)_{+} = \max(0, \cdot)
    $$
    

### 3. 최종 고객 위험도

$$
Customer\_Risk_{i, t} = 0.4 s^{cus\_drop}_{i, t} + 0.25 s^{loyal}_{i, t} + 0.2 s^{acq}_{i, t} + 0.1 s^{mix}_{i, t} + 0.05 s^{type}_{i, t}
$$

## 5. Market Risk

- 사용 컬럼
    - M12_SME_RY_ME_MCT_RAT(동일업종 해지비중)
    - M12_SME_BZN_ME_MCT_RAT(동일상권 해지비중)
    - M1_SME_RY_SAA_RAT
    - M1_SME_RY_CNT_RAT(동일업종 평균 대비 비율)
    - MCT_OPE_MS_CN(운영개월수 구간)
    - HPSN_MCT_ZCD_NM(업종)
    - HPSN_MCT_BZN_CD_NM(상권)

### 1. 폐업 및 해지 환경

$$
h^{ind}_{i, t} = nz(\frac{\text{M12\_SME\_RY\_ME\_MCT\_RAT}_{i, t}}{100}), \\ h^{bzn}_{i, t} = nz(\frac{\text{M12\_SME\_BZN\_ME\_MCT\_RAT}_{i, t}}{100}) \\ s^{closure\_env}_{i, t} = \frac{h^{ind}_{i, t} + h^{bzn}_{i, t}}{2}
$$

### 2. 동종 평균 대비 취약도

$$
u^{rev}_{i, t} = nz(1 - \frac{\text{M1\_SME\_RY\_SAA\_RAT}_{i, t}}{100}), \\ u^{cnt}_{i, t} = nz(1 - \frac{\text{M1\_SME\_RY\_CNT\_RAT}_{i, t}}{100}) \\ s^{underperf}_{i, t} = \frac{u^{rev}_{i, t} + u^{cnt}_{i, t}}{2}
$$

### 3. 영업 개월 수(생애주기)

$$
a_{i, t} = bin2rank(\text{MCT\_OPE\_MS\_CN}_{i, t})
$$

- 초기 및 말기 구간에서 위험 가중
    
    $$
    s^{age}_{i, t} = nz(4 \cdot \min(a_{i, t}, 1 - a_{i, t}))
    $$
    

### 4. 최종 시장 위험도

$$
Market\_Risk_{i, t} = 0.5 s^{closure\_env}_{i, t} + 0.35 s^{unerperf}_{i, t} + 0.15 s^{age}_{i, t}
$$

## 6. Risk Score

$$
RiskScore_{i, t} = 0.4 \cdot Sales\_Risk_{i, t} + 0.3 \cdot Customer\_Risk_{i, t} + 0.3 \cdot Market\_Risk_{i, t}
$$

## 7. 학습 모델 앙상블 & 파인튜닝

- 폐업/심각위험 확률 출력
    
    $$
    \hat{p}^{(m)}_{i, t} \in [0, 1], m \in \{\text{XGB}, \text{LGBM}, \text{RF}, \text{GB}, \text{DL}\}
    $$
    

### 1. 가중치 앙상블

$$
\hat{p}_{i, t} = 0.25 \hat{p}^{\text{XGB}}_{i, t} + 0.25 \hat{p}^{\text{LGBM}}_{i, t} + 0.25 \hat{p}^{\text{RF}}_{i, t} + 0.15 \hat{p}^{\text{GB}}_{i, t} + 0.1 \hat{p}^{\text{DL}}_{i, t}
$$

### 2. 확률 교정 - 이소토닉 회귀/Platt Scaling

$$
\tilde{p}_{i, t} = Calibrate(\hat{p}_{i, t})
$$

### 3. 위험 점수(최종 위기 확률)

$$
Pr(Crisis_{i, t}) = nz(\lambda \tilde{p}_{i, t} + (1 - \lambda)RiskScore_{i, t}) \\ \lambda \in [0.5, 0.7]
$$

## 8. 조기 경보 규칙

- ROC 기반 최적 임계치 $\tau$ 검증 결정
- 단일월 급등 < 지속성
    
    $$
    \text{Alert}_{i, t} = \begin{cases}
    \text{RED} & \text{if} \ Pr(Crisis_{i, t}) \geq \tau_{red} \& \overline{Pr}^{(3)}_{i, t} \geq \tau_{red} - \delta \ \\ \text{ORANGE} & \text{if} \ Pr(Crisis_{i, t}) \geq \tau_{org} \\ \text{YELLOW} & \text{if} \ Pr(Crisis_{i, t}) \geq \tau_{ylw} \\ \text{GREEN} & \text{otherwise}
    \end{cases} \\ \overline{Pr}^{(3)}_{i, t} = \frac{1}{3}\sum^{2}_{k = 0} Pr(Crisis_{i, t - k}) \\ \tau_{ylw} = 0.6, \tau_{org} = 0.7, \tau_{red} = 0.8, \delta = 0.05
    $$
    

## 9. 매출/고객/시장 개별 진단 수식

- 각 영영 세부 지표 → 0-100 점수 환산 후 대시보드 작성
    
    $$
    Score^{Sales}_{i, t} = 100 \cdot (1 - Sales\_Risk_{i, t}), \\ Score^{Customer}_{i, t} = 100 \cdot (1 - Customer\_Risk_{i, t}), \\ Score^{Market}_{i, t} = 100 \cdot (1 - Market\_Risk_{i, t})
    $$
    
- 민감도 → SHAP
    
    $$
    SHAP_{i, t}(x_j) = \phi_j \Rightarrow \Delta Score \approx -C \cdot \phi_j
    $$
    

## 10. 기타 사항

- 키 결합 : ENCODED_MCT, TA_YM 기준 데이터셋 2, 3 조인, 데이터셋 1 정적 속성 조인
- 결측 : SV → 결측으로 간주, 시계열 보간 + 결측 플래그(이진변수)
- 라벨 : 폐업 위험 학습 $y_{i, t} = 1\{\text{MCT\_ME\_D} \in [t, t + K)\}$ 형태
- 예측 시점 : $t$ 기준 $t + 1 \sim t + K$ 이벤트에서 예측
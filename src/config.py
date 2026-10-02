"""경로 · 상수 · 피처 목록 (모든 스크립트가 여기 값을 공유한다)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
RESULTS_DIR = ROOT / 'results'
FIG_DIR = RESULTS_DIR / 'figures'

RANDOM_STATE = 42
BATCHES = {'b1': '2017-05-12', 'b2': '2018-02-20', 'b3': '2018-04-12'}   # 학습 / 테스트 / 추가 검증
MAX_CYCLE = 101                  # Qdlin을 저장할 마지막 사이클 번호 (100사이클 피처 + 여유 1)

# 라벨 · 품질 기준
NOMINAL_AH = 1.1
EOL_AH = 0.88                    # 정격 1.1Ah의 80% = 수명 종료(EOL)
CENSOR_AH = 0.90                 # 기록 마지막 5사이클 평균이 이보다 크면 EOL 전에 실험 종료(수명 미확정)
QD_VALID = (0.80, 1.20)          # 이 범위 밖 방전 용량 = 측정 오류(0, 2.x Ah 스파이크)

# 피처 (1일차 전략: ΔQ 분산 확정 + 후보 4개는 학습 데이터 교차검증으로 선택)
CORE = ['dq_var_log']
CANDIDATES = ['dq_skew', 'qd_slope_2_100', 'ir_min', 'tavg_mean']
EXCLUDED = {                     # 1일차 EDA에서 제외한 피처와 이유
    'chargetime_2_6': 'ΔQ 분산과 중복 (ΔQ 반영 후 남는 상관 0.09, 더해도 교차검증 개선 없음)',
    'qd_2': '초기 용량 값은 수명별 차이 없음',
    'dq_min_log': 'ΔQ 분산과 중복 (상관 1.00)',
    'dq_mean_log': 'ΔQ 분산과 중복',
}
TARGET = 'log_life'              # log10(수명): 수명 범위가 수백~2천으로 넓어 비율 오차를 고르게
TARGET_MAPE = 9.1                # 원논문(Severson 2019) 1차 테스트 MAPE

# 손실 함수 (단위: 배터리 1개 교체비)
#   일찍 교체: 남긴 수명 동안 더 내줄 수 있었던 방전량 비율 (B1 실제 용량 곡선으로 측정, features.energy_value_curve)
#   늦게 교체: 1건당 R (공급 중단 · 위약금 · 긴급 교체 할증) — 회사 값이라 모름 → 여러 값으로 모두 실험
#   (R = 0, 즉 늦어도 손실이 없으면 '무조건 길게 예측'이 답이 되므로 0.1부터)
LATE_COST_GRID = [i / 10 for i in range(1, 31)] + [5.0, 10.0, float('inf')]

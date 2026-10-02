"""
3단계 학습 · 선택 · 평가 파이프라인.

데이터 분할 (테스트 데이터는 마지막에 한 번만 본다)
    Batch 1 36셀  → 학습 28셀 + 검증(Hold-out) 8셀
                    같은 충전 방식(policy)의 셀은 반드시 같은 쪽에 둔다 (짝 셀이 양쪽에 갈리면 누수).
                    방식 20개를 평균 수명 순으로 세우고 4개마다 1개(index % 4 == 1)를 검증으로 → 수명 범위 고르게
    학습 28셀 안 교차검증 → GroupKFold(5, groups=policy)
    Batch 2 39셀  → 테스트 (최종 모델로 1회 평가)
    Batch 3 44셀  → 추가 검증 (선택 과제)

두 층으로 나눠 평가한다
    ① 모델 = 수명을 얼마나 정확히 맞히나 → MAPE (원논문 9.1%와 비교, 과제 성능표 포맷)
    ② 여유 배율 = 예측에 곱해 일찍 경고하는 정도 → 회사 손실 기준
       셀 하나의 손실 (단위: 배터리 교체비)
         일찍 교체: v(남긴 수명 비율)   v = B1 용량 곡선으로 측정한 '남긴 방전량 비율'
         늦게 교체: R                  R = 고장 1건의 추가 손실 (회사 값이라 모름)
       R을 0 ~ 3(0.1 간격), 5, 10, ∞로 바꿔 가며 R마다
         - 학습 교차검증 예측에서 손실이 최소인 여유 배율을 구하고
         - 그 손실이 가장 작은 모델을 고른다 (학습 데이터만 사용)
       → R마다 고른 모델 · 여유 배율 · 검증/B2/B3 결과를 모두 표로 남긴다 (results/cost_sensitivity.csv)

전처리(결측 대체 · 표준화)는 Pipeline 안에 있어 각 fold의 학습 부분에만 fit 된다.

사용: python src/train.py   (data/processed/features.csv → results/*.csv)
"""
from itertools import combinations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (CANDIDATES, CORE, LATE_COST_GRID, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR,
                    TARGET, TARGET_MAPE)
from features import energy_value_curve
from preprocess import load_cells

N_FOLDS = 5
MARGINS = np.round(np.arange(1.20, 0.5999, -0.001), 3)   # 큰 값부터: 손실이 같으면 덜 일찍 경고하는 쪽을 고른다
BASELINE = '선형회귀 (ΔQ 분산)'


# ---------- 데이터 ----------
def load_data():
    df = pd.read_csv(PROCESSED_DIR / 'features.csv')
    used = df[df.status == 'used']
    b1, b2, b3 = (used[used.batch == b].reset_index(drop=True) for b in ['b1', 'b2', 'b3'])
    return b1, b2, b3


def split_holdout(b1):
    """충전 방식 단위 Hold-out: 평균 수명 순으로 정렬한 방식 중 4개마다 1개를 검증으로."""
    order = b1.groupby('policy').cycle_life.mean().sort_values().index
    is_valid = b1.policy.isin(set(order[1::4]))
    return b1[~is_valid].reset_index(drop=True), b1[is_valid].reset_index(drop=True)


# ---------- 지표 · 손실 ----------
def mape(y, p):
    return float(np.mean(np.abs((np.asarray(p) - y) / y)) * 100)


def cell_loss(y, p, R, curve):
    """셀별 손실 (교체비 단위). 늦게(p > y) = R, 일찍 = 남긴 방전량 비율."""
    e = (np.asarray(p, float) - y) / y
    early = np.interp(np.clip(-e, 0, None), *curve)
    return np.where(e > 0, R, early)


def summarize(y, p, R, curve):
    y = np.asarray(y, float)
    e = (np.asarray(p, float) - y) / y
    loss = cell_loss(y, p, R, curve) if np.isfinite(R) else cell_loss(y, p, 0, curve)
    return {'n': len(y), '늦은 교체': int((e > 0).sum()), 'MAPE': mape(y, p),
            '평균 일찍(%)': float(np.mean(np.clip(-e, 0, None)) * 100),
            '최대 늦음(%)': float(max(e.max(), 0) * 100),
            '셀당 손실': float(loss.mean()) if np.isfinite(R) or (e <= 0).all() else np.inf}


def best_margin(y, pred, R, curve):
    """손실(셀 평균)이 최소인 여유 배율. R = ∞ 이면 늦은 교체 0건 중 가장 덜 일찍 경고하는 배율."""
    y = np.asarray(y, float)
    losses = [cell_loss(y, pred * m, R, curve).mean() for m in MARGINS]
    i = int(np.argmin(losses))
    return float(MARGINS[i]), float(losses[i])


# ---------- 후보 ----------
def pipe(est):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def candidates():
    """1일차 전략의 후보: 선형회귀(기준선) · 피처 조합 선형회귀 · ElasticNet · 분위수 회귀 · 비교용 트리 2종."""
    out = [(BASELINE, CORE, pipe(LinearRegression()))]
    for k in range(1, len(CANDIDATES) + 1):                       # 후보 피처 조합 → 교차검증으로 선택
        for extra in combinations(CANDIDATES, k):
            out.append((f'선형회귀 (ΔQ 분산 + {" + ".join(extra)})', CORE + list(extra), pipe(LinearRegression())))
    feats = CORE + CANDIDATES
    for a in [0.001, 0.003, 0.01, 0.03]:
        for l1 in [0.1, 0.5, 0.9]:
            out.append((f'ElasticNet (alpha={a}, l1={l1})', feats, pipe(ElasticNet(alpha=a, l1_ratio=l1, max_iter=100000))))
    for q in [0.1, 0.2]:
        out.append((f'분위수 회귀 (하위 {int(q * 100)}%, ΔQ 분산)', CORE, pipe(QuantileRegressor(quantile=q, alpha=0, solver='highs'))))
        out.append((f'분위수 회귀 (하위 {int(q * 100)}%, 전체 후보)', feats, pipe(QuantileRegressor(quantile=q, alpha=0, solver='highs'))))
    out.append(('랜덤포레스트 (비교용)', feats, pipe(RandomForestRegressor(500, min_samples_leaf=2, random_state=RANDOM_STATE))))
    out.append(('LightGBM (비교용)', feats, pipe(LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=4,
                                                            min_child_samples=3, random_state=RANDOM_STATE, verbose=-1))))
    return out


def family(name):
    return name.split(' (')[0]


def oof_predict(est, X, y, groups):
    """GroupKFold(같은 충전 방식은 같은 fold) out-of-fold 예측 → 수명(사이클)."""
    p = np.zeros(len(y))
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        p[te] = clone(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return 10 ** p


# ---------- 단계별 예측 ----------
def stage_predictions(cand, train, valid, b1, b2, b3):
    """후보마다: 학습 28셀 교차검증 · 검증 8셀 · (B1 36셀 재학습 후) B1 교차검증 · B2 · B3 예측."""
    out = {}
    for name, feats, est in cand:
        model_tr = clone(est).fit(train[feats], train[TARGET])
        model_b1 = clone(est).fit(b1[feats], b1[TARGET])
        out[name] = {
            'train_cv': oof_predict(est, train[feats], train[TARGET], train.policy),
            'valid': 10 ** model_tr.predict(valid[feats]),
            'b1_cv': oof_predict(est, b1[feats], b1[TARGET], b1.policy),
            'b2': 10 ** model_b1.predict(b2[feats]),
            'b3': 10 ** model_b1.predict(b3[feats]),
        }
    return out


def performance_table(name, P, train, valid, b2, b3):
    """과제 리포팅 포맷 (모델 정확도, 여유 배율 적용 전): Train(B1 CV) / Valid(B1 Hold-out) / Test(B2) / Gap."""
    p = P[name]
    t = pd.Series({
        'Train (B1 CV)': mape(train.cycle_life, p['train_cv']),
        'Valid (B1 Hold-out)': mape(valid.cycle_life, p['valid']),
        'Test (B2)': mape(b2.cycle_life, p['b2']),
        'Test (B3, 추가)': mape(b3.cycle_life, p['b3']),
    })
    # MAPE는 낮을수록 좋으므로 Gap = 뒤 − 앞 : (+)면 뒤 단계에서 성능 저하
    t['Gap (Train−Valid)'] = t['Valid (B1 Hold-out)'] - t['Train (B1 CV)']
    t['Gap (Valid−Test)'] = t['Test (B2)'] - t['Valid (B1 Hold-out)']
    t['Gap (Target−Test)'] = t['Test (B2)'] - TARGET_MAPE
    t['Gap (B2−B3, 추가)'] = t['Test (B3, 추가)'] - t['Test (B2)']
    return t.rename(name)


def cost_sweep(P, train, valid, b1, b2, b3, curve):
    """R마다 학습 데이터로 모델 · 여유 배율 선택 → 검증 · B2 · B3에 그대로 적용."""
    rows = []
    for R in LATE_COST_GRID:
        scores = {n: best_margin(train.cycle_life, p['train_cv'], R, curve) for n, p in P.items()}
        name = min(scores, key=lambda n: scores[n][1])
        m_tr, loss_tr = scores[name]
        m_b1, _ = best_margin(b1.cycle_life, P[name]['b1_cv'], R, curve)   # 최종 모델(B1 36셀)용 여유 배율
        row = {'R': R, '선택 모델': name, '여유 배율 (학습 28셀)': m_tr, '여유 배율 (B1 36셀)': m_b1}
        for split, y, p, m in [('학습 CV', train.cycle_life, P[name]['train_cv'], m_tr), ('검증', valid.cycle_life, P[name]['valid'], m_tr),
                               ('B2', b2.cycle_life, P[name]['b2'], m_b1), ('B3', b3.cycle_life, P[name]['b3'], m_b1)]:
            s = summarize(y, p * m, R, curve)
            row[f'{split} 늦은 교체'] = f"{s['늦은 교체']}/{s['n']}"
            row[f'{split} 평균 일찍(%)'] = s['평균 일찍(%)']
            row[f'{split} 셀당 손실'] = s['셀당 손실']
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    b1, b2, b3 = load_data()
    train, valid = split_holdout(b1)
    print(f'학습 {len(train)}셀({train.policy.nunique()}방식) / 검증 {len(valid)}셀({valid.policy.nunique()}방식) / 테스트 B2 {len(b2)} / 추가 B3 {len(b3)}')

    # 일찍 교체 손실 곡선 — 학습 배치(B1) 용량 곡선으로 측정
    curve = energy_value_curve(load_cells(['b1']), dict(zip(b1.cell_id, b1.cycle_life)))
    pd.DataFrame({'남긴 수명 비율': curve[0], '남긴 방전량 비율': curve[1]}).round(4).to_csv(RESULTS_DIR / 'energy_value_curve.csv', index=False)
    print('남긴 수명 10/20/30/50% → 남긴 방전량', np.round(np.interp([.1, .2, .3, .5], *curve), 3))

    cand = candidates()
    P = stage_predictions(cand, train, valid, b1, b2, b3)

    # ① 모델 정확도 — 후보 비교는 학습 · 검증만
    acc = pd.DataFrame([{'후보': n, '계열': family(n), '피처': ' + '.join(f),
                         'Train CV MAPE': mape(train.cycle_life, P[n]['train_cv']), 'Valid MAPE': mape(valid.cycle_life, P[n]['valid'])}
                        for n, f, _ in cand]).sort_values('Train CV MAPE').reset_index(drop=True)
    acc.round(3).to_csv(RESULTS_DIR / 'candidates_cv.csv', index=False)

    # ② 손실 기준 선택 — R마다
    sweep = cost_sweep(P, train, valid, b1, b2, b3, curve)
    sweep.round(4).to_csv(RESULTS_DIR / 'cost_sensitivity.csv', index=False)
    show = ['R', '선택 모델', '여유 배율 (B1 36셀)', '학습 CV 늦은 교체', '검증 늦은 교체', 'B2 늦은 교체', 'B2 평균 일찍(%)', 'B3 늦은 교체']
    print('\n[R별 선택 — R = 고장 1건 추가 손실 ÷ 교체비]')
    print(sweep[show].round(3).to_string(index=False))

    # 성능표 — 기준선 + 계열별 최선(학습 CV) + R별로 선택된 모델
    report = [BASELINE] + [n for n in acc.groupby('계열', sort=False).head(1).후보 if n != BASELINE]
    report += [n for n in sweep['선택 모델'].unique() if n not in report]
    perf = pd.concat([performance_table(n, P, train, valid, b2, b3) for n in report], axis=1)
    perf.index.name = '구분 (MAPE %, 여유 배율 적용 전)'
    perf.round(2).to_csv(RESULTS_DIR / 'model_performance.csv')
    print('\n[모델 정확도 — 과제 포맷]')
    print(perf.round(2).to_string())

    preds = pd.concat([pd.DataFrame({'model': n, 'split': s, 'cell_id': d.cell_id, 'policy': d.policy,
                                     'cycle_life': d.cycle_life, 'pred': P[n][k]})
                       for n in report for s, k, d in [('train_cv', 'train_cv', train), ('valid', 'valid', valid),
                                                       ('test_b2', 'b2', b2), ('test_b3', 'b3', b3)]])
    preds.round(2).to_csv(RESULTS_DIR / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()

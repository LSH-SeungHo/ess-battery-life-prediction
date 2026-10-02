"""
3단계 학습 · 선택 · 평가 파이프라인.

데이터 분할 (B1 = 학습 배치, B2 = 테스트 배치)
    B1 36셀 → 학습 28셀 + 검증(Hold-out) 8셀
              같은 충전 방식(policy)의 셀은 반드시 같은 쪽에 둔다 (짝 셀이 양쪽에 갈리면 누수).
              방식 20개를 평균 수명 순으로 세우고 4개마다 1개(index % 4 == 1)를 검증으로 → 수명 범위 고르게
    학습 28셀 안 교차검증 → GroupKFold(5, groups=policy)
    선택이 끝나면 B1 36셀 전부로 다시 학습 = 최종 모델
    B2 39셀 전부 → 테스트 · B3 44셀 전부 → 추가 검증 (선택이 끝난 뒤 보고 대상 모델에만 1회 적용)

두 층으로 나눠 평가한다
    ① 모델 = 수명을 얼마나 정확히 맞히나 → MAPE (원논문 9.1%와 비교, 과제 성능표 포맷)
    ② 여유 배율 = 예측에 곱해 일찍 경고하는 정도 → 회사 손실 기준
       셀 하나의 손실 (단위: 배터리 교체비)
         일찍 교체: v(남긴 수명 비율)   v = 용량 곡선으로 측정한 '남긴 방전량 비율'
         늦게 교체: R                  R = 수명 종료를 넘겨 쓴 1건의 추가 손실 (회사 값이라 모름)
       R = 0.1 ~ 3.0(0.1 간격), 5, 10, ∞ 마다
         - 학습 28셀 교차검증 예측 + 학습 28셀로 만든 v로 손실이 최소인 여유 배율과 모델을 고른다
         - 최종 여유 배율은 B1 36셀 교차검증 예측 + B1 36셀로 만든 v로 다시 구한다
       검증 8셀에는 학습용 배율, B2 · B3에는 최종 배율을 곱한다

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


def value_curve(df, cells):
    return energy_value_curve(cells, dict(zip(df.cell_id, df.cycle_life)))


# ---------- 지표 · 손실 ----------
def mape(y, p):
    return float(np.mean(np.abs((np.asarray(p) - y) / y)) * 100)


def cell_loss(y, p, R, curve):
    """셀별 손실 (교체비 단위). 늦게(p > y) = R, 일찍 = 남긴 방전량 비율."""
    e = (np.asarray(p, float) - y) / y
    early = np.interp(np.clip(-e, 0, None), *curve)
    return np.where(e > 0, R, early)


def summarize(y, p):
    """늦은 교체 건수와 '평균 일찍(%)' = 셀마다 max(실제 − 예측, 0) / 실제 의 평균 (늦은 셀은 0으로 포함)."""
    y = np.asarray(y, float)
    e = (np.asarray(p, float) - y) / y
    return {'n': len(y), '늦은 교체': int((e > 0).sum()), '평균 일찍(%)': float(np.mean(np.clip(-e, 0, None)) * 100)}


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


def oof_predict(est, X, y, groups, life=None):
    """GroupKFold(같은 충전 방식은 같은 fold) out-of-fold 예측 → 수명(사이클). life를 주면 fold별 MAPE도 반환."""
    p, fold_mape = np.zeros(len(y)), []
    for tr, te in GroupKFold(N_FOLDS).split(X, y, groups):
        p[te] = 10 ** clone(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        if life is not None:
            fold_mape.append(mape(life.iloc[te].values, p[te]))
    return (p, fold_mape) if life is not None else p


# ---------- 1) 선택 단계: B1만 사용 ----------
def b1_predictions(cand, train, valid, b1):
    """후보마다 학습 28셀 교차검증 · 검증 8셀 · B1 36셀 교차검증 예측 (B2 · B3는 아직 쓰지 않는다)."""
    out = {}
    for name, feats, est in cand:
        p_cv, folds = oof_predict(est, train[feats], train[TARGET], train.policy, train.cycle_life)
        out[name] = {'train_cv': p_cv, 'train_folds': folds,
                     'valid': 10 ** clone(est).fit(train[feats], train[TARGET]).predict(valid[feats]),
                     'b1_cv': oof_predict(est, b1[feats], b1[TARGET], b1.policy)}
    return out


def select_by_cost(P, train, b1, curve_tr, curve_b1):
    """R마다 학습 28셀 기준으로 모델 · 여유 배율 선택, 최종 배율은 B1 36셀로."""
    rows = []
    for R in LATE_COST_GRID:
        scores = {n: best_margin(train.cycle_life, p['train_cv'], R, curve_tr) for n, p in P.items()}
        name = min(scores, key=lambda n: scores[n][1])
        m_tr, loss_tr = scores[name]
        m_b1, _ = best_margin(b1.cycle_life, P[name]['b1_cv'], R, curve_b1)
        rows.append({'R': R, '선택 모델': name, '여유 배율 (학습 28셀)': m_tr, '여유 배율 (최종, B1 36셀)': m_b1})
    return pd.DataFrame(rows)


# ---------- 2) 평가 단계: 보고 대상 모델만 B2 · B3에 적용 ----------
def final_predictions(names, cand, b1, b2, b3):
    spec = {n: (f, e) for n, f, e in cand}
    out = {}
    for n in names:
        feats, est = spec[n]
        model = clone(est).fit(b1[feats], b1[TARGET])
        out[n] = {'b2': 10 ** model.predict(b2[feats]), 'b3': 10 ** model.predict(b3[feats])}
    return out


def evaluate_sweep(sel, P, F, train, valid, b1, b2, b3, curve_tr, curve_b1):
    """R마다 선택 결과를 검증 · B2 · B3에 적용. 셀당 손실(교체비 단위)은 R이 유한할 때만 계산하고,
    같은 R에서 기준선(ΔQ 선형회귀 + 그 모델의 최적 배율)의 손실도 함께 남긴다."""
    rows = []
    for _, r in sel.iterrows():
        n, row, R = r['선택 모델'], r.to_dict(), r['R']
        m_base_tr, _ = best_margin(train.cycle_life, P[BASELINE]['train_cv'], R, curve_tr)
        m_base_b1, _ = best_margin(b1.cycle_life, P[BASELINE]['b1_cv'], R, curve_b1)
        for split, y, p, m, pb, mb, cv in [
                ('학습 CV', train.cycle_life, P[n]['train_cv'], r['여유 배율 (학습 28셀)'], P[BASELINE]['train_cv'], m_base_tr, curve_tr),
                ('검증', valid.cycle_life, P[n]['valid'], r['여유 배율 (학습 28셀)'], P[BASELINE]['valid'], m_base_tr, curve_tr),
                ('B2', b2.cycle_life, F[n]['b2'], r['여유 배율 (최종, B1 36셀)'], F[BASELINE]['b2'], m_base_b1, curve_b1),
                ('B3', b3.cycle_life, F[n]['b3'], r['여유 배율 (최종, B1 36셀)'], F[BASELINE]['b3'], m_base_b1, curve_b1)]:
            s = summarize(y, p * m)
            row[f'{split} 늦은 교체'] = f"{s['늦은 교체']}/{s['n']}"
            row[f'{split} 평균 일찍(%)'] = s['평균 일찍(%)']
            if np.isfinite(R):
                row[f'{split} 셀당 손실'] = float(cell_loss(y.values, p * m, R, cv).mean())
                row[f'{split} 셀당 손실 (기준선)'] = float(cell_loss(y.values, pb * mb, R, cv).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def nested_cv(cand, b1, cells_b1):
    """선택 과정 전체(피처 조합 · 모델 · 여유 배율)를 B1 안에서 한 번 더 감싸 평가한다 (중첩 교차검증).
    바깥 GroupKFold(5, 충전 방식)의 학습 부분에서 안쪽 GroupKFold로 선택을 처음부터 다시 하고,
    바깥 평가 부분(처음 보는 충전 방식의 셀)에 적용한다. → 선택 과정의 낙관 편향까지 포함한 B1 안 성능"""
    cell_rows, acc_rows = [], []
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(b1, b1[TARGET], b1.policy)):
        inner, test = b1.iloc[tr].reset_index(drop=True), b1.iloc[te].reset_index(drop=True)
        curve = value_curve(inner, cells_b1)
        oof, fold_mean, outp = {}, {}, {}
        for n, f, e in cand:
            oof[n], folds = oof_predict(e, inner[f], inner[TARGET], inner.policy, inner.cycle_life)
            fold_mean[n] = np.mean(folds)
            outp[n] = 10 ** clone(e).fit(inner[f], inner[TARGET]).predict(test[f])
        best = min(fold_mean, key=fold_mean.get)                       # 정확도(MAPE) 기준 선택
        acc_rows.append({'fold': k, '선택 모델 (MAPE 기준)': best, 'MAPE': mape(test.cycle_life, outp[best]), 'n': len(test)})
        for R in LATE_COST_GRID:                                       # 손실 기준 선택
            scores = {n: best_margin(inner.cycle_life, oof[n], R, curve) for n in oof}
            n = min(scores, key=lambda x: scores[x][1])
            m = scores[n][0]
            y, p = test.cycle_life.values, outp[n] * m
            e = (p - y) / y
            cell_rows.append(pd.DataFrame({'fold': k, 'R': R, '선택 모델': n, '여유 배율': m, 'cell_id': test.cell_id,
                                           '늦은 교체': e > 0, '일찍(%)': np.clip(-e, 0, None) * 100,
                                           '손실': cell_loss(y, p, R, curve) if np.isfinite(R) else np.nan}))
    cells = pd.concat(cell_rows)
    acc = pd.DataFrame(acc_rows)
    summary = cells.groupby('R').agg(늦은_교체=('늦은 교체', 'sum'), 셀=('cell_id', 'size'), 평균_일찍=('일찍(%)', 'mean'),
                                     셀당_손실=('손실', 'mean'), 배율_최소=('여유 배율', 'min'), 배율_최대=('여유 배율', 'max'),
                                     선택_모델=('선택 모델', lambda x: ' / '.join(sorted(set(x))))).reset_index()
    return summary, acc


def performance_table(name, P, F, train, valid, b2, b3):
    """과제 리포팅 포맷 (모델 정확도, 여유 배율 적용 전 MAPE %).
    Gap은 행 이름 그대로 '앞 − 뒤'로 계산한다. MAPE는 낮을수록 좋으므로 Gap이 (−)이면 뒤 단계에서 오차가 커졌다는 뜻."""
    p, f = P[name], F[name]
    ok2 = b2.early_data_ok.astype(bool).values
    t = pd.Series({
        'Train (B1 CV, fold 평균)': np.mean(p['train_folds']),
        'Train (B1 CV, fold 표준편차)': np.std(p['train_folds']),
        'Valid (B1 Hold-out)': mape(valid.cycle_life, p['valid']),
        'Test (B2)': mape(b2.cycle_life, f['b2']),
        'Test (B3, 추가)': mape(b3.cycle_life, f['b3']),
    })
    t['Gap (Train−Valid)'] = t['Train (B1 CV, fold 평균)'] - t['Valid (B1 Hold-out)']
    t['Gap (Valid−Test)'] = t['Valid (B1 Hold-out)'] - t['Test (B2)']
    t['Gap (Target−Test)'] = TARGET_MAPE - t['Test (B2)']
    t['Gap (B2−B3, 추가)'] = t['Test (B2)'] - t['Test (B3, 추가)']
    t['Gap (Target−Test, B3 기준)'] = TARGET_MAPE - t['Test (B3, 추가)']
    t[f'참고: Test (B2, 초기 데이터 품질 통과 {ok2.sum()}셀)'] = mape(b2.cycle_life[ok2], f['b2'][ok2])
    return t.rename(name)


def main():
    b1, b2, b3 = load_data()
    train, valid = split_holdout(b1)
    print(f'학습 {len(train)}셀({train.policy.nunique()}방식) / 검증 {len(valid)}셀({valid.policy.nunique()}방식) / 테스트 B2 {len(b2)} / 추가 B3 {len(b3)}')

    # 일찍 교체 손실 곡선 — 선택용은 학습 28셀, 최종 배율용은 B1 36셀
    cells_b1 = load_cells(['b1'])
    curve_tr, curve_b1 = value_curve(train, cells_b1), value_curve(b1, cells_b1)
    pd.DataFrame({'남긴 수명 비율': curve_tr[0], '남긴 방전량 비율 (학습 28셀)': curve_tr[1],
                  '남긴 방전량 비율 (B1 36셀)': curve_b1[1]}).round(4).to_csv(RESULTS_DIR / 'energy_value_curve.csv', index=False)
    print('남긴 수명 10/20/30/50% → 남긴 방전량 (학습 28셀)', np.round(np.interp([.1, .2, .3, .5], *curve_tr), 3))

    # 1) 선택 — B1만
    cand = candidates()
    P = b1_predictions(cand, train, valid, b1)
    acc = pd.DataFrame([{'후보': n, '계열': family(n), '피처': ' + '.join(f),
                         'Train CV MAPE (fold 평균)': np.mean(P[n]['train_folds']), 'Train CV MAPE (fold 표준편차)': np.std(P[n]['train_folds']),
                         'Valid MAPE': mape(valid.cycle_life, P[n]['valid'])}
                        for n, f, _ in cand]).sort_values('Train CV MAPE (fold 평균)').reset_index(drop=True)
    acc.round(3).to_csv(RESULTS_DIR / 'candidates_cv.csv', index=False)
    sel = select_by_cost(P, train, b1, curve_tr, curve_b1)

    # 2) 평가 — 보고 대상: 기준선 + 계열별 최선(학습 CV) + R별 선택 모델
    report = [BASELINE] + [n for n in acc.groupby('계열', sort=False).head(1).후보 if n != BASELINE]
    report += [n for n in sel['선택 모델'].unique() if n not in report]
    F = final_predictions(report, cand, b1, b2, b3)

    sweep = evaluate_sweep(sel, P, F, train, valid, b1, b2, b3, curve_tr, curve_b1)
    sweep.round(4).to_csv(RESULTS_DIR / 'cost_sensitivity.csv', index=False)
    show = ['R', '선택 모델', '여유 배율 (학습 28셀)', '여유 배율 (최종, B1 36셀)', '학습 CV 늦은 교체', '검증 늦은 교체', 'B2 늦은 교체', 'B3 늦은 교체', 'B3 평균 일찍(%)']
    print('\n[R별 선택]')
    print(sweep[show].round(3).to_string(index=False))

    perf = pd.concat([performance_table(n, P, F, train, valid, b2, b3) for n in report], axis=1)
    perf.index.name = '구분 (MAPE %, 여유 배율 적용 전)'
    perf.round(2).to_csv(RESULTS_DIR / 'model_performance.csv')
    print('\n[모델 정확도 — 과제 포맷]')
    print(perf.round(2).to_string())

    # 3) 선택 과정까지 포함한 B1 중첩 교차검증
    nest, nest_acc = nested_cv(cand, b1, cells_b1)
    nest.round(4).to_csv(RESULTS_DIR / 'nested_cv.csv', index=False)
    print('\n[B1 중첩 교차검증 — 정확도 기준 선택 모델의 바깥 fold MAPE]')
    print(nest_acc.round(2).to_string(index=False), '\n가중 평균 MAPE %.2f%%' % np.average(nest_acc.MAPE, weights=nest_acc.n))
    print('\n[B1 중첩 교차검증 — R별 손실 기준 선택]')
    print(nest[['R', '늦은_교체', '셀', '평균_일찍', '셀당_손실', '배율_최소', '배율_최대']].round(3).to_string(index=False))

    preds = []
    for n in report:
        for s, d, p in [('train_cv', train, P[n]['train_cv']), ('valid', valid, P[n]['valid']),
                        ('test_b2', b2, F[n]['b2']), ('test_b3', b3, F[n]['b3'])]:
            preds.append(pd.DataFrame({'model': n, 'split': s, 'cell_id': d.cell_id, 'policy': d.policy,
                                       'cycle_life': d.cycle_life, 'pred': p}))
    pd.concat(preds).round(2).to_csv(RESULTS_DIR / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()

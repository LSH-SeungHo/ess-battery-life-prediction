"""
2단계 피처: 셀 dict(preprocess.py 결과) → 셀 단위 피처 테이블.

피처는 초기 100사이클(사이클 번호 2~100)만 사용한다. 100사이클 이후 값은 예측 시점에 없는 미래 정보.

핵심 피처 ΔQ(V)
    ΔQ(V) = Q_100(V) − Q_10(V)
    Q_n(V): n번째 사이클 방전 곡선을 공통 전압 눈금(3.5→2.0V, 1,000점)에 맞춰 다시 찍은 방전 용량(Qdlin)
    dq_var_log = log10( Var_V[ΔQ(V)] )   ← 1,000개 전압 점에서의 분산

사용: python src/features.py   (data/processed/b*.pkl → data/processed/features.csv)
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from config import PROCESSED_DIR
from preprocess import alignment_check, early_data_ok, eol_cycle, label_status, load_cells, valid_ir, valid_qd


def delta_q(cell, hi=100, lo=10):
    """ΔQ(V) = Q_hi(V) − Q_lo(V)  (같은 전압 눈금 1,000점에서 뺄셈)."""
    return cell['qdlin'][hi - 1] - cell['qdlin'][lo - 1]


def knee_point(cell, start=2):
    """2구간 선형 근사로 무릎점(급격한 열화 시작점) 탐색.
    QD(사이클)를 두 직선으로 나눴을 때 제곱오차 합이 최소인 분기 사이클을 무릎점으로 본다.
    반환: (knee 사이클, 무릎 전 기울기, 무릎 후 기울기)  [Ah/cycle]"""
    qd = valid_qd(cell)
    c = np.arange(1, len(qd) + 1)
    m = (c >= start) & ~np.isnan(qd)
    x, y = c[m].astype(float), qd[m]
    best = (np.inf, None)
    for k in range(20, len(x) - 20, 2):
        p1 = np.polyfit(x[:k], y[:k], 1)
        p2 = np.polyfit(x[k:], y[k:], 1)
        sse = ((np.polyval(p1, x[:k]) - y[:k]) ** 2).sum() + ((np.polyval(p2, x[k:]) - y[k:]) ** 2).sum()
        if sse < best[0]:
            best = (sse, (x[k], p1[0], p2[0]))
    return best[1] if best[1] else (np.nan, np.nan, np.nan)


def parse_policy(p):
    """'5.4C(40%)-3.6C' → (C1=5.4, Q1=40, C2=3.6): 1단계 C-rate, 전환 SOC(%), 2단계 C-rate."""
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)C', p)
    if not m:
        return np.nan, np.nan, np.nan
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def cell_features(cell):
    s = cell['summary']
    qd = valid_qd(cell)
    c = np.arange(1, len(qd) + 1)
    w = (c >= 2) & (c <= 100)
    qd_w = pd.Series(qd[w]).interpolate(limit_direction='both').values
    cyc_w = c[w]
    dq = delta_q(cell)
    ir = valid_ir(cell)
    slope, _ = np.polyfit(cyc_w, qd_w, 1)
    slope_9, _ = np.polyfit(cyc_w[-10:], qd_w[-10:], 1)
    c1, q1, c2 = parse_policy(cell['policy'])
    return {
        # ΔQ(V) 계열 — 방전 곡선 모양 변화
        'dq_var_log': np.log10(np.nanvar(dq)),
        'dq_min_log': np.log10(np.abs(np.nanmin(dq))),
        'dq_mean_log': np.log10(np.abs(np.nanmean(dq))),
        'dq_skew': skew(dq, nan_policy='omit'),
        'dq_kurt': kurtosis(dq, nan_policy='omit'),
        # 방전 용량 계열 — 용량 자체의 변화
        'qd_2': qd_w[0],
        'qd_100': qd_w[-1],
        'qd_fade_pct_100': (qd_w[-1] - qd_w[0]) / qd_w[0] * 100,
        'qd_slope_2_100': slope,
        'qd_slope_91_100': slope_9,
        # 충전 조건
        'chargetime_2_6': np.nanmean(s['chargetime'][1:6]),
        'c1': c1, 'q1': q1, 'c2': c2,
        # 온도
        'tavg_mean': np.nanmean(s['Tavg'][1:100]),
        'tmax_mean': np.nanmean(s['Tmax'][1:100]),
        'tmin_mean': np.nanmean(s['Tmin'][1:100]),
        # 내부 저항 (B2 6셀은 전 구간 0 기록 → NaN, 모델 안에서 학습 데이터 중앙값으로 대체)
        'ir_2': ir[1],
        'ir_min': np.nanmin(ir[1:100]) if np.isfinite(ir[1:100]).any() else np.nan,
        'ir_diff_100_2': ir[99] - ir[1],
    }


def energy_value_curve(cells, life_by_id, grid=np.linspace(0, 1, 101)):
    """일찍 교체 손실의 기준 곡선: 남긴 수명 비율 f → 남긴 방전량 비율 v(f).
    셀마다 사이클 2~수명의 방전 용량으로 '마지막 f 구간이 전체 방전량에서 차지하는 비율'을 구하고 중앙값을 쓴다.
    (용량은 끝으로 갈수록 빨리 줄지만 1.07 → 0.88Ah 범위라 v(0.2) ≈ 0.18로 거의 고르게 줄어든다)"""
    curves = []
    for cell in cells:
        if cell['cell_id'] not in life_by_id:
            continue
        life = int(life_by_id[cell['cell_id']])
        q = pd.Series(valid_qd(cell)[1:life]).interpolate(limit_direction='both').values
        curves.append([q[int(round(len(q) * (1 - f))):].sum() / q.sum() for f in grid])
    return grid, np.median(curves, axis=0)


def build_table(cells):
    rows = []
    ref_vdlin = next(c['vdlin'] for c in cells if c['batch'] == 'b1')
    for cell in cells:
        row = {
            'batch': cell['batch'], 'cell_id': cell['cell_id'], 'policy': cell['policy'],
            'cycle_life_raw': cell['cycle_life_raw'], 'n_cycles': cell['n_cycles'],
            'status': label_status(cell), 'eol_cycle_calc': eol_cycle(cell),
        }
        if cell['n_cycles'] >= 101 and not np.isnan(cell['qdlin'][99]).all():
            row.update(cell_features(cell))
            row.update(alignment_check(cell, ref_vdlin))
            row['early_data_ok'] = early_data_ok(row)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['cycle_life'] = np.where(df['status'] == 'used', df['cycle_life_raw'], np.nan)
    df['log_life'] = np.log10(df['cycle_life'])
    return df


if __name__ == '__main__':
    df = build_table(load_cells())
    df.to_csv(PROCESSED_DIR / 'features.csv', index=False)
    print(df.groupby(['batch', 'status']).size().unstack(fill_value=0))
    u = df[df.status == 'used']
    print(u.groupby('batch')[['cycle_gaps_100', 'qdlin_ok', 'vdlin_same', 'early_data_ok']].agg(['sum', 'count']))
    print(u.groupby('batch')[['qd_jump_max', 'qdlin_offset']].median().round(4))
    print('saved', PROCESSED_DIR / 'features.csv', df.shape)

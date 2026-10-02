"""
1단계 전처리: 배치 .mat(MATLAB v7.3/HDF5) → 셀 단위 pickle + 데이터 품질 규칙.

- mat73로 통째로 읽으면 메모리를 많이 쓰므로 h5py로 필요한 값만 읽는다.
- 셀마다 저장: 메타(batch, cell_id, policy, cycle_life), 사이클별 summary,
  초기 사이클(1~MAX_CYCLE)의 Qdlin(공통 전압 눈금 1,000점 위 방전 용량), Vdlin.
- 인덱스 규칙: summary/cycles의 index j = 사이클 번호 j+1.

품질 규칙 (1일차 4장 처리표와 동일)
- 결측 · 수명 값 : cycle_life 없음 (B2 특수 충전 실험 8셀, B3 2셀) → 셀 제외 (missing)
- 수명 미확정     : 기록 마지막 5사이클 평균 용량 > 0.90Ah = EOL 전에 기록 종료 → 셀 제외 (censored)
- 측정 오류       : 방전 용량 0.8~1.2Ah 밖, 저항 ≤ 0 → 그 값만 NaN (셀은 유지)
- 수명 극단값     : 유지 (실제로 일어난 값)

사용: python src/preprocess.py   (data/raw/*.mat → data/processed/b{1,2,3}.pkl)
"""
import pickle
import sys

import h5py
import numpy as np

from config import BATCHES, CENSOR_AH, EOL_AH, MAX_CYCLE, PROCESSED_DIR, QD_VALID, RAW_DIR

SUMMARY_KEYS = ['QDischarge', 'QCharge', 'IR', 'Tavg', 'Tmax', 'Tmin', 'chargetime', 'cycle']


def _str(h, ref):
    return ''.join(chr(c) for c in np.array(h[ref]).flatten())


def extract_batch(key):
    path = next(RAW_DIR.glob(f'{BATCHES[key]}_batchdata*.mat'))
    cells = []
    with h5py.File(path, 'r') as h:
        b = h['batch']
        n_cells = b['cycle_life'].shape[0]
        for i in range(n_cells):
            cl = np.array(h[b['cycle_life'][i, 0]]).squeeze()
            summ = h[b['summary'][i, 0]]
            summary = {k: np.array(summ[k]).squeeze().astype(float) for k in SUMMARY_KEYS}
            cyc = h[b['cycles'][i, 0]]
            n_cyc = cyc['Qdlin'].shape[0]
            qdlin = np.full((MAX_CYCLE, 1000), np.nan, dtype=np.float32)   # row r = 사이클 번호 r+1
            for j in range(1, min(MAX_CYCLE, n_cyc)):
                arr = np.array(h[cyc['Qdlin'][j, 0]]).flatten()
                if arr.size == 1000:
                    qdlin[j] = arr
            cells.append({
                'batch': key,
                'cell_id': f'{key}c{i}',
                'policy': _str(h, b['policy_readable'][i, 0]),
                'cycle_life_raw': float(cl) if cl.size else np.nan,
                'n_cycles': len(summary['cycle']),
                'summary': summary,
                'qdlin': qdlin,
                'vdlin': np.array(h[b['Vdlin'][i, 0]]).flatten(),
            })
            print(f'\r{key}: {i + 1}/{n_cells}', end='', flush=True)
    print()
    return cells


def load_cells(keys=('b1', 'b2', 'b3')):
    cells = []
    for k in keys:
        with open(PROCESSED_DIR / f'{k}.pkl', 'rb') as f:
            cells += pickle.load(f)
    return cells


def valid_qd(cell):
    """측정 오류(0.8~1.2Ah 밖) 방전 용량만 NaN으로 바꾼 사이클별 QD."""
    qd = cell['summary']['QDischarge'].copy()
    qd[(qd < QD_VALID[0]) | (qd > QD_VALID[1])] = np.nan
    return qd


def valid_ir(cell):
    """측정 오류(≤ 0) 내부 저항만 NaN으로 바꾼 사이클별 IR."""
    ir = cell['summary']['IR'].copy()
    ir[ir <= 0] = np.nan
    return ir


def label_status(cell):
    """used(학습 · 평가에 사용) / missing(수명 값 없음) / censored(수명 미확정)."""
    if np.isnan(cell['cycle_life_raw']):
        return 'missing'
    qd = valid_qd(cell)
    last = qd[~np.isnan(qd)][-5:]
    if len(last) and last.mean() > CENSOR_AH:
        return 'censored'
    return 'used'


def eol_cycle(cell):
    """라벨 검증용 EOL 사이클 재계산 (제공 라벨과 119/119 일치 확인용).
    - 기록 중 0.88Ah 아래로 내려가면: 처음 내려간 사이클 번호
    - 0.88Ah 직전(마지막 5사이클 평균 ≤ 0.89Ah)에서 기록이 끝났으면: 다음 사이클(n+1)
    - 그 외(EOL 전에 실험 종료): NaN
    """
    qd = valid_qd(cell)
    idx = np.where(qd < EOL_AH)[0]
    if len(idx):
        return int(idx[0] + 1)
    last = qd[~np.isnan(qd)][-5:]
    return len(qd) + 1 if len(last) and last.mean() <= 0.89 else np.nan


if __name__ == '__main__':
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for k in sys.argv[1:] or list(BATCHES):
        cells = extract_batch(k)
        with open(PROCESSED_DIR / f'{k}.pkl', 'wb') as f:
            pickle.dump(cells, f)
        print(f'{k}: {len(cells)} cells saved')

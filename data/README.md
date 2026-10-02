# data/

원본 `.mat`(총 8.4GB)과 셀 단위 pkl은 용량 때문에 저장소에 올리지 않는다. 대신 학습 · 평가에 필요한 작은 CSV 두 개는 포함한다.

| 경로 | 내용 | 저장소 | 만드는 법 |
|---|---|---|---|
| `raw/` | Kaggle [MIT-Stanford Battery Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 `.mat` 4개 중 3개 사용 (아래 표) | 제외 | 내려받아 넣기 |
| `processed/b{1,2,3}.pkl` | 셀 단위로 필요한 값만 뽑은 파일 | 제외 | `python src/preprocess.py` |
| `processed/features.csv` | 셀 단위 피처 테이블 (139셀) | **포함** | `python src/features.py` |
| `processed/energy_curves_b1.csv` | B1 셀별 일찍 교체 손실 곡선 (남긴 수명 비율 → 남긴 방전량 비율) | **포함** | `python src/features.py` |

`src/train.py`와 `notebooks/03_modeling.ipynb`는 포함된 CSV 두 개만으로 실행된다. `01_EDA` · `02_feature_engineering` 노트북과 `features.py`는 원본에서 만든 pkl이 필요하다.

사용하는 원본 파일

| 역할 | Batch | 파일 |
|---|---|---|
| 학습 | Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |
| 테스트 | Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |
| 추가 검증 | Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |

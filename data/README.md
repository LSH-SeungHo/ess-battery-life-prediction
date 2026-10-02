# data/

원본 `.mat`(총 8.4GB)과 가공 파일은 용량 때문에 저장소에 올리지 않는다.

| 폴더 | 내용 | 만드는 법 |
|---|---|---|
| `raw/` | Kaggle [MIT-Stanford Battery Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)의 `.mat` 4개 | 내려받아 그대로 넣기 |
| `processed/b{1,2,3}.pkl` | 셀 단위로 필요한 값만 뽑은 파일 | `python src/preprocess.py` |
| `processed/features.csv` | 셀 단위 피처 테이블 (139셀) | `python src/features.py` |

사용하는 원본 파일

| 역할 | Batch | 파일 |
|---|---|---|
| 학습 | Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |
| 테스트 | Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |
| 추가 검증 | Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |

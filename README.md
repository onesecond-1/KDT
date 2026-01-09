 # ICU 입실 초기 임상정보를 이용한 패혈증 환자의 사망 위험 조기 예측

+
+MIMIC-IV 기반 ICU 입실 초기 임상정보에서 SOFA score를 계산하고,
+Sepsis-3 기준으로 패혈증을 라벨링한 뒤 사망 위험 예측 모델을 학습하는
+파이프라인을 제공합니다.
+
+## 사용 방법
+
+```bash
+pip install -r requirements.txt
+python scripts/train_mortality_model.py --input path/to/icu_early_features.csv
+```
+
+입력 데이터는 아래 컬럼을 포함해야 합니다.
+
+- 필수: `pao2`, `fio2`, `platelets`, `bilirubin`, `map`, `gcs`, `creatinine`,
+  `suspected_infection`, `mortality_28d`
+- 선택: `ventilated`, `dopamine`, `dobutamine`, `epinephrine`, `norepinephrine`,
+  `urine_output`, `subject_id`

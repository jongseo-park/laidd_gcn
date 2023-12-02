# LAIDD 교육
그래프 신경망을 활용한 분자 가상 스크리닝 - 결과물 제출

<br>

### 학습/추론 환경
```
OS: ubuntu 22.04
GPU: RTX 4070 ti
CUDA: 12.2
Torch: 2.1.0
```

<br>

실제 학습/추론에 사용한 환경
```
conda create --name laidd_gnn --file requirements.txt
```

<br>


### 파일 설명

`Best_GCN_model_v1.pt`
- 교육과정 동안 학습시킨 모델 (Enamine DB 기반 학습)

`Enamine_ADGPU_dock.csv`
- 모델 학습에 사용한 Enamine DB 의 도킹 결과 (smi, 도킹스코어)

`mcule-instock_11.smi`
- 추론용 smi 파일 (연습용 파일)

`mcule-instock_11.csv`
- 추론 결과 파일

<br>

### step_by_step 폴더 설명
`01_docking`
- ADGPU 를 통해 초기 학습용 데이터 46만개 (EnamineDB) 에 대한 도킹을 수행하는 과정에 관한 자료
- protein 압축파일 내 pdbqt 파일과 도킹결과일부 폴더 내 리간드 pdbqt 파일을 통해 도킹 결과를 visualize 할 수 있음

`02_DL_training`
- ADGPU 도킹 결과를 GCN 으로 학습시키는 과정에 관한 자료

`03_score_prediction_through_DLmodel`
- 학습시킨 DL 모델을 이용하여 mcule DB 에 대한 도킹 스코어를 예측하는 과정에 관한 자료

<br>

### Train

```
python3 laidd_gcn_train.py
```

<br>

### Inference

```
python3 inference.py
```


<br>

### 학습 결과

![result](https://github.com/jongseo-park/laidd_gcn/blob/master/img/im.png)
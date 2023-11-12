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
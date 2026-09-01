# Reconhecimento de Sinais de LIBRAS com MediaPipe + Keras

Projeto para classificar sinais de LIBRAS a partir dos landmarks (pontos
de articulação) detectados pelo MediaPipe, usando uma rede neural
treinada com Keras/TensorFlow.

> **Nota:** o projeto usa a API nova do MediaPipe (`HandLandmarker` /
> Tasks API) em vez da antiga `mediapipe.solutions`, que está quebrada
> em várias instalações recentes do MediaPipe (erro `AttributeError:
> module 'mediapipe' has no attribute 'solutions'`). Na primeira
> execução, o script baixa automaticamente o arquivo do modelo
> (`hand_landmarker.task`, ~10MB) — é preciso ter internet nessa hora.

## Como funciona

1. O **MediaPipe Hands** detecta a mão na imagem e devolve 21 pontos
   (landmarks) em 3D: pontas dos dedos, nós, pulso, etc.
2. Esses pontos são normalizados (centralizados no pulso e escalados),
   virando um vetor de 63 números.
3. Uma **rede neural densa (MLP)** aprende a classificar esse vetor
   em um gesto específico.

## Passo a passo

### 1. Instale as dependências
```bash
pip install -r requirements.txt
```

### 2. Colete os dados
```bash
python collect_data.py
```
- Ao iniciar, digite no terminal a **palavra do sinal** que vai gravar
  (ex: `oi`).
- Na janela da webcam, pressione **`r`** para ligar a gravação contínua
  (grava um frame a cada ~0.1s enquanto a mão estiver visível) — pressione
  `r` de novo para pausar. Você pode ligar/pausar quantas vezes quiser;
  cada amostra já é salva na hora, então pode parar a qualquer momento
  sem perder nada.
- Pressione **`n`** a qualquer momento para trocar de palavra. **Se você
  digitar uma palavra que já existe no dataset, as novas amostras se
  somam às que já existem para aquele sinal** (não cria um rótulo
  duplicado) — é assim que a IA vai "aprendendo mais e mais" o mesmo
  sinal ao longo de várias sessões.
- Pressione **`q`** para sair. Os dados ficam acumulados em `dataset.csv`.
- Colete **pelo menos 100-200 amostras por sinal**, variando um pouco
  o ângulo e a distância da mão — isso melhora muito a precisão.

### 3. Treine o modelo
```bash
python train_model.py
```
Isso gera `hand_gesture_model.h5` e `label_encoder.npy`.

### 4. Rode o reconhecimento em tempo real
```bash
python predict_realtime.py
```

## Dicas para melhorar a precisão

- **Quantidade de dados**: quanto mais amostras variadas (ângulos,
  iluminação, distância da câmera), melhor o modelo generaliza.
- **Gestos parecidos**: se dois gestos forem muito similares
  geometricamente, considere aumentar a camada intermediária da rede
  ou coletar ainda mais dados para esses casos específicos.
- **Data augmentation**: você pode gerar variações artificiais dos
  vetores de landmarks (pequenas rotações/ruído) para aumentar o
  dataset sem precisar gravar mais vídeo.
- **Sequências temporais**: se algum gesto depende de movimento (não
  só da pose estática), vale evoluir para uma rede recorrente (LSTM)
  que analisa uma sequência de frames em vez de um frame isolado —
  me avise se quiser essa versão.

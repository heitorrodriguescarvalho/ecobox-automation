# Ecobox — Detecção Local de Entrada de Lixo com Webcam + Classificação na Nuvem

Pipeline de detecção de entrada de lixo para uma lixeira inteligente. A
**detecção local** usa apenas **processamento clássico de imagem (OpenCV +
NumPy)** — **sem machine learning, redes neurais ou modelos de visão** no
dispositivo (CPU/RAM limitados). A **classificação do resíduo** é feita pela
**Gemini API** (nuvem, via Wi-Fi), somente depois que a detecção local
confirma um objeto estável.

Este sistema detecta quando um objeto entra na área da lixeira, espera ele
estabilizar, captura uma imagem em alta resolução, envia a ROI em baixa
resolução para o Gemini e disponibiliza o resultado estruturado (categoria +
confiança) para a camada de atuadores.

## Arquitetura

```
Webcam (aberta em alta resolução, 1 handle)
   ↓ downscale
Frame de monitoramento (640×360) @ 2–5 FPS
   ↓
Crop da ROI
   ↓
Grayscale + Gaussian Blur
   ↓
absdiff vs background (mediana de N frames)
   ↓
Threshold + (morphology open opcional)
   ↓
countNonZero → razão de pixels alterados
   ↓
StateMachine (EMPTY → … → EMPTY)
   ↓
Capture em alta resolução (1920×1080)
   ↓
Crop da ROI + resize (640×640, preservando proporção) + JPEG
   ↓
GeminiClassifier.classify(jpeg_bytes) → ClassificationResult
   ↓
sort_waste(result.category)   (interface para atuadores)
```

Módulos:

```
src/
├── camera.py          # 1 handle de câmera; read_monitor() faz downscale
├── detector.py        # background (mediana) + analyze() → razão alterada
├── state_machine.py   # Estados e transições (inclui CLASSIFYING/CLASSIFIED)
├── capture.py         # Captura em alta resolução + preparação p/ API (crop/resize/JPEG)
├── classification.py  # ClassificationResult (tipo compartilhado)
├── gemini.py          # GeminiClassifier — única camada que fala com a API
├── calibrate.py       # Calibração interativa (ROI + background)
├── debug.py           # Overlay de debug (janelas)
├── config.py          # Config (dataclass frozen + JSON)
└── main.py            # Wiring + retry + logging + sort_waste (stub)
```

### Estados da máquina

`EMPTY → POSSIBLE_OBJECT → OBJECT_DETECTED → WAITING_FOR_STABILITY → READY_TO_CAPTURE → CLASSIFYING → CLASSIFIED → WAITING_FOR_EMPTY → EMPTY`

- **EMPTY** — monitora a ROI.
- **POSSIBLE_OBJECT** — alteração acima do limiar; exige `confirm_frames`
  frames consecutivos para confirmar (reduz falsos positivos).
- **OBJECT_DETECTED** — objeto confirmado; aguarda parar de se mover.
- **WAITING_FOR_STABILITY** — imagem estável por `stability_time` (razão de
  alteração abaixo de `stability_threshold`) antes de capturar.
- **READY_TO_CAPTURE** — captura frame em alta resolução.
- **CLASSIFYING** — aguarda a resposta do Gemini (com retry). Nenhuma nova
  captura/detecção ocorre enquanto classifica.
- **CLASSIFIED** — classificação finalizada; resultado entregue ao controlador
  (`sort_waste`).
- **WAITING_FOR_EMPTY** — espera a ROI voltar a parecer com o background antes
  de permitir nova detecção.

## Configuração

Todos os parâmetros estão centralizados em `src/config.py` (dataclass frozen)
e podem ser sobrescritos por um `config.json` (gerado pela calibração). Sem
valores mágicos espalhados pelo código.

Principais parâmetros e valores padrão:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `camera_index` | `0` | índice do dispositivo de vídeo |
| `capture_width/height` | `1920×1080` | resolução da câmera / captura final |
| `monitor_width/height` | `640×360` | resolução de monitoramento |
| `monitor_fps` | `3` | FPS de monitoramento (2–5 recomendado) |
| `roi_x/y/width/height` | centro (`160,90,320,180`) | ROI em px de monitoramento |
| `diff_threshold` | `25` | limiar de diferença por pixel |
| `min_changed_ratio` | `0.02` | fração mínima de pixels alterados |
| `confirm_frames` | `3` | frames p/ confirmar objeto |
| `blur_ksize` | `5` | kernel do Gaussian blur (ímpar) |
| `morph_ksize` | `3` | kernel da abertura morfológica (0 desliga) |
| `stability_threshold` | `0.005` | razão máxima p/ considerar estável |
| `stability_time` | `0.8` | duração mínima de estabilidade (s) |
| `background_frames` | `15` | frames p/ mediana do background |
| `background_sample_interval` | `0.1` | intervalo entre amostras (s) |
| `background_path` | `background.npy` | arquivo do background |
| `captures_dir` | `captures/` | onde as capturas são salvas |
| `debug` | `false` | janelas de debug (também via `--debug`) |
| `debug_display_fps` | `10` | taxa de atualização do preview no debug (independente do FPS de detecção) |
| `gemini_model` | `gemini-2.5-flash-lite` | modelo usado na classificação |
| `gemini_timeout` | `30` | timeout por requisição (s) |
| `classification_confidence_threshold` | `0.85` | confiança mínima p/ aceitar |
| `max_classification_attempts` | `2` | máx. de tentativas por objeto (1 = sem retry) |
| `classification_retry_delay` | `1.0` | espera entre tentativas (s) |
| `classification_image_width/height` | `640×640` | imagem enviada ao Gemini (ROI cropped + letterbox) |
| `jpeg_quality` | `80` | qualidade JPEG da imagem enviada (75–85) |
| `gemini_debug` | `false` | salvar a imagem p/ revisão no gerenciador de arquivos + aprovar/recusar no terminal antes de cada requisição (ativa também o preview da câmera) |

## Instalação

Requer Python ≥ 3.11 e [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # instala opencv-python, numpy, google-genai
```

> Em produção (sem GUI / headless), troque `opencv-python` por
> `opencv-python-headless` no `pyproject.toml` e rode `uv sync` novamente.

## Calibração (ROI + background)

Rode o calibrador interativo com a webcam apontada para a **lixeira vazia**:

```bash
uv run ecobox-calibrate            # ou: uv run python -m src.calibrate
```

Na janela que abre:

1. **Arraste** o mouse sobre a área onde o lixo é colocado para desenhar a ROI
   (retângulo verde).
2. Pressione **`c`** para capturar o background (mediana de `background_frames`).
3. Pressione **`s`** para salvar a ROI (em `config.json`) e o background (em
   `background.npy`) e sair.

Outros atalhos: **`r`** redefine a ROI para o padrão centralizado, **`q`/ESC**
sai sem salvar.

Para escolher outra câmera: `uv run ecobox-calibrate --index 1`.

## Execução

```bash
uv run ecobox                      # ou: uv run python -m src.main
```

O pipeline exige um background já calibrado; caso contrário, instrui a rodar o
calibrador. Para recalibrar o background em execução (lixeira vazia):

```bash
uv run ecobox --recalibrate
```

### Debug

```bash
uv run ecobox --debug
```

Abre uma janela mostrando o feed da câmera, a ROI, o background, a máscara de
diferença, a razão de pixels alterados e o estado atual. No modo normal **nenhuma
janela é aberta** e não há consumo extra de recursos. No debug, pressione **`r`**
para recalibrar o background ao vivo e **`q`/ESC** para sair.

## Integração com a Gemini API

A classificação usa a **Gemini API** (`gemini-2.5-flash-lite`) via o SDK
oficial `google-genai`. **Nenhum modelo roda no dispositivo.**

### 1. Configurar a API key

A chave é lida da variável de ambiente `GEMINI_API_KEY` (nunca do código ou de
logs). Obtenha uma em <https://aistudio.google.com/apikey> e:

```bash
export GEMINI_API_KEY="sua-chave-aqui"
uv run ecobox
```

Se a variável estiver ausente, o programa aborta com uma mensagem clara:
`Missing GEMINI_API_KEY environment variable`.

### 2. Testar com uma imagem local (sem webcam)

```bash
uv run ecobox --image foto.jpg
```

Envia a imagem diretamente ao Gemini (usando a imagem inteira, sem crop de ROI)
e imprime `categoria<TAB>confiança`. Não abre câmera nem ativa atuadores.
Exemplos:

- Sucesso: `plastic	0.940`
- Erro de API (ex.: chave inválida): log `[ERROR] Gemini request failed: auth (...)`
  e saída com código ≠ 0.

Para logs detalhados do SDK: `uv run ecobox --debug --image foto.jpg`.

### Modo de revisão do Gemini (`--gemini-debug`)

Mantém o **preview da câmera** aberto (janela com feed + ROI) e, ao capturar um
objeto:

1. Salva a imagem que será enviada em `captures/gemini_review_<timestamp>.jpg`
   (log `[INFO] Image captured for review -> ...`).
2. Você abre a imagem no gerenciador de arquivos para inspecioná-la.
3. No terminal, responde ao prompt:

   - **`y` / `s`** — enviar ao Gemini.
   - **`n`** (ou Enter) — pular (classificação vira `unknown`, sem retry).

4. A resposta do Gemini (categoria, confiança e texto bruto) é exibida no
   terminal via `[INFO] Gemini response: ...`.

```bash
uv run ecobox --gemini-debug                # pipeline com revisão + preview da câmera
uv run ecobox --gemini-debug --image foto.jpg   # teste com imagem local
```

Também pode ser habilitado pela env `ECOBOX_GEMINI_DEBUG=1` ou por
`gemini_debug: true` no `config.json` (ambos também ativam o preview da
câmera).

### 3. Testar com a webcam

```bash
uv run ecobox                 # pipeline completo
uv run ecobox --debug         # + janelas e logs detalhados
```

Fluxo por objeto: detecção local → estabilização → captura 1080p → **crop da
ROI** → resize 640×640 (preservando proporção, com letterbox cinza) → JPEG
(qualidade `jpeg_quality`) → `GeminiClassifier.classify(jpeg)` → resultado.

### 4. Exemplo de classificação bem-sucedida

```text
[INFO] Object detected
[INFO] Object stabilized
[INFO] Sending image to Gemini (attempt 1/2)
[INFO] Classification: plastic (confidence=0.94)
[INFO] sort_waste: plastic (confidence=0.94)
[INFO] Waiting for bin to become empty
```

### 5. Tratamento de erro da API

Erros são normalizados e **nunca derrubam o processo**:

| Situação | Tipo logado (`ClassificationError.kind`) |
|---|---|
| timeout | `timeout` |
| erro HTTP | `http` |
| API indisponível (5xx) | `unavailable` |
| resposta inválida / fora do schema | `invalid_response` |
| autenticação (401/400/403) | `auth` |
| rate limit (429) | `rate_limit` |

Cada erro conta como uma tentativa. Depois de `max_classification_attempts`
(2) tentativas sem resultado aceito, o resultado vira `unknown` e o sistema
segue esperando a lixeira esvaziar:

```text
[INFO] Sending image to Gemini (attempt 1/2)
[ERROR] Gemini request failed: timeout (Gemini request timed out)
[INFO] Re-capturing for retry 2/2
[INFO] Sending image to Gemini (attempt 2/2)
[ERROR] Gemini request failed: rate_limit (rate limit exceeded (HTTP 429))
[INFO] Classification inconclusive after 2 attempt(s); treating as unknown
```

### 6. Arquitetura da integração

- `src/gemini.py` — **único** módulo que fala com o SDK (`GeminiClassifier`).
  `classify(image: bytes) -> ClassificationResult` usa `response_schema`
  (saída estruturada com enum de categorias) e `temperature=0`. Respostas são
  validadas; qualquer categoria/confiança inválida → `ClassificationError`.
- `src/classification.py` — `ClassificationResult(category, confidence, ...)`,
  o único tipo que detector/state machine consomem.
- `src/capture.py` — `prepare_classification_image()` faz crop da ROI, resize
  com proporção e encoding JPEG (configurável).
- `src/main.py` — retry (no máx. `max_classification_attempts`), threshold de
  confiança, logging e o stub de atuador `sort_waste(category, confidence)`.

### 7. Categorias aceitas

`plastic`, `paper`, `metal`, `non-recyclable`, `unknown`.

- **plastic / paper / metal** — material identificado como um dos recicláveis.
- **non-recyclable** — o modelo identifica o material, mas ele não é plástico,
  papel nem metal (ex.: vidro, orgânico, tecido, borracha).
- **unknown** — usado **somente** quando o modelo não tem certeza do material.

O schema enviado ao Gemini restringe o enum e o prompt instrui a escolher
exatamente uma categoria; respostas fora do schema são tratadas como
`unknown`.

### 8. Integrando os atuadores (servo/motor)

Substitua o stub `sort_waste` em `src/main.py` pela lógica real. Ela só é
chamada para classificações **aceitas** (confiança ≥ `classification_confidence_threshold`).

```python
def sort_waste(category: str, confidence: float) -> None:
    if category == "plastic":
        servo.angle(0)
    elif category == "paper":
        servo.angle(90)
    # ...
```

## Notas de desempenho

- A câmera é aberta **uma única vez** em alta resolução; o monitoramento usa o
  mesmo frame reduzido via `cv2.resize` (sem reabrir/reconfigurar o dispositivo).
  Com `CAP_PROP_BUFFERSIZE=1`, a leitura sempre devolve o frame mais recente.
- Todo processamento de detecção acontece em 640×360 e apenas dentro da ROI.
- O loop é limitado a `monitor_fps` frames por segundo; alta resolução só é lida
  quando um objeto é efetivamente detectado.
- No modo debug, o preview é atualizado a `debug_display_fps` (padrão 10),
  desacoplado da detecção, para o feed ficar fluido — sem aumentar o custo do
  modo normal.
- O background **nunca** é atualizado automaticamente na presença de um objeto,
  para não "aprender" o lixo como parte do fundo.

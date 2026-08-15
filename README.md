# Hamster Cam - Reconhecimento de Gestos e Expressões

Um aplicativo interativo em Python que utiliza **OpenCV** e **MediaPipe** para rastrear gestos de **1 mão**, **2 mãos** e **expressões faciais (boca)** via webcam em tempo real, exibindo automaticamente imagens customizadas sobrepostas na tela (Picture-in-Picture).

<p align="center">
  <img src="assets/feliz.jpg" alt="happy" width="300"/>
</p>

---

## 📸 Funcionalidades

- 🖐️ **Reconhecimento de Gestos de 1 Mão**: Punho fechado, apontando, paz e amor, palma aberta, positivo e negativo.
- 🤲 **Gestos de 2 Mãos Avançados**: Duplo positivo, duplo negativo, duas palmas abertas, duplo punho fechado, duplo paz e amor, pontas se tocando e Coração com as mãos.
- 😮 **Expressões Faciais com a Boca**: Boca aberta, biquinho, sorriso com boca fechada, língua para fora.
- 🎭 **Reações Combinadas**: Expressões faciais combinadas com gestos de mão.
- 🖼️ **Exibição Dinâmica (Picture-in-Picture)**: As imagens de reação da pasta `assets/` são sobrepostas diretamente na própria janela da webcam.
---

## 🛠️ Pré-requisitos e Instalação

O projeto é gerenciado usando o [uv](https://github.com/astral-sh/uv).

### 1. Clonar o repositório:
```bash
git clone https://github.com/kaynanxd/hamster_cam.git
cd hamster_cam
```

### 2. Instalar as dependências com `uv`:
```bash
uv sync
```

---

## 🚀 Como Executar

Para iniciar o aplicativo via código-fonte com a webcam:

```bash
uv run python main.py
```

ou baixe o executavel disponivel em releases no github e execute na sua maquina

---

## 🛠️ Tecnologias Utilizadas

- [Python 3.11](https://www.python.org/)
- [OpenCV](https://opencv.org/) - Captura e manipulação de vídeo.
- [MediaPipe](https://mediapipe.dev/) - Rastreamento avançado de mãos (Hands) e malha facial (Face Mesh).
- [NumPy](https://numpy.org/) - Operações geométricas e matemáticas de distância entre landmarks.

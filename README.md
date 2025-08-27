# MonReader Cognitive Engine: A Multi-Modal AI Pipeline

> **Turn a physical Shona hymnbook into an interactive, accessible, and audible experience.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#) [![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-orange)](#) [![Torch](https://img.shields.io/badge/PyTorch-%F0%9F%94%A5-red)](#)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlvinSMoyo/2XYDqXDc6wzA716j/blob/main/notebooks/monreader_cognitive_engine.ipynb)

MonReader is a six-phase, end-to-end AI system spanning **Computer Vision (CNN)**, **OCR**, **NLP**, and **Text-to-Speech**. The engine detects page turns, extracts text from images, understands cross-lingual meaning, and finally **speaks** the content in a custom-cloned Shona voice.

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#) [![Issues](https://img.shields.io/github/issues/AlvinSMoyo/2XYDqXDc6wzA716j.svg)](https://github.com/AlvinSMoyo/2XYDqXDc6wzA716j/issues) [![Last Commit](https://img.shields.io/github/last-commit/AlvinSMoyo/2XYDqXDc6wzA716j.svg)](https://github.com/AlvinSMoyo/2XYDqXDc6wzA716j/commits/main) [![Stars](https://img.shields.io/github/stars/AlvinSMoyo/2XYDqXDc6wzA716j.svg?style=social)](https://github.com/AlvinSMoyo/2XYDqXDc6wzA716j/stargazers)

## 🔭 Project Overview

The mission was to build a complete pipeline that could:

1. **Visually detect** page turns to trigger a scanner.
2. **Accurately extract** Shona text from scanned images.
3. **Analyze and align** meaning across Shona ⇄ English.
4. **Synthesize** high-quality audio in a custom-cloned Shona voice.

This was accomplished over six distinct phases, each building upon the last.

---

## 🗺️ The Six Phases of Development

### 📸 Phase 1 — Image Classification Pipeline

**Essence:** An end-to-end **image classification** project to flawlessly distinguish a page being turned (**flip**) vs held steady (**notflip**)—the trigger for scanning.

**Highlights**

* **Transfer Learning with CNNs:** A rigorous bake-off across leading architectures.
* **AlvinNet + Lion Optimizer:** A lightweight, custom CNN refined with **Lion**.
* **Performance & Explainability:** Near-perfect **F1 score**; **Grad-CAM** verified feature focus.

---

### 📖 Phase 2 — From Scanned Page to Digital Text (OCR on the Shona Hymnal)

**Essence:** A focused **OCR** challenge on high-resolution ELCZ Shona hymnbook images.

**Highlights**

* **Competitive OCR:** **EasyOCR** vs **Tesseract** head-to-head.
* **Quantitative Evaluation:** Winner selected via **Levenshtein distance**.

---

### 🧠 Phase 3 — Hybrid AI for Hymn Transcription & Analysis

**Essence:** Beyond standalone OCR—build a **Hybrid AI Pipeline** comparing **Vision-Language Models (VLMs)** against a two-step flow where a **PLM** corrects OCR.

**Highlights**

* **Modern Model Bake-Off:** VLMs reading text from images vs OCR→PLM correction.
* **Adjudication:** **Google Vision** + **GPT-4** for quality judgments.
* **Thematic & Liturgical Insight:** Pipeline scaffolding for **context**, **themes**, and **translation checks**.

---

### ↔️ Phase 4 — Cross-Lingual Alignment via Semantic Similarity

**Essence:** Add an **NLP** layer to validate translations by measuring Shona–English **semantic similarity**.

**Highlights**

* **Vector Representations:** Sentence embeddings encode meaning as vectors.
* **Distance-Based Validation:** Smaller vector distance ⇒ stronger alignment.

---

### 🗣️ Phase 5 — Case Study in Text-to-Speech (TTS) Evaluation

**Essence:** A **comparative TTS study** to choose the best base model for Shona speech synthesis.

**Highlights**

* **Comparative Evaluation:** Multiple TTS models assessed for clarity, naturalness, and pronunciation.
* **Human-Centric Judgment:** Final pick based on listening tests and usability for Shona.

---

### 🎶 Phase 6 — Enhanced Inference Conditioning for Shona Voice Cloning

**Essence:** A **training-free** strategy using **CSM-1B** with **inference-time conditioning** on a curated reference voice bank (no model retraining required).

**Highlights**

* **Prompt + Reference Bank:** Short anchor prompt + curated clips reinforce accent/timbre.
* **Rapid Prototyping:** Low-cost gains in pronunciation and style.

---

## 📦 Dataset (Phase 1)

* **Goal:** Predict whether a single image represents a **page flip** or **not flip**.
* **Primary Metric:** **F1 score** (higher is better).
* **Where to put data (Colab):** mount Drive and place images under a folder like
  `/content/drive/MyDrive/MonReader/data/phase1/`.

---

## ⚙️ How to Run (Colab-First)

This project is designed for **Google Colab** with GPU enabled.

### ✅ Prerequisites

* Google account + **Colab** access (Pro recommended)
* Project files + reference audio in **Google Drive**
* **Hugging Face** account + access token with read permissions

### 🚀 Setup & Execution

1. **Clone the repository**

   ```bash
   git clone https://github.com/AlvinSMoyo/2XYDqXDc6wzA716j.git
   cd 2XYDqXDc6wzA716j
   ```
2. **Open the Notebook in Colab**
   Click this badge/link: [Open in Colab](https://colab.research.google.com/github/AlvinSMoyo/2XYDqXDc6wzA716j/blob/main/notebooks/monreader_cognitive_engine.ipynb).
3. **Organize Your Files**
   Place voice prompts/reference clips in your Drive using the expected folder structure (see **Reference Audio** notes in the notebook).
4. **Set Colab Secrets**
   Click the **🔑 (Secrets)** icon in the left sidebar and add `HF_TOKEN` with your Hugging Face token.
5. **Run Cells in Order**

   * **Step 6.0 — Mount Drive**
   * **Step 6.1 — Environment Setup** (installs PyTorch, torchaudio, Whisper, etc.)
   * **Step 6.2 — Prepare Reference Audio** (anchor prompt)
   * **Step 6.4 — Build Reference Bank** (curated clips library)
   * **Step 6.6 — Final Generation** (baseline vs bank-boosted outputs)

> Outputs are saved to Colab and previewed inline. The notebook also prints save locations.

---

## 🧪 Run the API locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r app/requirements.txt
python app/app.py  # serves on http://127.0.0.1:8000
```

**Quick test:**

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"MonReader says hello in Shona.","mode":"bank"}'
```

> Full API docs → see **[app/README.md](app/README.md)**.

---

## 🧩 Example Folder Structure (Guidance)

```
<repo-root>/
├─ notebooks/
│  └─ monreader_cognitive_engine.ipynb
├─ csm/               # CSM-1B helper code (if used)
├─ reference_audio/   # local samples (optional; Drive path preferred)
├─ app/               # optional Flask/Gradio demo
│  ├─ app.py
│  └─ requirements.txt
├─ README.md
└─ LICENSE
```

> Note: Exact paths in the notebook point to **Google Drive**. Keep Drive and repo paths consistent.

---

## 📦 Key Dependencies

* Python **3.10+**
* **PyTorch**, **torchaudio**
* **OpenAI Whisper** (transcription)
* **Flask** / **Gradio** (optional demos)
* Plus others pinned in the notebook install cell

**Install locally (optional)**

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r app/requirements.txt  # if provided
```

---

## 🧪 Results & Notes

* Phase 1 achieved **near-perfect F1** and interpretable Grad-CAM heatmaps.
* OCR winner selected via **Levenshtein** distance.
* Hybrid (VLM vs OCR→PLM) adjudicated with **Google Vision** + **GPT-4**.
* Cross-lingual alignment validated using **sentence embeddings**.
* TTS shortlisted via listening tests; **CSM-1B** conditioning provided fast voice-clone gains without fine-tuning.

---

## 🔐 Secrets & Tokens

* Store **`HF_TOKEN`** in Colab Secrets (not in code).
* Do **not** commit tokens to Git.

---

## 📝 Citation & Acknowledgments

* Thanks to open-source communities behind **PyTorch**, **Whisper**, **CSM-1B**, and OCR libraries.
* Hymnbook content used for research and accessibility purposes.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙌 Contributing

PRs that improve docs, robustness, and language support are welcome. For larger changes, please open an issue first to discuss direction.



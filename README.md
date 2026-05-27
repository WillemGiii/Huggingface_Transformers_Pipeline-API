# Hugging Face Transformers Pipeline API 教學實作專案

## 1. 專案概述 (Project Overview)
本專案為針對自然語言處理 (Natural Language Processing, NLP) 與大型語言模型 (Large Language Model, LLM) 推理技術所設計的系統化教學實作平台。本專案以 **Hugging Face `transformers`** 函式庫為核心，重點探討高階 `pipeline` API 及其與底層組件 (包括分詞器 Tokenizer、基礎模型 Base Model、任務模型 Task-specific Model) 的協同運作機制。

本專案旨在提供嚴謹、具學術指導性的程式範例，幫助研究者與開發者掌握現代 NLP 模型推理的完整生命週期。專案內容涵蓋：
* **核心推理機制剖析**：情緒分析 (Sentiment Analysis)、文字特徵提取與序列分類推理。
* **零樣本分類 (Zero-shot Classification)**：無須微調模型即可進行動態標籤文本分類。
* **命名實體識別 (Named Entity Recognition, NER)**：實體片段切分、類別標記與彙整策略分析。

---

## 2. 核心技術原理 (Theoretical Background & Principles)

在進入具體實作前，本章節將遵循 **「先理解原理 (Why)，再進行實作 (How)** 之學術規範，深入剖析本專案涉及之核心 NLP 技術原理。

### 2.1 為什麼使用 Pipeline API (The Why)
在傳統深度學習工作流程中，要讓模型對一段文字進行推理，開發者必須手動執行分詞、文字轉換為 Token ID、張量對齊、模型正向傳播 (Forward Pass) 以及後處理 (如將 logits 轉換為機率或類別標籤)。這不僅增加了開發成本，也容易在資料維度與對齊上出錯。
`pipeline` API 是 Hugging Face 提供的最高階抽象接口，其核心價值在於**封裝了上述完整之推理生命週期**。它將「前處理 (Preprocessing) ➔ 模型推理 (Model Inference) ➔ 後處理 (Post-processing)」三個獨立階段整合為單一函數呼叫，提供開發者極致的開發效率，適合作為原型設計與基準測試 (Benchmarking) 的起點。

### 2.2 為什麼需要分詞器 Tokenizer (The Why)
深度學習模型本質上是基於線性代數與張量運算的數值計算系統，無法直接處理或理解原始字串 (String)。因此，必須透過分詞器 (Tokenizer) 作為文字與數值張量之間的橋樑。
分詞器的核心任務包含：
1. **子詞切分 (Subword Tokenization)**：使用如 BPE (Byte-Pair Encoding) 或 WordPiece 演算法，將文字切分為詞綴 (Tokens)，平衡詞表大小與未登錄詞 (Out-of-Vocabulary, OOV) 的問題。
2. **詞表映射 (Vocabulary Mapping)**：將 Token 轉換為其在預訓練詞表中的唯一數值索引 (Token ID)。
3. **序列控制 (Sequence Control)**：加入模型專屬的特殊標記 (如 BERT 的 `[CLS]`, `[SEP]` 或 RoBERTa 的 `<s>`, `</s>`)。
4. **維度對齊 (Dimension Alignment)**：透過**填充 (Padding)** 確保批次資料中的所有序列長度一致，並利用**截斷 (Truncation)** 確保序列長度不超過模型最大容量 (如 512 個 token)。同時，生成**注意力遮罩 (Attention Mask)** 以指示模型在計算 Self-Attention 時忽略填充符號 (Padding Tokens)。

### 2.3 為什麼區分 AutoModel 與 AutoModelForSequenceClassification (The Why)
在 Hugging Face 的架構中，模型被設計為模組化的雙層結構：
1. **基礎模型 (Base Model / Transformer Backbone)**：透過 `AutoModel` 載入。此模型僅包含 Transformer 的多頭自注意力機制 (Self-Attention) 與前饋神經網路層。其輸出為每個 Token 的**隱藏狀態特徵張量 (Hidden States / Last Hidden State)**，通常具有高維度 (例如 $768$ 或 $1024$)。這代表了文本在語意空間中的稠密表徵，可用於語意搜尋或特徵提取，但**不具備直接分類或預測的能力**。
2. **任務特定模型 (Task-specific Model / Model with Head)**：例如透過 `AutoModelForSequenceClassification` 載入。此模型在基礎模型的頂部附加了一個**分類頭 (Classification Head)**。分類頭通常由一至數個線性層 (Linear Layer) 與 Dropout 層組成，負責將高維度的隱藏狀態映射到目標類別的數量上，其直接輸出稱為 **Logits** (未經歸一化的實數分數)。
3. **機率映射 (Probability Mapping)**：Logits 數值範圍為負無窮大至正無窮大，無法直接作為機率解讀。因此，必須引入 **Softmax** 函數：
   $$P(y_i | x) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
   將 Logits $z$ 轉換為區間在 $[0, 1]$ 且總和為 $1$ 的標準機率分佈。

### 2.4 為什麼 Zero-shot Classification 能在無樣本下進行分類 (The Why)
傳統文本分類 (Supervised Classification) 要求必須針對特定類別標籤收集大量標註資料並進行模型微調。
**零樣本分類 (Zero-shot Classification)** 突破了此限制，其核心原理是基於**自然語言推理 (Natural Language Inference, NLI)**。NLI 任務旨在判斷兩個句子 (前提 Premise 與假設 Hypothesis) 之間的語意關係，類別為：
* **蘊含 (Entailment)**：前提成立，假設必然成立。
* **矛盾 (Contradiction)**：前提成立，假設必然不成立。
* **中立 (Neutral)**：兩者無直接關係。

在執行 Zero-shot 分類時，系統會自動重構輸入：
* **前提 (Premise)**：使用者欲分類的文字 $X$。
* **假設模板 (Hypothesis Template)**：例如 `"This text is about {candidate_label}."`，並將候選標籤依序帶入。
接著，模型會計算每個「前提-假設對」在 NLI 模型中輸出為**蘊含 (Entailment)** 的機率值。透過對所有候選標籤的蘊含機率進行歸一化，即可在不需要任何微調的情況下，動態實現語意層面的文本分類。

### 2.5 命名實體識別 NER 的片段彙整策略 (The Why)
**命名實體識別 (Named Entity Recognition, NER)** 是序列標記 (Sequence Labeling) 的典型應用，旨在對文本中的每個 Token 標記其所屬的實體類別 (如人名 PER、組織 ORG、地點 LOC)。
然而，由於分詞器採用子詞切分演算法，一個完整的單字或專有名詞 (例如 `"Taipei"` 或 `"Tim Cook"`) 經常會被拆解為多個 Subwords (例如 `"Tai"`, `"#pei"`)。如果直接輸出 Token 等級的預測結果，將會導致實體碎片化，難以供下游應用使用。
因此，需要引入**彙整策略 (Aggregation Strategy)** (如 `aggregation_strategy="simple"`)。其原理是根據模型預測的 B- (Begin) 和 I- (Inside) 標記規範，將屬於同一個語意實體的多個連續 Subwords 片段進行重新對齊與合併，並對其置信度分數進行加權平均，最終輸出結構完整、易於閱讀的實體區間 (Entity Groups)。

---

## 3. 專案目錄與文件結構 (Directory Structure)

本專案目錄採用清晰且高模組化的結構設計，方便使用者循序漸進地學習：

```directory
generate-readme-traditional-chinese/
│
├── requirements.txt                         # 核心相依套件清單
├── huggingface_demo.py                      # 模組化且具強型別之 Python 教學示範腳本
├── huggingface_pipeline.ipynb               # 互動式 Jupyter Notebook 實作示範
│
├── module-transformers-pipeline/            # 主題式深度學習實戰模組
│   ├── .env                                 # 環境變數設定檔 (內含 Hugging Face API 認證憑證)
│   ├── .gitignore                           # Git 忽略配置文件
│   ├── 01_Zero-shot Classification Pipeline.ipynb   # 零樣本分類專題教學 Notebook
│   └── 02_Named Entity Recognition Pipeline.ipynb  # 命名實體識別專題教學 Notebook
```

* **`requirements.txt`**：明確列出專案運行所需的套件，包括 `transformers`、`torch` 與環境變數管理套件 `python-dotenv`。
* **`huggingface_demo.py`**：完全符合 **PEP 8** 規範、使用**強型別提示 (Type Hinting)** 與 **Google Style Docstrings** 的高品質 Python 程式碼，適合生產環境與教學範例。
* **`module-transformers-pipeline/`**：收錄了更具針對性的 NLP 任務教學。此資料夾配備了 `.env` 檔案以支援 Hugging Face 安全驗證，保護敏感的 API 金鑰不被外洩。

---

## 4. 環境與硬體規格適應性說明 (Hardware & Environment Guide)

為了確保專案運行的穩定性，以下針對本機之硬體硬體配置與效能定位進行嚴謹說明：

### 4.1 硬體規格分析
* **主機名稱**：DESKTOP-MB842R1
* **處理器 (CPU)**：Intel Core i7-1355U (13th Gen) @ 1.70 GHz (具備優異的單核心與多執行緒 CPU 推理能力)
* **記憶體 (RAM)**：32.0 GB (31.7 GB 可用，充足的記憶體容許同時載入多個中大型 Transformer 模型)
* **硬碟**：1TB NVMe SSD (Samsung PM9B1，高速隨機讀寫速度大幅縮短模型權重載入時間)
* **顯示卡 (GPU)**：Intel Iris Xe Graphics (128 MB 獨立 VRAM，此為**整合型顯示卡**)

### 4.2 硬體執行策略與限制 (Hardware Strategy)
> [!IMPORTANT]
> 由於本機配置之 **Intel Iris Xe Graphics** 為處理器整合之顯示晶片，其專用顯示記憶體 (VRAM) 僅約為 128 MB，不具備 NVIDIA CUDA 硬體加速生態系。
> * **不建議執行**：大型語言模型的深度監督式訓練 (Supervised Fine-Tuning, SFT) 或需要極大 VRAM 的 CUDA 訓練任務。
> * **最佳執行策略**：本專案所有教學與推理任務皆建議運行於 **CPU 模式**。得益於強大的 13 代 Core i7 處理器與 32 GB 超大記憶體，執行如 DistilBERT、BART-large 或 BERT-large 等模型之**推理任務 (Inference)** 時，速度依然極為優異，且不會遇到 VRAM 溢出 (Out of Memory, OOM) 之風險。

---

## 5. 安裝與快速開始 (Installation & Quick Start)

本指南引導您於 Windows 11 的 PowerShell 環境中，使用符合現代 Python 規範的 `pathlib` 跨平台路徑管理機制來建置與運行本專案。

### 步驟 1: 複製或建立專案目錄
開啟 PowerShell 並切換至您的工作目錄。

### 步驟 2: 建立與啟用虛擬環境
為避免全域 Python 環境套件衝突，建議針對本專案建立專屬虛擬環境：
```powershell
# 建立名為 .venv 的虛擬環境
python -m venv .venv

# 在 Windows PowerShell 中啟用虛擬環境
.\.venv\Scripts\Activate.ps1
```

### 步驟 3: 安裝相依套件
升級 pip 並安裝 `requirements.txt` 中指定的函式庫：
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 步驟 4: 配置環境變數
在 `module-transformers-pipeline/.env` 中確認或填入您的 Hugging Face 存取權杖 (HF Token)，以確保能順利下載某些受版權保護或需要驗證的模型：
```env
HF_TOKEN = your_huggingface_token_here
```
> [!WARNING]
> 請注意：環境變數 `.env` 檔案內包含您的私密 Token，請勿將其提交至公開的 Git 儲存庫。本專案已在 `.gitignore` 中將其排除。

### 步驟 5: 執行示範腳本
執行主示範程式以驗證環境建置成功：
```powershell
python huggingface_demo.py
```

---

## 6. 核心程式碼深度解析 (Code Walkthrough)

以下擷取 `huggingface_demo.py` 中具代表性的模組化設計，展示如何嚴謹地實現一個完整的 Tokenizer 與任務模型推理流程。

### 6.1 分詞與特徵準備
此函數展示了如何使用 `pathlib` 與強型別宣告載入 Tokenizer，並將原始文本轉換為 PyTorch 張量字典。

```python
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer

def run_tokenizer_demo(checkpoint: str) -> Dict[str, torch.Tensor]:
    """示範使用 AutoTokenizer 處理原始文本資料。

    原理 (Why):
        機器無法直接理解文字，必須先將文字轉換為數字索引 (Token IDs)。
        Tokenizer 負責將字串切分為詞綴 (Tokens)，並加上模型所需的特殊標記，
        以及處理內建的填充 (Padding) 與截斷 (Truncation) 操作。
        輸入轉為 PyTorch 張量才能送入模型中計算。

    Args:
        checkpoint: 預訓練模型的名稱或路徑。

    Returns:
        包含 'input_ids' 與 'attention_mask' 的 PyTorch 張量字典。
    """
    print("--- 2. 使用 AutoTokenizer 處理文本 ---")
    
    # 載入對應之分詞器
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # 準備去識別化且客觀之示範資料
    raw_inputs: List[str] = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    
    # 進行編碼，啟用填充與截斷，並指定回傳 PyTorch 張量
    inputs: Dict[str, torch.Tensor] = tokenizer(
        raw_inputs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    return inputs
```

### 6.2 具備分類頭之模型推理與機率後處理
此函數展示了如何載入分類模型、執行正向傳播取得 Logits，並嚴謹地應用 Softmax 進行後處理轉換。

```python
from transformers import AutoModelForSequenceClassification

def run_model_inference_demo(checkpoint: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """示範使用 AutoModelForSequenceClassification 進行模型推理。

    原理 (Why):
        對於分類任務，我們需要模型附帶分類頭 (Classification Head)。因此，使用
        `AutoModelForSequenceClassification` 載入具有線性分類層的模型，
        以直接獲得未經 Softmax 處理的分類分數 (Logits)。接著使用 Softmax 
        將實數範圍的 Logits 映射至 [0, 1] 區間，形成標準機率分佈。

    Args:
        checkpoint: 預訓練模型的名稱或路徑.
        inputs: 由 Tokenizer 處理後的張量特徵.

    Returns:
        經過 Softmax 轉換後各類別的機率分佈張量.
    """
    print("--- 4. 使用 AutoModelForSequenceClassification 進行推理 ---")
    
    # 載入附帶序列分類頭的模型
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    
    # 使用 ** 運算子解包張量字典，送入模型中進行正向計算
    outputs = model(**inputs)
    
    # 應用 Softmax 函數將 Logits 轉換為機率值 (Probabilities)
    predictions: torch.Tensor = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return predictions
```

---

## 7. 驗證與測試 (Verification Plan)

為確保專案程式碼之正確性，本專案建議遵循以下驗證方案：

### 7.1 自動化驗證 (Automated Verification)
請在虛擬環境啟用狀態下，於 PowerShell 中執行以下指令：
```powershell
# 1. 執行基礎推理流驗證
python huggingface_demo.py
```
預期輸出中應包含：
* 情緒分析分類預測置信度。
* Tokenizer 編碼後之 `input_ids` 與 `attention_mask` 的整數張量。
* 模型推理輸出的 Logits 矩陣與經 Softmax 後的機率百分比數值。

### 7.2 互動式教材驗證
啟動 Jupyter Lab 或 Jupyter Notebook：
```powershell
pip install jupyterlab
jupyter lab
```
依序打開並執行：
1. `huggingface_pipeline.ipynb`
2. `module-transformers-pipeline/01_Zero-shot Classification Pipeline.ipynb`
3. `module-transformers-pipeline/02_Named Entity Recognition Pipeline.ipynb`

確認每個儲存格 (Cells) 皆能無錯誤執行，且能正確下載並載入 `facebook/bart-large-mnli` 與 `dbmdz/bert-large-cased-finetuned-conll03-english` 模型，輸出正確的分類機率與實體彙整結果。

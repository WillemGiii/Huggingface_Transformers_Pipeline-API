"""
Hugging Face Transformers 教學示範模組 (Teaching Demonstration for Hugging Face Transformers)

此模組展示了如何使用 Hugging Face 的 `transformers` 函式庫建立基本的自然語言處理 (NLP) 流程，
包含直接使用 `pipeline`，以及手動透過 `AutoTokenizer` 與 `AutoModelForSequenceClassification`
進行模型推理。

依賴套件 (Dependencies):
    - transformers
    - torch
"""

from typing import List, Dict, Any
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers.pipelines.base import Pipeline


def run_pipeline_demo() -> List[Dict[str, Any]]:
    """
    示範使用預先封裝好的 pipeline 進行情緒分析 (Sentiment Analysis)。

    原理 (Why):
        `pipeline` 是 transformers 庫中最高階的 API，隱藏了底層的分詞 (Tokenization)
        與模型推理 (Inference) 細節，讓發展者能以最少的程式碼完成 NLP 任務。

    回傳值 (Returns):
        List[Dict[str, Any]]: 包含每個文本情緒預測結果的字典列表。
    """
    print("--- 1. 使用 Pipeline 進行情緒分析 ---")
    
    # 初始化情緒分析 pipeline
    classifier: Pipeline = pipeline("sentiment-analysis")
    texts: List[str] = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    
    # 執行預測
    results: List[Dict[str, Any]] = classifier(texts)
    print(f"預測結果: {results}\n")
    return results


def run_tokenizer_demo(checkpoint: str) -> Dict[str, torch.Tensor]:
    """
    示範使用 AutoTokenizer 處理原始文本資料。

    原理 (Why):
        機器無法直接理解文字，必須先將文字轉換為數字索引 (Token IDs)。
        Tokenizer 負責將字串切分為詞綴 (Tokens)，並加上模型所需的特殊標記，
        以及處理內建的填充 (Padding) 與截斷 (Truncation) 操作。輸入轉為 
        PyTorch 張量才能送入模型中計算。

    參數 (Args):
        checkpoint (str): 預訓練模型的名稱或路徑。

    回傳值 (Returns):
        Dict[str, torch.Tensor]: 處理後的張量字典，包含 input_ids 與 attention_mask。
    """
    print("--- 2. 使用 AutoTokenizer 處理文本 ---")
    
    # 根據 checkpoint 載入對應的分詞器
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_inputs: List[str] = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    
    # 進行分詞處理，設定 padding 與 truncation，並指定輸出為 PyTorch Tensor ("pt")
    inputs: Dict[str, torch.Tensor] = tokenizer(
        raw_inputs, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    print(f"處理後的輸入特徵 (Inputs): \n{inputs}\n")
    return inputs


def run_base_model_demo(checkpoint: str) -> None:
    """
    示範使用 AutoModel 載入基礎模型。

    原理 (Why):
        `AutoModel` 會載入不帶任何特定任務輸出頭 (Head) 的基礎模型。
        這通常用於獲取文本的隱藏狀態特徵 (Hidden States)，而非直接得出分類結果。

    參數 (Args):
        checkpoint (str): 預訓練模型的名稱或路徑。
    """
    print("--- 3. 載入 AutoModel 基礎架構 ---")
    
    # 載入基礎模型
    model = AutoModel.from_pretrained(checkpoint)
    print(f"已成功載入基礎模型: {model.__class__.__name__}\n")


def run_model_inference_demo(checkpoint: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    示範使用 AutoModelForSequenceClassification 進行模型推理。

    原理 (Why):
        對於分類任務，我們需要模型附帶分類頭 (Classification Head)。因此，使用
        `AutoModelForSequenceClassification` 載入具有線性分類層的模型，
        以直接獲得未經 Softmax 處理的分類分數 (Logits)。接著使用 Softmax 
        將實數範圍的 Logits 映射至 [0, 1] 區間，形成標準機率分佈。

    參數 (Args):
        checkpoint (str): 預訓練模型的名稱或路徑。
        inputs (Dict[str, torch.Tensor]): 由 Tokenizer 處理後的張量特徵。

    回傳值 (Returns):
        torch.Tensor: 經過 Softmax 轉換後各類別的機率分佈張量。
    """
    print("--- 4. 使用 AutoModelForSequenceClassification 進行推理 ---")
    
    # 載入附帶序列分類頭的模型
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    
    # 使用 ** 運算子解包字典，將 input_ids 與 attention_mask 傳入模型中進行推理
    outputs = model(**inputs)
    
    print(f"模型輸出 Logits: \n{outputs.logits}\n")

    # [新增] 應用 Softmax 函數將 Logits 轉換為機率值 (Probabilities)
    predictions: torch.Tensor = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    print(f"預測機率值 (Probabilities): \n{predictions}\n")
    return predictions


def main() -> None:
    """
    主程式進入點，循序執行各項 Hugging Face 教學示範。
    """
    checkpoint: str = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # 1. 執行 Pipeline 範例
    run_pipeline_demo()
    
    # 2. 執行 Tokenizer 範例
    inputs: Dict[str, torch.Tensor] = run_tokenizer_demo(checkpoint)
    
    # 3. 執行 Base Model 範例
    run_base_model_demo(checkpoint)
    
    # 4. 執行 Sequence Classification Model 推理範例
    predictions: torch.Tensor = run_model_inference_demo(checkpoint, inputs)


if __name__ == "__main__":
    main()

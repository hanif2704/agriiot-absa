import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL2ID = {"Positif": 0, "Netral": 1, "Negatif": 2}
ID2LABEL = {0: "Positif", 1: "Netral", 2: "Negatif"}

ASPEK_LIST = [
    "Manfaat",
    "Kemudahan",
    "Sosial",
    "Dukungan Infrastruktur",
]

_model = None
_tokenizer = None


def load_model(model_source: str):
    """Muat model dari folder lokal atau repo ID Hugging Face Hub."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        low_cpu_mem_usage=True,
    )
    if device.type == "cpu":
        _model = torch.quantization.quantize_dynamic(
            _model, {torch.nn.Linear}, dtype=torch.qint8
        )
    _model.to(device)
    _model.eval()
    return _model, _tokenizer


def predict_sentiment(text_preprocessed: str, aspek: str, model_source: str,
                      max_length: int = 512) -> dict:
    model, tokenizer = load_model(model_source)
    device = next(model.parameters()).device

    aspek_text = f"aspek: {aspek}"

    encoding = tokenizer(
        text_preprocessed,
        aspek_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        predicted_id = int(np.argmax(probabilities))

    return {
        "aspek": aspek,
        "sentimen": ID2LABEL[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "prob_positif": float(probabilities[0]),
        "prob_netral": float(probabilities[1]),
        "prob_negatif": float(probabilities[2]),
    }


def predict_multi_aspek(text_preprocessed: str, model_source: str) -> list:
    results = []
    for aspek in ASPEK_LIST:
        results.append(predict_sentiment(text_preprocessed, aspek, model_source))
    return results

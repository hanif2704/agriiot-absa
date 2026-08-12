import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

LABEL2ID = {"Positif": 0, "Netral": 1, "Negatif": 2}
ID2LABEL = {0: "Positif", 1: "Netral", 2: "Negatif"}

ASPEK_LIST = [
    "Manfaat",
    "Kemudahan",
    "Sosial",
    "Dukungan Infrastruktur",
]

ONNX_FILENAME = "onnx/model_quant.onnx"

_tokenizer = None
_session = None


def load_model(model_source: str):
    """Muat tokenizer + model ONNX dari HF Hub (ringan, tanpa PyTorch)."""
    global _tokenizer, _session

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_source)

    if _session is None:
        onnx_path = hf_hub_download(repo_id=model_source, filename=ONNX_FILENAME)
        _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    return _tokenizer, _session


def predict_sentiment(text_preprocessed: str, aspek: str, model_source: str,
                      max_length: int = 512) -> dict:
    tokenizer, session = load_model(model_source)

    aspek_text = f"aspek: {aspek}"
    encoding = tokenizer(
        text_preprocessed,
        aspek_text,
        truncation=True,
        max_length=max_length,
    )

    input_names = [inp.name for inp in session.get_inputs()]
    feed = {
        name: np.array([encoding[name]], dtype=np.int64)
        for name in input_names
        if name in encoding
    }

    logits = session.run(None, feed)[0][0]
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / exp.sum()
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
    return [
        predict_sentiment(text_preprocessed, aspek, model_source)
        for aspek in ASPEK_LIST
    ]

import re

NORMALIZATION_DICT = {
    "bgt": "banget", "bgtt": "banget", "bangett": "banget",
    "tp": "tapi", "tpi": "tapi", "tdk": "tidak", "gak": "tidak", "syg": "sayang",
    "ga": "tidak", "nggak": "tidak", "enggak": "tidak", "gk": "tidak",
    "jg": "juga", "jgn": "jangan", "sy": "saya", "aq": "aku",
    "gw": "saya", "gue": "saya", "lu": "kamu", "km": "kamu",
    "dgn": "dengan", "utk": "untuk", "dlm": "dalam", "krn": "karena",
    "karna": "karena", "sm": "sama", "aja": "saja", "aj": "saja",
    "nih": "ini", "ni": "ini", "mantab": "mantap", "mantul": "mantap",
    "canggihg": "canggih", "iotnya": "iot", "teknologinya": "teknologi",
    "alatnya": "alat",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_folding(text: str) -> str:
    return str(text).lower()


def normalize_text(text: str) -> str:
    tokens = str(text).split()
    normalized = []
    for token in tokens:
        if token in NORMALIZATION_DICT:
            replacement = NORMALIZATION_DICT[token]
            if replacement != "":
                normalized.append(replacement)
        else:
            normalized.append(token)
    return " ".join(normalized)


def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = case_folding(text)
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

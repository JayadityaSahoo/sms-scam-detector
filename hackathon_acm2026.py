import os
import re
import sqlite3
import unicodedata
import urllib.request

import numpy as np
from flask import Flask, request, jsonify

# Tech Stack Libraries mapped directly to flowchart stages
import fasttext
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)


# ==========================================
# INITIALIZATION & SETUP
# ==========================================

# ------------------------------------------
# 1. STAGE 2: SQLite Setup
# ------------------------------------------

def init_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE known_scams "
        "(id INTEGER PRIMARY KEY, pattern TEXT)"
    )

    cursor.executemany(
        "INSERT INTO known_scams (pattern) VALUES (?)",
        [
            (
                "aapka bank account suspend ho gaya hai "
                "turant kyc update karein",
            ),
            (
                "urgent verify your account to stop block click link",
            ),
            (
                "you won lottery claim cash reward now",
            ),
        ],
    )

    conn.commit()
    return conn


db_conn = init_db()


# ------------------------------------------
# 2. STAGE 1: FastText Language Model
# ------------------------------------------

FASTTEXT_MODEL = "lid.176.bin"
FASTTEXT_URL = (
    "https://dl.fbaipublicfiles.com/"
    "fasttext/supervised-models/lid.176.bin"
)


def load_fasttext_model():
    """
    Download the FastText language model automatically
    if it is not already present.
    """

    if not os.path.exists(FASTTEXT_MODEL):
        print("FastText model not found.")
        print("Downloading lid.176.bin...")

        try:
            urllib.request.urlretrieve(
                FASTTEXT_URL,
                FASTTEXT_MODEL
            )
            print("FastText model downloaded successfully.")

        except Exception as e:
            print("FastText download failed:", e)
            return None

    try:
        model = fasttext.load_model(FASTTEXT_MODEL)
        print("FastText model loaded successfully.")
        return model

    except Exception as e:
        print("FastText model could not be loaded:", e)
        return None


ft_model = load_fasttext_model()


# ------------------------------------------
# 3. STAGE 3 PATH B: mT5
# ------------------------------------------

print("Loading mT5 model...")

mt5_tokenizer = AutoTokenizer.from_pretrained(
    "google/mt5-small"
)

mt5_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-small"
)

print("mT5 model loaded successfully.")


# ------------------------------------------
# 4. STAGE 4: Sentence Transformers
# ------------------------------------------

print("Loading Sentence Transformer model...")

embedder = SentenceTransformer(
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Sentence Transformer loaded successfully.")


# ------------------------------------------
# 5. STAGE 5: Risk Classifier
# ------------------------------------------

clf = GradientBoostingClassifier()

# Features:
# [semantic_agreement,
#  scam_similarity,
#  intent,
#  urgency,
#  obfuscation_level,
#  url_metadata]

X_train = np.array([
    [0.15, 0.10, 0.1, 0.1, 0.0, 0],
    [0.55, 0.50, 0.6, 0.5, 0.4, 1],
    [0.92, 0.88, 0.9, 0.9, 0.8, 1],
])

y_train = np.array([
    0,  # Low
    1,  # Suspicious
    2,  # High
])

clf.fit(X_train, y_train)


# ==========================================
# PIPELINE STAGES
# ==========================================


# ==========================================
# STAGE 1: BASIC MESSAGE ANALYSIS
# ==========================================

def stage1_basic_analysis(raw_text: str) -> dict:

    scripts = set()

    for char in raw_text:

        if char.isalpha():

            name = unicodedata.name(char, "")

            if "DEVANAGARI" in name:
                scripts.add("Devanagari")

            elif "LATIN" in name:
                scripts.add("Latin")

    # Default language
    lang_hint = "en"

    # FastText language detection
    if ft_model:

        try:

            labels, probabilities = ft_model.predict(
                raw_text.replace("\n", " "),
                k=1
            )

            if labels:
                lang_hint = labels[0].replace(
                    "__label__",
                    ""
                )

        except Exception:
            pass

    # URL extraction
    urls = re.findall(
        r"https?://[^\s]+|bit\.ly/[^\s]+",
        raw_text
    )

    # Email extraction
    emails = re.findall(
        r"[\w\.-]+@[\w\.-]+",
        raw_text
    )

    # Phone extraction
    phones = re.findall(
        r"\+?\d{10,12}",
        raw_text
    )

    return {
        "scripts": list(scripts),
        "lang_hint": lang_hint,
        "extracted_urls": urls,
        "extracted_emails": emails,
        "extracted_phones": phones,
        "has_url_email_phone": bool(
            urls or emails or phones
        )
    }


# ==========================================
# STAGE 2: KNOWN PATTERN SEARCH
# ==========================================

def stage2_known_pattern_search(
    raw_text: str
) -> tuple[bool, float, str]:

    cursor = db_conn.cursor()

    cursor.execute(
        "SELECT pattern FROM known_scams"
    )

    patterns = [
        row[0]
        for row in cursor.fetchall()
    ]

    best_score = 0.0
    matched_pattern = ""

    normalized_text = raw_text.lower().strip()

    for pattern in patterns:

        score = fuzz.ratio(
            normalized_text,
            pattern
        )

        if score > best_score:

            best_score = score
            matched_pattern = pattern

    is_known = best_score >= 85.0

    return (
        is_known,
        best_score,
        matched_pattern
    )


# ==========================================
# STAGE 3 - PATH A
# OBFUSCATION ENGINE
# ==========================================

def path_a_obfuscation_engine(
    raw_text: str
) -> dict:

    # Character substitutions
    char_map = {
        "@": "a",
        "4": "a",
        "3": "e",
        "0": "o",
        "$": "s",
        "!": "i"
    }

    normalized = raw_text.lower()

    for symbol, letter in char_map.items():

        normalized = normalized.replace(
            symbol,
            letter
        )

    # Collapse repeated characters
    normalized = re.sub(
        r"(.)\1+",
        r"\1",
        normalized
    )

    # Scam-related keywords
    keywords = [
        "bank",
        "account",
        "freeze",
        "paisa",
        "turant",
        "verify",
        "update",
        "kyc"
    ]

    recovered_words = []

    for word in normalized.split():

        # Remove punctuation around words
        clean_word = re.sub(
            r"[^\w]",
            "",
            word
        )

        for keyword in keywords:

            if (
                fuzz.ratio(
                    clean_word,
                    keyword
                ) > 75
                or keyword in clean_word
            ):

                if keyword not in recovered_words:
                    recovered_words.append(keyword)

    obfuscation_score = min(
        len(recovered_words) * 0.25,
        1.0
    )

    return {
        "output_a": normalized,
        "recovered_keywords": recovered_words,
        "obfuscation_level": round(
            obfuscation_score,
            2
        )
    }


# ==========================================
# STAGE 3 - PATH B
# mT5 SEMANTIC ANALYSIS
# ==========================================

def path_b_mt5_semantic_analysis(
    raw_text: str
) -> str:

    try:

        inputs = mt5_tokenizer(
            "summarize: " + raw_text,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )

        outputs = mt5_model.generate(
            **inputs,
            max_length=30
        )

        meaning = mt5_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return (
            meaning
            if meaning
            else raw_text
        )

    except Exception as e:

        print(
            "mT5 semantic analysis failed:",
            e
        )

        return raw_text


# ==========================================
# STAGE 4: SEMANTIC COMPARISON
# ==========================================

def stage4_semantic_comparison(
    output_a: str,
    output_b: str
) -> float:

    emb_a = embedder.encode(
        output_a,
        convert_to_numpy=True
    )

    emb_b = embedder.encode(
        output_b,
        convert_to_numpy=True
    )

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(
        emb_a,
        emb_b
    ) / (norm_a * norm_b)

    # Keep value within expected range
    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return float(similarity)


# ==========================================
# STAGE 5: COMBINED RISK ANALYSIS
# ==========================================

def stage5_combined_risk_analysis(
    s1_res: dict,
    path_a_res: dict,
    semantic_agreement: float
) -> dict:

    scam_sim = semantic_agreement

    # Intent score
    intent_score = (
        0.8
        if path_a_res["recovered_keywords"]
        else 0.2
    )

    # Urgency score
    output_a = path_a_res["output_a"]

    urgency_score = (
        0.9
        if (
            "turant" in output_a
            or "urgent" in output_a
            or "immediately" in output_a
        )
        else 0.3
    )

    # Obfuscation level
    obf_level = path_a_res[
        "obfuscation_level"
    ]

    # URL / email / phone metadata
    url_meta = (
        1.0
        if s1_res["has_url_email_phone"]
        else 0.0
    )

    features = np.array([[
        semantic_agreement,
        scam_sim,
        intent_score,
        urgency_score,
        obf_level,
        url_meta
    ]])

    # Get classifier probabilities
    risk_probs = clf.predict_proba(
        features
    )[0]

    # Calculate final score
    final_score = int(
        risk_probs[1] * 40
        + risk_probs[2] * 100
    )

    final_score = min(
        max(final_score, 5),
        99
    )

    # Decision routing
    if final_score <= 24:

        category = "LOW RISK"
        action = "ALLOW"

    elif final_score <= 49:

        category = "SUSPICIOUS"
        action = "FLAG"

    else:

        category = "HIGH RISK"
        action = "ZERO-TRUST"

    return {

        "risk_score": final_score,

        "category": category,

        "action": action,

        "evaluated_features": {

            "semantic_agreement": round(
                semantic_agreement,
                2
            ),

            "scam_similarity": round(
                scam_sim,
                2
            ),

            "intent": intent_score,

            "urgency": urgency_score,

            "obfuscation_level": obf_level,

            "url_metadata": url_meta
        }
    }


# ==========================================
# ROUTE ENDPOINT
# ==========================================

@app.route(
    "/api/v1/analyze",
    methods=["POST"]
)
def analyze():

    # Safely receive JSON
    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "error": "Invalid JSON request"
        }), 400

    raw_text = data.get(
        "message",
        ""
    )

    if not isinstance(
        raw_text,
        str
    ):

        return jsonify({
            "error": "Message must be a string"
        }), 400

    if not raw_text.strip():

        return jsonify({
            "error": "Empty message"
        }), 400

    # --------------------------------------
    # Stage 1
    # --------------------------------------

    s1_res = stage1_basic_analysis(
        raw_text
    )

    # --------------------------------------
    # Stage 2
    # --------------------------------------

    (
        is_known,
        match_score,
        matched_pattern
    ) = stage2_known_pattern_search(
        raw_text
    )

    # --------------------------------------
    # Known scam short-circuit
    # --------------------------------------

    if is_known:

        return jsonify({

            "stage1_basic_analysis": s1_res,

            "stage2_known_pattern_search": {

                "route": "KNOWN PATTERN",

                "matched_pattern":
                    matched_pattern,

                "confidence":
                    round(match_score, 2)
            },

            "final_decision": {

                "risk_score": 98,

                "category": "HIGH RISK",

                "action": "ZERO-TRUST"
            }
        })

    # --------------------------------------
    # Stage 3
    # Parallel Analysis
    # --------------------------------------

    path_a_res = (
        path_a_obfuscation_engine(
            raw_text
        )
    )

    output_b_meaning = (
        path_b_mt5_semantic_analysis(
            raw_text
        )
    )

    # --------------------------------------
    # Stage 4
    # --------------------------------------

    semantic_agreement = (
        stage4_semantic_comparison(
            path_a_res["output_a"],
            output_b_meaning
        )
    )

    # --------------------------------------
    # Stage 5
    # --------------------------------------

    risk_summary = (
        stage5_combined_risk_analysis(
            s1_res,
            path_a_res,
            semantic_agreement
        )
    )

    # --------------------------------------
    # Final Response
    # --------------------------------------

    return jsonify({

        "stage1_basic_analysis": s1_res,

        "stage2_known_pattern_search": {

            "route":
                "UNKNOWN / UNCERTAIN"
        },

        "stage3_parallel_analysis": {

            "path_a_obfuscation":
                path_a_res,

            "path_b_mt5_semantic": {

                "output_b":
                    output_b_meaning
            }
        },

        "stage4_semantic_comparison": {

            "semantic_agreement":
                round(
                    semantic_agreement,
                    3
                )
        },

        "stage5_risk_analysis":
            risk_summary
    })


# ==========================================
# APPLICATION START
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
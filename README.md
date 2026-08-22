# 🛡️ Multilingual SMS Scam Detector

An AI-powered SMS scam and phishing detection system designed to detect suspicious messages, including multilingual and obfuscated scam attempts.

## 🚀 Features

- Multilingual language detection using FastText
- Known scam pattern matching using SQLite and RapidFuzz
- Obfuscated text detection and normalization
- AI-based semantic analysis using mT5
- Semantic similarity comparison using Sentence Transformers
- Machine learning-based risk analysis using Scikit-learn

## ⚠️ Risk Classification

- LOW RISK → ALLOW
- SUSPICIOUS → FLAG
- HIGH RISK → ZERO-TRUST

## 🏗️ Detection Pipeline

1. Basic Message Analysis
2. Known Pattern Search
3. Parallel Analysis
   - Path A: Obfuscation Detection
   - Path B: Semantic Analysis
4. Semantic Comparison
5. Combined Risk Analysis

## 🛠️ Tech Stack

- Python
- Flask
- FastText
- SQLite
- RapidFuzz
- Transformers
- mT5
- Sentence Transformers
- Scikit-learn
- NumPy

## 📁 Project Structure

```text
sms-scam-detector/
├── hackathon_acm2026.py
├── requirements.txt
├── README.md
└── .gitignore
## 🔌 API

### Endpoint

POST `/api/v1/analyze`

### Example Request

```json
{
  "message": "Your bank account will be blocked urgently. Update your KYC now."
}
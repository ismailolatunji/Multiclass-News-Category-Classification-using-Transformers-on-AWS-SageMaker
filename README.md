# Multiclass-News-Category-Classification-using-Transformers-on-AWS-SageMaker
**Project Status**: Ongoing (Last Updated: June 3, 2025)

**Imagine This**: You dump 10,000 news articles into a system, and seconds later they're neatly sorted into folders:
- 📂 Business (Market trends, company news)
- 📂 Health (Medical breakthroughs, wellness)
- 📂 Science (Space exploration, discoveries)
- 📂 Entertainment (Movies, celebrities, culture)

**This isn’t magic, it’s AI working for you.**


![deepseek_mermaid_20250612_f6bf62](https://github.com/user-attachments/assets/9a197b60-3b86-4c95-9aaf-358967fad407)

---

## ❓ Why This Matters

### For Organizations:

| Problem Before	                                              |    Solution Now                                  |
|---------------------------------------------------------------|--------------------------------------------------|
| Employees waste 200+ hours/month manually tagging documents	  |    Instant sorting with 89% accuracy             |
| Inconsistent categories lead to missed insights	              |    Standardized taxonomy across all documents    |
| Scaling requires hiring more staff	                          |    Processes millions of docs with zero added    |


---

## 🚀 Project Summary

This ongoing project showcases how to fine-tune and deploy a **transformer-based NLP model (DistilBERT)** on **Amazon SageMaker** for multiclass text classification. It demonstrates real-world MLOps capabilities such as:

- Scalable **GPU-based training** on AWS
- Deploying real-time **inference endpoints**
- Visualizing data with **Seaborn/Matplotlib**
- Integrating **S3 for data I/O** and **IAM for secure access**
- Tokenization, custom data pipelines, and model serving via `HuggingFaceModel`

---

## 💡 Problem Statement

Given thousands of news headlines, can we automatically **predict their category**?

This helps:

- Improve **content curation** and tagging in media platforms
- Automate **news filtering and personalization**
- Enable smarter **recommendation engines**

---

## 🧰 Tech Stack

| Area            | Tools & Frameworks                                                                 |
|-----------------|-------------------------------------------------------------------------------------|
| Language        | Python                                                                              |
| ML/NLP          | Hugging Face Transformers, PyTorch                                                  |
| Data Engineering| Pandas, NumPy, Seaborn, Matplotlib                                                   |
| Cloud           | Amazon SageMaker, S3, IAM                                                            |
| Model Deployed  | `distilbert-base-uncased`, `cardiffnlp/twitter-roberta-base-sentiment-latest`       |
| Deployment Type | Real-time SageMaker endpoint                                                         |

---

## 🧱 Architecture Overview

```text
                     +----------------+
     News Headlines  |                |   Model Output
     +-------------> |  Tokenization  +----------------> [Entertainment | Business | Science | Health]
                     |  (DistilBERT)  |
                     +----------------+
                             |
                             v
                   +--------------------+
                   |  Fine-tuned Model  | (HuggingFace - PyTorch)
                   +--------------------+
                             |
                             v
                     +----------------+
                     | SageMaker GPU  |
                     |  Training Job  |
                     +----------------+
                             |
                             v
                    S3 Bucket (Output + Logs)

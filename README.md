# Fashion Discovery Engine: MLOps Final Project
Key MLOPS Features - 
* **Live A/B Testing:** Compares raw CLIP model (Variant A) against a domain-specific fine tuned Fasj=hion-CLIP (Variant B).
* **Feedback Logging:** Users can rate search results (👍/👎). Feedback is securely pushed to a Google Cloud Storage (GCS) bucket in real-time.
* **Cloud-Native (Google Cloud) Deployment:** Fully containerized using Docker and orchestrated on Kubernetes (GCP) with load-balanced pods.
* **Full Observability:** Custom backend metrics are scraped by **Prometheus** and visualized in **Grafana** (tracking total searches, error rates, and A/B test win rates).
* **Large File Handling:** Heavy `.pt` model weights and image datasets are cleanly tracked using Git LFS.


## Important Links 
- Google Cloud Deployed Application link - http://34.93.246.183:8501/
- Live Monitoring (using Grafana) - http://35.244.56.146/
- Raw Dataset Link - https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/code
- Demo Video link - [MLOPS demo video.mov](https://drive.google.com/file/d/1ht9pp6N-8RLuMww5yAPStInaWysv5Xej/view?usp=drive_link)

  
## Tech Stack
* **Machine Learning:** PyTorch, HuggingFace Transformers (CLIP / FashionCLIP)
* **Backend:** Flask
* **Frontend:** Streamlit
* **Infrastructure:** Kubernetes (K8s), Docker, Google Cloud Platform (GCP)
* **Storage:** Google Cloud Storage (GCS Buckets)
* **Monitoring:** Prometheus, Grafana
* **Version Control:** Git, Git LFS

 
![Project Pipeline Flowchart](MLOPS_Flowchart.png)

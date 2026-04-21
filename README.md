# Fashion Discovery Engine: MLOps Final Project
Key MLOPS Features - 
* **Live A/B Testing:** Compares raw CLIP model (Variant A) against a domain-specific fine tuned Fasj=hion-CLIP (Variant B).
* **Feedback Logging:** Users can rate search results (using thums up/down icons present on the application). Feedback is securely pushed to a Google Cloud Storage (GCS) bucket in real-time.
* **Cloud-Native (Google Cloud) Deployment:** Fully containerized using Docker and orchestrated on Kubernetes (GCP) with load-balanced pods.
* **Full Observability:** Custom backend metrics are scraped by **Prometheus** and visualized in **Grafana** (tracking total searches, error rates, and A/B test win rates).
* **Large File Handling:** Heavy .pt model weights and image datasets are cleanly tracked using Git LFS.


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


## Flowchart depicting the project pipeline
<img width="700" height="1000" alt="Image" src="https://github.com/user-attachments/assets/53e022bb-5c51-45ae-aa60-33af892ee052" />



- **Data Ingestion & Baseline Model (Stage 1):** A raw Kaggle dataset is processed using a standard CLIP model (setup.py) to generate baseline embeddings (embedding_a.pt), which serves as Variant A.

  
- **Variant B embedding generation (Stage 2):** A specialized Fashion-CLIP model is used (setup_variant_b.py) to generate a second set of fine-tuned embeddings (embedding_b.pt), serving as Variant B. This is how we create A/B testing in our project. 
- **Backend endpooints usage(Stage 3):** A Flask backend manages the A/B testing logic, serving both Variant A and Variant B models through /search (text queries) and /image-search (image queries) endpoints. The endpoint /feedback is used for feedback logging (explained in points below).
- **Interactive Frontend (Stage 4):** A Streamlit application (ui.py) provides the user interface, displaying the top 3 most similar visual matches based on the user's search.
- **Continuous Feedback Loop (Stage 4):** The frontend actively collects user sentiment (+1 for good results, -1 for bad) and logs this data into feedback_log.json to evaluate model performance.
- **Live Real Time Monitoring (Stage 5):** Prometheus scrapes and stores system/application metrics, which are then visualized on live, real-time dashboards using Grafana.
- **Containerization (Stage 6):** The entire application environment is packaged using a Dockerfile and docker-compose.yml, with the final images hosted securely on Docker Hub.
- **Cloud Orchestration (Stage 7):** The final containers are deployed to Google Kubernetes Engine (GKE) on Google Cloud, utilizing K8s deployments and Cloud Storage buckets for scalable, robust hosting.

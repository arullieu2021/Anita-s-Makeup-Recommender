# Anita's Makeup Recommender
### A Computer Vision Approach to Skin Tone Analysis for Makeup Recommendation
 
> **IE University — School of Human Sciences & Technology**
> Ana Maria Rull Orti · Dual Degree in Business Administration & Data and Business Analytics
> Supervised by Prof. Luciano Dyballa · April 2026
 
---
 
## 📌 Overview
 
This project builds a **personalized makeup recommendation system** driven entirely by a photo. Given a single selfie, the pipeline:
 
1. **Segments facial regions** (skin, lips, eyebrows, nose, irises) using Meta's SAM3 model
2. **Classifies makeup style** (no makeup / clean / glam / festival) with a fine-tuned ResNet-18 CNN
3. **Maps skin tone** to the 10-level Monk Skin Tone Scale
4. **Recommends makeup products** from a catalog of 630 items using a hybrid color + content + style scoring engine
5. **Presents results** through an interactive Gradio web interface
 
The key insight is treating **skin tone as an intermediate representation** — a bridge between facial image analysis and cosmetic product data — rather than as a classification endpoint in itself.
 
---
 
## 🗂️ Repository Structure
 
```
├── 1_Data_Cleaning_and_Preparation.ipynb     # Kaggle dataset download & catalog cleaning
├── 2_Face_Segmentation.ipynb                 # SAM3 facial region segmentation & hex extraction
├── 3_Catalog_EDA.ipynb                       # Exploratory analysis of the product catalog
├── 4_Pinterest_Web_Scraping.ipynb            # Playwright-based Pinterest image scraper
├── 4_1_Pinterest_EDA.ipynb                   # Skin tone distribution in the training dataset
├── 5_ResNet18_Classifier.ipynb               # Transfer learning makeup style classifier
├── 6_Recommender_Gradio.ipynb                # End-to-end pipeline + Gradio UI (main app)
├── 7_Survey_Results.ipynb                    # User evaluation analysis (n=30)
├── products_catalog.csv                      # Cleaned cosmetic product catalog (630 products)
├── makeup_classifier.pth                     # Trained ResNet-18 checkpoint
└── README.md
```
 
---
 
## 🚀 Quick Start
 
### Run the Full App (Recommended)
 
Open **`notebooks/6_Recommender_Gradio.ipynb`** in [Google Colab](https://colab.research.google.com/) with a **GPU runtime**.
 
```
Runtime → Change runtime type → T4 GPU
```
 
The notebook will:
- Install all dependencies automatically
- Load SAM3 from HuggingFace Hub (`facebook/sam3`) — requires a HuggingFace token
- Load the ResNet-18 classifier from `makeup_classifier.pth`
- Launch a Gradio app (Step 1: Analyze photo → Step 2: Get recommendations)
 
### HuggingFace Token Setup
 
SAM3 is a gated model. Before running, add your token as a Colab secret:
 
```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")
```
 
Or set it as an environment variable: `HF_TOKEN=your_token`
 
---
 
## 🔬 Pipeline Details
 
### 1. Facial Segmentation — SAM3
 
Five text prompts are issued per image to segment:
- `"skin of the face"` → face_skin
- `"lips"` → lips
- `"eyebrows"` → eyebrows
- `"nose"` → nose
- `"circular iris region of each eye"` → irises
 
Dominant color per region is extracted using **K-means clustering (k=3) in CIELAB space**, with the top/bottom 10% of pixels trimmed to remove specular reflections and shadows.
 
### 2. Monk Skin Tone Mapping
 
Extracted skin RGB is matched to the closest of the 10 official [Monk Skin Tone Scale](https://skintone.google/) reference hex codes using Euclidean distance in RGB space:
 
$$k^* = \arg\min_{k \in \{1,\ldots,10\}} \|\mathbf{s} - \mathbf{m}_k\|_2$$
 
### 3. Makeup Style Classifier — ResNet-18
 
| Hyperparameter | Value |
|---|---|
| Architecture | ResNet-18 (ImageNet pretrained) |
| Trainable layers | `layer4` + fully connected head |
| Input size | 224 × 224 px |
| Training epochs | 20 |
| Optimizer | Adam (lr=0.001, StepLR ×0.5 every 5 epochs) |
| Dropout | 0.3 |
| Validation accuracy | **95.5%** (374 images) |
 
Classes: `no_makeup` · `clean_makeup` · `glam_makeup` · `festival_makeup`
 
Training data: 1,870 images scraped from Pinterest across 4 style categories.
 
### 4. Hybrid Recommendation Engine
 
Products are scored by a weighted combination of three signals:
 
| Signal | Weight | Description |
|---|---|---|
| Color similarity | **0.45** | CIELAB Euclidean distance between product shade and facial region |
| Content relevance | **0.30** | TF-IDF cosine similarity (product text vs. style query) |
| Style boost | **0.25** | Per-style, per-category multiplier |
 
Final score: `hybrid = 0.45 × color_norm + 0.30 × content_norm + 0.25 × style_norm`
 
Products are ranked per category (foundation, lipstick, blush, mascara, bronzer, eyeshadow, eyeliner, eyebrow, lip liner). Users can filter by price range and undertone.
 
---
 
## 📊 Datasets
 
| Dataset | Source | Size |
|---|---|---|
| Cosmetic Brand Products | [Kaggle — shivd24coder](https://www.kaggle.com/datasets/shivd24coder/cosmetic-brand-products-dataset) | 630 products, 9 categories |
| Products Catalog (cleaned) | [Kaggle — anamararullorti](https://www.kaggle.com/datasets/anamararullorti/products-catalog-dataset) | Included in `data/` |
| Pinterest Makeup Style Dataset | [Kaggle — anamararullorti](https://www.kaggle.com/datasets/anamararullorti/makeup-style-pinterest-web-scrapping-dataset) | 1,870 images, 4 classes |
 
---
 
## 📦 Dependencies
 
All dependencies are installed within each notebook. The core stack:
 
```
Python 3.10 · Google Colab (GPU)
 
transformers >= 4.40     # SAM3 loading & inference
torch >= 2.0             # GPU tensor operations
torchvision >= 0.15      # ResNet-18 & image transforms
Pillow >= 9.0            # Image loading & processing
scikit-learn >= 1.3      # TF-IDF, cosine similarity, K-means
colormath >= 3.0         # sRGB → CIELAB conversion
gradio >= 4.0            # Interactive web UI
pandas >= 2.0            # Catalog loading & filtering
matplotlib >= 3.7        # Visualization
```
 
---
 
## 📈 Results Summary
 
| Component | Result |
|---|---|
| Makeup style classifier (validation accuracy) | **95.5%** |
| Segmentation accuracy (user survey, n=30) | **μ = 4.71 / 5** |
| Skin tone recognition | **μ = 3.97 / 5** |
| Recommendation quality | **μ = 4.89 / 5** |
| System utility | **μ = 4.95 / 5** |
| End-to-end inference time | **< 30 seconds** |
 
---
 
## 📄 Citation
 
If you use this work, please cite:
 
```
Rull Orti, A. M. (2026). A Computer Vision Approach to Skin Tone Analysis 
for Makeup Recommendation. IE University, School of Human Sciences & Technology.
```
 
---
 
## 📬 Contact
 
Ana Maria Rull Orti · IE University
GitHub: [@arullieu2021](https://github.com/arullieu2021)

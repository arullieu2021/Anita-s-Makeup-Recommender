# A Computer Vision Approach to Skin Tone Analysis for Makeup Recommendation

**Author:** Ana Maria Rull Orti  
**Institution:** IE University — School of Human Sciences & Technology  
**Degree:** Dual Degree in Business Administration & Data and Business Analytics  
**Supervisor:** Prof. Luciano Dyballa  
**Date:** March 2026  

---

## Overview

This repository contains all code and documentation for the capstone thesis 
*A Computer Vision Approach to Skin Tone Analysis for Makeup Recommendation*. 
The project builds an end-to-end pipeline that segments facial regions from a 
photograph, extracts dominant skin, lip, and eye colors, classifies the makeup 
style, and returns personalized cosmetic product recommendations via a Gradio interface.

## Pipeline Components

| Notebook | Description |
|----------|-------------|
| `1_data_cleaning_and_preparation.ipynb` | Cosmetic product catalog cleaning and preprocessing |
| `2_face_segmentation.ipynb` | SAM3-based facial region segmentation and color extraction |
| `3_catalog_eda.ipynb` | Exploratory data analysis of the product catalog |
| `4_pinterest_scraping.ipynb` | Pinterest web scraping to build the makeup style dataset |
| `5_resnet18_classifier.ipynb` | ResNet-18 fine-tuning for makeup style classification |
| `6_recommender_gradio.ipynb` | Hybrid recommendation engine and Gradio interface |

## Key Results

- **Makeup style classifier:** 95.5% validation accuracy (ResNet-18, 20 epochs, n=1,870 images)
- **Facial segmentation:** SAM3 prompt-based segmentation across 5 facial regions
- **Recommendation engine:** Hybrid scoring across 630 products using CIELAB color distance, TF-IDF, and style boost weights
- **User evaluation:** Mean satisfaction score 4.75/5 across 30 participants

## How to Run

All notebooks are designed to run on **Google Colab** with a GPU runtime (T4 or higher).

1. Open any notebook in Google Colab
2. Upload the required dataset files when prompted
3. Run all cells in order
4. For the full pipeline, run notebook 6 last

## Datasets

- [Cosmetic Brand Products Dataset (Kaggle)](https://www.kaggle.com/datasets/shivd24coder/cosmetic-brand-products-dataset)
- [Products Catalog Dataset (Kaggle)](https://www.kaggle.com/datasets/anamararullorti/products-catalog-dataset)
- [Pinterest Makeup Style Dataset (Kaggle)](https://www.kaggle.com/datasets/anamararullorti/makeup-style-pinterest-web-scrapping-dataset)

## Thesis

The full thesis document is available in this repository: 
`thesis_skin_tone_makeup_recommendation.pdf`

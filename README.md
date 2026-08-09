---
title: Sleep Disorder Prediction
emoji: 😴
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Sleep Disorder Prediction

An interactive Gradio application that predicts sleep-disorder categories from
lifestyle and biometric inputs using a soft-voting ensemble of Random Forest
and XGBoost classifiers.

The application trains from `Sleep_health_and_lifestyle_dataset (1).csv` when
the Space starts. It is intended for educational use and is not a clinical
diagnostic tool.

## Streamlit Community Cloud

Deploy `streamlit_app.py` as the main file on Streamlit Community Cloud. The
same dataset file and `requirements.txt` must remain in the repository.

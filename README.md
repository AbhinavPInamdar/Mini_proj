#  Fake Review Detection Chrome Extension

A lightweight Chrome extension that identifies fake reviews on e-commerce platforms like Amazon and Flipkart using a Bi-LSTM model served via a scalable cloud API.

##  Overview

This extension enhances online shopping by detecting potentially fake reviews using machine learning and NLP. It scrapes visible reviews from product pages, sends them to a Flask-based backend hosted on AWS ECS, and classifies them using a trained Bi-LSTM model.

###  Features

-  Real-time review classification (Fake / Real)
-  Bi-LSTM-based NLP model trained on review text + metadata
-  Cloud-hosted backend (Flask + Docker + AWS ECS)
-  DOM scraping for Amazon and Flipkart pages
-  Fast, asynchronous API calls for real-time inference
-  Minimal, responsive UI popup (React + TypeScript)

---

##  Tech Stack

| Layer          | Tech                                                  |
|----------------|-------------------------------------------------------|
| Frontend       | TypeScript, Chrome Extension APIs              |
| ML Backend     | Python, Flask, TensorFlow, Bi-LSTM, Scikit-learn      |
| Infrastructure | Docker, AWS ECS (Fargate)                             |
| Communication  | REST API (JSON over HTTP)                             |

---

##  Model Details

- Architecture: **Bi-directional LSTM**
- Input: Tokenized review text + reviewer metadata
- Accuracy: **97%** on test data
- Deployment: **TensorFlow Serving** behind a Flask API

---

##  How It Works

1. **User browses a product page**
2. **Extension scrapes visible reviews** from the DOM
3. **API sends text → ML backend → prediction**
4. **Results are displayed inline or in the popup**

---

```mermaid
graph TD
    subgraph User Browser
        A1[User visits product page ,Amazon / Flipkart]
        A2[Chrome Extension]
        A3[Content Script scrapes reviews]
        A1 --> A2 --> A3
    end

    A3 --> B1[REST API Request, Review Data]

    subgraph Backend [Cloud Backend AWS ECS]
        B1 --> B2[Flask API Server]
        B2 --> B3[Bi-LSTM ML Model]
        B3 --> B4[Prediction: Real or Fake]
    end

    B4 --> C1[API Response with Classification]

    subgraph Extension UI
        C1 --> D1[Popup UI Displays Classification]
    end

```



##  Future Improvements

- [ ] **Add model confidence visualization**  
  Display prediction confidence alongside classification, e.g., "78% Fake" or "91% Real".

- [ ] **Improve domain support**  
  Extend support beyond Amazon and Flipkart to other marketplaces like Snapdeal, Meesho, or international sites like eBay.

- [ ] **Track user feedback for active retraining**  
  Allow users to confirm or correct predictions, and feed that data into a retraining pipeline to continuously improve the model.

##  License

This project is licensed under the [MIT License](./LICENSE).




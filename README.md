# Intrusion Detection System (IDS)

This project is a basic implementation of an **Intrusion Detection System (IDS)** using the KDD Cup 1999 dataset. The system performs data preprocessing, handles class imbalance with SMOTE, trains a RandomForest model, and evaluates its performance.

---

## 📂 Project Structure

```
Intrusion Detection System (IDS)/
├── data/                    # Place your dataset here
│   ├── KDDTrain+.csv        # Training data
│   └── KDDTest+.csv         # Test data
├── src/                    # Source code
│   ├── preprocess.py       # Preprocessing functions
│   ├── model.py            # Model code (training, cross-validation)
│   └── utils.py            # Helper functions (evaluation, etc.)
├── main.py                 # Main script to run the entire pipeline
├── requirements.txt        # Dependencies list
└── README.md               # Project overview
```

---

## 🎓 How to Use

### 1. Clone the repository

```bash
git clone https://github.com/your-username/intrusion-detection-system.git
cd intrusion-detection-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Download the `KDDTrain+.csv` and `KDDTest+.csv` files and put them inside the `data/` directory.

### 4. Run the main script

```bash
python main.py
```

---

## 👩‍💼 Features

* **Data Preprocessing** with categorical encoding and normalization
* **SMOTE** to handle class imbalance
* **RandomForestClassifier** with `class_weight='balanced'`
* **Cross-Validation** to check robustness
* **Evaluation** using precision, recall, F1-score, and confusion matrix

---

## 🌐 Requirements

See `requirements.txt`. Includes:

```
scikit-learn
pandas
numpy
imblearn
```

---

## 🚀 Future Improvements

* Support for multiple models (e.g., XGBoost, SVM)
* Stream-based real-time intrusion detection
* UI dashboard for monitoring results

---

## 🎉 Credits

Created by \[Your Name]. Based on the KDD Cup 1999 dataset.

---

## ⚖️ License

MIT License

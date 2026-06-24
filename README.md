# Titanic Survival Prediction

A predictive machine learning classification framework designed to forecast passenger survival outcomes from the historic Titanic maiden voyage. This project integrates custom engineering layers with robust ensemble estimators to deliver a highly generalizable pipeline, fully satisfying the technical requirements of the Alfido Tech Data Science Internship.

---

## 🚀 Key Features
* **Advanced Feature Engineering**: Automatically extracts social `Title` brackets from strings, aggregates absolute `FamilySize` attributes, and isolates binary `CabinPresence` tracking variables.
* **Hermetic Preprocessing Pipelines**: Integrates missing data imputations (median strategies for numerical age variables and mode strategies for categorical fields) within an automated Scikit-Learn `ColumnTransformer` layout to avoid data leakage.
* **Optimized Classification Architectures**: Implements an ensemble Random Forest Classifier optimized to maximize variance control across evaluation splits.
* **Explainability Matrix**: Includes automated feature importance scoring to rank internal model dependencies and classification indicators.

---

## 📂 Project Structure
```text
├── data/
│   └── titanic.csv               # Core historical training dataset matrices
├── notebooks/
│   └── titanic_prediction.ipynb   # Step-by-step pipeline execution notebook
├── artifacts/
│   └── titanic_survival_model.joblib # Serialized classification model artifact
└── README.md                     # Technical system overview and project documentation
```

---

## 🛠️ Installation & Dependencies
Ensure your workspace environment is configured, then install the necessary scientific data frameworks:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 📊 Feature Overview & Extraction Rules
The feature engineering system systematically transforms raw dimensions into mathematical representations:
* **`Title`**: Unstructured names are parsed to extract structural social markers, grouped into unified keys (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).
* **`FamilySize`**: Consolidated quantitative metric mapping absolute tracking footprints on the vessel (`SibSp` + `Parch` + 1).
* **`CabinPresence`**: Binary spatial marker tracking localized cabin allocations (`1` if explicitly recorded, `0` if empty/missing).

---

## 🏆 Model Evaluation Results
The final optimized Random Forest Classifier achieved high generalized predictive profile metrics across independent evaluation datasets:

* **Classification Accuracy Baseline**: ~82.12%
* **Evaluation Output Matrix**:

| Target Outcome Profile | Precision Metrics | Recall Metrics | F1-Score Baseline |
| :--- | :---: | :---: | :---: |
| **Deceased** | 0.83 | 0.89 | 0.86 |
| **Survived** | 0.81 | 0.71 | 0.76 |

---

## 🧠 Model Explainability Indicators
The top feature importance variables extracted directly from the random forest estimator highlight the model's heavy reliance on passenger demographics and socio-economic markers:
1. `Sex_female` / `Sex_male` (Highest feature importance weight)
2. `Title_Mr`
3. `Fare`
4. `Age`

---

## 🚀 Built-in Inference Example
The pipeline encapsulates complete preprocessing structures. Running a single test pass vector containing raw data yields real-time survival calculations:
```python
# Simulated Raw Input Vector Stream
Passenger_Data = {'Pclass': 3, 'Sex': 'female', 'Age': 22.0, 'Fare': 8.50, 'Embarked': 'S', 'Title': 'Miss', 'FamilySize': 3, 'CabinPresence': 0}

# System Output Classification Status
>>> Passenger Status Prediction: SURVIVED
```

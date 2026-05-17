# 👔 Employee Attrition Intelligence App

A machine learning web application that predicts whether an employee is likely to leave the company based on HR data. Built with Python, Scikit-learn (Logistic Regression), and deployed using Streamlit.

---

## 🚀 Live Demo

👉 [Click here to try the live app!](https://employee-attrition-intelligence-6.streamlit.app/)
> Deployed on Streamlit Cloud

---

## 📌 Problem Statement

Employee attrition is one of the biggest challenges for organizations. Losing key employees costs time, money, and productivity. This app helps HR teams identify at-risk employees early so they can take action to retain them.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing |
| Scikit-learn | Logistic Regression model |
| StandardScaler | Feature scaling |
| OneHotEncoder | Categorical encoding |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web app deployment |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~86% |
| Precision | Evaluated on test set |
| Recall | Evaluated on test set |
| F1 Score | Evaluated on test set |

---

## 📁 Project Structure
'''
employee-attrition-intelligence/
│
├── app.py                        # Streamlit web app
├── classifi.ipynb                # Jupyter notebook with EDA and model training
├── artifacts.pkl                 # Saved model, scaler and encoder
├── HR-Employee-Attrition.csv     # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation '''
---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/nithinreddyp2004/employee-attrition-intelligence.git

# 2. Go into the folder
cd employee-attrition-intelligence

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🔍 Features Used

- Employee demographics (Age, Gender, MaritalStatus)
- Job details (Department, JobRole, JobLevel)
- Work metrics (OverTime, YearsAtCompany, MonthlyIncome)
- Satisfaction scores (JobSatisfaction, WorkLifeBalance)

---

## 🙋 Author

**Nithin Reddy P** — [GitHub](https://github.com/nithinreddyp2004)

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #f8fafc;
        }

        .main {
            background: transparent;
            color: #f8fafc;
        }

        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #1e293b 0%, #1d4ed8 55%, #2563eb 100%);
            color: white;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            color: white;
        }

        .hero p {
            margin: 0.35rem 0 0 0;
            opacity: 0.95;
            font-size: 1rem;
            color: #e2e8f0;
        }

        .card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            color: #f8fafc;
        }

        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.6rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #93c5fd;
        }

        .small-note {
            color: #cbd5e1;
            font-size: 0.92rem;
        }

        .result-ok {
            padding: 1rem;
            border-radius: 16px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid #10b981;
            color: #d1fae5;
        }

        .result-warn {
            padding: 1rem;
            border-radius: 16px;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid #ef4444;
            color: #fee2e2;
        }

        .footer-note {
            margin-top: 1.5rem;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: #1e293b;
            border: 1px solid #334155;
            color: #cbd5e1;
            font-size: 0.92rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0b1120;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        /* Labels */
        label, .stSelectbox label, .stNumberInput label, .stTextInput label {
            color: #f8fafc !important;
            font-weight: 500;
        }

        /* Inputs */
        .stTextInput input,
        .stNumberInput input,
        div[data-baseweb="select"] {
            background-color: #0f172a !important;
            color: #f8fafc !important;
            border-radius: 10px !important;
            border: 1px solid #334155 !important;
        }

        /* Dropdown text */
        div[data-baseweb="select"] * {
            color: #f8fafc !important;
        }

        /* Buttons */
        .stButton button,
        .stDownloadButton button {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white !important;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            padding: 0.55rem 1rem;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            color: white !important;
        }

        /* Tabs/radio containers */
        .stRadio label {
            color: #f8fafc !important;
        }

        /* Metrics */
        [data-testid="stMetric"] {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
        }

        [data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================
@st.cache_resource
def load_artifacts(path: str = "artifacts.pkl"):
    return joblib.load(path)


def get_category_value(col_name: str):
    options_map = {
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        "Gender": ["Male", "Female"],
        "JobRole": [
            "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
            "Manufacturing Director", "Research Director", "Research Scientist",
            "Sales Executive", "Sales Representative",
        ],
        "MaritalStatus": ["Single", "Married", "Divorced"],
        "Over18": ["Y"],
        "OverTime": ["Yes", "No"],
    }
    if col_name in options_map:
        return st.selectbox(col_name, options_map[col_name])
    return st.text_input(col_name, value="")


def get_numeric_value(col_name: str):
    defaults = {
        "Age": 30,
        "DailyRate": 0,
        "DistanceFromHome": 5,
        "Education": 1,
        "EnvironmentSatisfaction": 3,
        "HourlyRate": 0,
        "JobInvolvement": 3,
        "JobLevel": 1,
        "JobSatisfaction": 3,
        "MonthlyIncome": 0,
        "MonthlyRate": 0,
        "NumCompaniesWorked": 0,
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 1,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 0,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 0,
    }
    return st.number_input(col_name, value=float(defaults.get(col_name, 0)), step=1.0, format="%.0f")


def preprocess_input(user_df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    num_cols = list(artifacts["num_cols"])
    cat_cols = list(artifacts["cat_cols"])
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]

    user_df = user_df.copy()

    hidden_numeric_defaults = {
        "EmployeeCount": 1,
        "EmployeeNumber": 0,
        "StandardHours": 80,
    }
    hidden_categorical_defaults = {
        "Over18": "Y",
    }

    for c in num_cols:
        if c not in user_df.columns:
            user_df[c] = hidden_numeric_defaults.get(c, 0)
    for c in cat_cols:
        if c not in user_df.columns:
            user_df[c] = hidden_categorical_defaults.get(c, "")

    user_df = user_df[num_cols + cat_cols]

    user_num = pd.DataFrame(scaler.transform(user_df[num_cols]), columns=num_cols)
    user_cat_arr = encoder.transform(user_df[cat_cols])

    try:
        cat_feature_names = encoder.get_feature_names_out(cat_cols)
    except Exception:
        cat_feature_names = [f"cat_{i}" for i in range(user_cat_arr.shape[1])]

    user_cat = pd.DataFrame(user_cat_arr, columns=cat_feature_names)
    return pd.concat([user_num.reset_index(drop=True), user_cat.reset_index(drop=True)], axis=1)


def prediction_card(pred_value, probability=None):
    pred_text = str(pred_value).strip().lower()
    if pred_text in {"yes", "1", "true", "left"}:
        title = "High Attrition Risk"
        message = "The model predicts that the employee may leave."
        box_class = "result-warn"
    else:
        title = "Low Attrition Risk"
        message = "The model predicts that the employee is likely to stay."
        box_class = "result-ok"

    st.markdown(f'<div class="{box_class}"><h3>{title}</h3><p>{message}</p></div>', unsafe_allow_html=True)

    if probability is not None:
        st.metric("Attrition Probability", f"{probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))


def normalize_label_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def find_actual_column(df: pd.DataFrame):
    candidates = ["Attrition", "Actual", "Target", "Label", "y_true", "Outcome"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def display_metric_cards(metrics_dict):
    cols = st.columns(len(metrics_dict))
    for col, (name, value) in zip(cols, metrics_dict.items()):
        col.metric(name, f"{value:.2%}")


def build_sample_csv(visible_num_cols, visible_cat_cols):
    sample = {}
    for c in visible_num_cols:
        sample[c] = [0]
    for c in visible_cat_cols:
        if c == "BusinessTravel":
            sample[c] = ["Travel_Rarely"]
        elif c == "Department":
            sample[c] = ["Sales"]
        elif c == "EducationField":
            sample[c] = ["Life Sciences"]
        elif c == "Gender":
            sample[c] = ["Male"]
        elif c == "JobRole":
            sample[c] = ["Sales Executive"]
        elif c == "MaritalStatus":
            sample[c] = ["Single"]
        elif c == "OverTime":
            sample[c] = ["Yes"]
        elif c == "Over18":
            sample[c] = ["Y"]
        else:
            sample[c] = [""]
    return pd.DataFrame(sample)


# =========================================================
# Load model artifacts
# =========================================================
try:
    artifacts = load_artifacts()
except Exception as e:
    st.error("Could not load artifacts.pkl. Keep it in the same folder as app.py.")
    st.exception(e)
    st.stop()

model = artifacts["model"]
num_cols = list(artifacts["num_cols"])
cat_cols = list(artifacts["cat_cols"])

HIDE_NUMERIC = {"EmployeeCount", "EmployeeNumber", "StandardHours"}
HIDE_CATEGORICAL = {"Over18"}
visible_num_cols = [c for c in num_cols if c not in HIDE_NUMERIC]
visible_cat_cols = [c for c in cat_cols if c not in HIDE_CATEGORICAL]

# =========================================================
# Header
# =========================================================
st.markdown(
    """
    <div class="hero">
        <h1>Employee Attrition Intelligence Platform</h1>
        <p>Machine learning-powered workforce attrition prediction platform with interactive analytics, batch evaluation, and decision support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚙️ Model Panel")
st.sidebar.write("Professional dashboard layout with hidden technical fields.")
st.sidebar.caption(f"Visible numeric features: {len(visible_num_cols)}")
st.sidebar.caption(f"Visible categorical features: {len(visible_cat_cols)}")
st.sidebar.caption(f"Total training features: {len(num_cols) + len(cat_cols)}")

page = st.sidebar.radio("Navigate", ["Single Prediction", "Batch Analytics", "Model Insights"])

# =========================================================
# Single Prediction tab
# =========================================================
if page == "Single Prediction":
    st.subheader("Single Employee Prediction")
    st.write("Enter employee details to estimate attrition risk.")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("predict_form"):
            st.markdown('<div class="section-title">Personal & Work Information</div>', unsafe_allow_html=True)
            num_values = {}

            c1, c2 = st.columns(2)
            with c1:
                for c in ["Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel"]:
                    if c in visible_num_cols:
                        num_values[c] = get_numeric_value(c)
            with c2:
                for c in ["JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "TrainingTimesLastYear", "PerformanceRating"]:
                    if c in visible_num_cols:
                        num_values[c] = get_numeric_value(c)

            st.markdown('<div class="section-title">Salary & Experience</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                for c in ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole"]:
                    if c in visible_num_cols:
                        num_values[c] = get_numeric_value(c)
            with c4:
                for c in ["YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked", "PercentSalaryHike"]:
                    if c in visible_num_cols:
                        num_values[c] = get_numeric_value(c)

            st.markdown('<div class="section-title">Job Details</div>', unsafe_allow_html=True)
            cat_values = {}
            c5, c6 = st.columns(2)
            with c5:
                for c in ["BusinessTravel", "Department", "Gender", "JobRole"]:
                    if c in visible_cat_cols:
                        cat_values[c] = get_category_value(c)
            with c6:
                for c in ["EducationField", "MaritalStatus", "OverTime"]:
                    if c in visible_cat_cols:
                        cat_values[c] = get_category_value(c)

            submitted = st.form_submit_button("Predict Attrition", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        st.markdown('<p class="small-note">Hidden fields like EmployeeNumber and StandardHours are filled automatically.</p>', unsafe_allow_html=True)

        if submitted:
            try:
                with st.spinner("Generating prediction..."):
                    row = {**num_values, **cat_values}
                    user_df = pd.DataFrame([row])
                    X = preprocess_input(user_df, artifacts)
                    pred = model.predict(X)[0]
                    prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

                prediction_card(pred, prob)

                with st.expander("Show submitted input"):
                    st.dataframe(user_df, use_container_width=True)
            except Exception as e:
                st.error("Prediction failed. Check the saved encoder/model files.")
                st.exception(e)
        else:
            st.info("Fill the form and click Predict Attrition to see the result.")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Batch Analytics tab
# =========================================================
elif page == "Batch Analytics":
    st.subheader("Batch Prediction and Analytics")
    st.write("Upload a CSV file to generate predictions. If your file includes the actual target column, evaluation charts will appear automatically.")

    uploaded = st.file_uploader("Choose CSV", type=["csv"])

    sample_df = build_sample_csv(visible_num_cols, visible_cat_cols)
    st.download_button(
        "Download sample CSV format",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_attrition_input.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Preview of uploaded data")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run batch prediction", use_container_width=True):
            try:
                with st.spinner("Generating batch predictions..."):
                    X = preprocess_input(df, artifacts)
                    preds = model.predict(X)
                    out = df.copy()
                    out["Prediction"] = preds

                    if hasattr(model, "predict_proba"):
                        out["Attrition_Probability"] = model.predict_proba(X)[:, 1]

                st.success("Batch predictions ready.")
                st.dataframe(out, use_container_width=True)

                actual_col = find_actual_column(out)
                if actual_col is not None:
                    actual = normalize_label_series(out[actual_col])
                    predicted = normalize_label_series(out["Prediction"])

                    label_map = {
                        "yes": "yes",
                        "no": "no",
                        "left": "yes",
                        "stayed": "no",
                        "1": "yes",
                        "0": "no",
                        "true": "yes",
                        "false": "no",
                    }

                    actual_mapped = actual.map(lambda x: label_map.get(x, x))
                    predicted_mapped = predicted.map(lambda x: label_map.get(x, x))

                    valid_mask = actual_mapped.isin(["yes", "no"]) & predicted_mapped.isin(["yes", "no"])
                    actual_eval = actual_mapped[valid_mask]
                    pred_eval = predicted_mapped[valid_mask]

                    if len(actual_eval) > 0:
                        acc = accuracy_score(actual_eval, pred_eval)
                        prec = precision_score(actual_eval, pred_eval, pos_label="yes", zero_division=0)
                        rec = recall_score(actual_eval, pred_eval, pos_label="yes", zero_division=0)
                        f1 = f1_score(actual_eval, pred_eval, pos_label="yes", zero_division=0)

                        st.markdown("### Batch Evaluation Summary")
                        display_metric_cards({
                            "Accuracy": acc,
                            "Precision": prec,
                            "Recall": rec,
                            "F1 Score": f1,
                        })

                        chart_left, chart_right = st.columns(2)

                        with chart_left:
                            cm = confusion_matrix(actual_eval, pred_eval, labels=["yes", "no"])
                            fig, ax = plt.subplots()
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt="d",
                                cmap="Blues",
                                xticklabels=["Yes", "No"],
                                yticklabels=["Yes", "No"],
                                ax=ax,
                            )
                            ax.set_title("Confusion Matrix")
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            st.pyplot(fig)
                            plt.close(fig)

                        with chart_right:
                            count_df = pd.DataFrame({
                                "Class": ["Yes", "No"],
                                "Actual": [int((actual_eval == "yes").sum()), int((actual_eval == "no").sum())],
                                "Predicted": [int((pred_eval == "yes").sum()), int((pred_eval == "no").sum())],
                            })
                            plot_df = count_df.melt(id_vars="Class", var_name="Type", value_name="Count")

                            fig2, ax2 = plt.subplots()
                            sns.barplot(data=plot_df, x="Class", y="Count", hue="Type", ax=ax2)
                            ax2.set_title("Actual vs Predicted Counts")
                            ax2.set_xlabel("Class")
                            ax2.set_ylabel("Count")
                            st.pyplot(fig2)
                            plt.close(fig2)

                        st.caption(f"Evaluation used column: {actual_col}")
                    else:
                        st.info("An actual label column was found, but its values could not be mapped to Yes/No for evaluation.")
                else:
                    st.info("No actual label column found. Add a column like Attrition to show evaluation charts.")

                st.download_button(
                    "Download results CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="attrition_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as e:
                st.error("Could not process the CSV. Ensure the required columns exist.")
                st.exception(e)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Model Insights tab
# =========================================================
elif page == "Model Insights":
    st.subheader("Model Insights")
    st.write("A clear summary of the trained model, input features, and deployment notes.")

    model_name = type(model).__name__
    has_proba = hasattr(model, "predict_proba")
    has_coef = hasattr(model, "coef_")
    total_encoded_features = len(num_cols) + len(cat_cols)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model type", model_name)
    c2.metric("Numeric columns", len(num_cols))
    c3.metric("Categorical columns", len(cat_cols))
    c4.metric("Total input fields", total_encoded_features)

    st.markdown("### Model summary")
    st.write(f"**Algorithm:** {model_name}")
    st.write(f"**Supports probability output:** {'Yes' if has_proba else 'No'}")
    st.write(f"**Supports coefficient interpretation:** {'Yes' if has_coef else 'No'}")
    st.write("**Task:** Binary classification for employee attrition prediction")

    st.markdown("### Hidden fields filled automatically")
    st.write("- EmployeeCount")
    st.write("- EmployeeNumber")
    st.write("- StandardHours")
    st.write("- Over18")

    st.markdown("### Visible input groups")
    left_info, right_info = st.columns(2)
    with left_info:
        st.write("**Numeric features shown in the form:**")
        st.write(", ".join(visible_num_cols) if visible_num_cols else "None")
    with right_info:
        st.write("**Categorical features shown in the form:**")
        st.write(", ".join(visible_cat_cols) if visible_cat_cols else "None")

    st.markdown("### Raw columns used by the model")
    st.write("**Numeric:**", ", ".join(num_cols))
    st.write("**Categorical:**", ", ".join(cat_cols))

    if has_coef:
        st.markdown("### Coefficient insight")
        try:
            try:
                feature_names = list(num_cols) + list(artifacts["encoder"].get_feature_names_out(cat_cols))
            except Exception:
                feature_names = list(num_cols)

            coef_values = model.coef_[0]
            coef_df = pd.DataFrame({"Feature": feature_names[: len(coef_values)], "Coefficient": coef_values})
            pos_df = coef_df.sort_values("Coefficient", ascending=False).head(10)
            neg_df = coef_df.sort_values("Coefficient", ascending=True).head(10)

            coef_left, coef_right = st.columns(2)
            with coef_left:
                st.write("Top positive coefficients")
                st.dataframe(pos_df, use_container_width=True)
            with coef_right:
                st.write("Top negative coefficients")
                st.dataframe(neg_df, use_container_width=True)
        except Exception:
            st.info("Coefficient view could not be generated for this model.")

    st.markdown("### Deployment notes")
    st.write("- Keep `app.py` and `artifacts.pkl` in the same folder.")
    st.write("- Upload a CSV with an actual target column like `Attrition` to see batch evaluation charts.")
    st.write("- Use the same scikit-learn version that created the encoder and scaler whenever possible.")

# =========================================================
# Footer
# =========================================================
st.markdown(
    """
    <div class="footer-note">
        Developed using Streamlit, Scikit-learn, Pandas, Matplotlib, and Seaborn for employee attrition analytics.
    </div>
    """,
    unsafe_allow_html=True,
)


import os
import io
import base64
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "users.db"
MODEL_PATH = BASE_DIR / "models" / "dengue_model.pkl"
DATASET_PATH = BASE_DIR / "dengue_dataset.pkl"
UPLOAD_FOLDER = BASE_DIR / "uploads"

REQUIRED_FEATURES = [
    "WBC",
    "Platelets",
    "Hematocrit",
    "Hemoglobin"
]

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 10000

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "local-development-key-change-before-deployment"
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fullname TEXT NOT NULL,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


initialize_database()


# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

model = None
master_df = pd.DataFrame(columns=REQUIRED_FEATURES + ["Result"])

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as error:
    print(f"Model could not be loaded: {error}")
    print("Check model compatibility or run generate_data.py.")

try:
    if DATASET_PATH.exists():
        master_df = pd.read_pickle(DATASET_PATH)

        if not isinstance(master_df, pd.DataFrame):
            raise ValueError("The saved dataset is not a DataFrame.")

        print(f"Dataset loaded: {len(master_df)} records.")
    else:
        print("Dataset file not found. Using an empty dataset.")
except Exception as error:
    print(f"Dataset could not be loaded: {error}")


# --------------------------------------------------
# VALIDATION HELPERS
# --------------------------------------------------

def validate_measurements(form):
    values = {}

    limits = {
        "wbc": (0, 100000),
        "platelets": (0, 2000000),
        "hct": (0, 100),
        "hb": (0, 100)
    }

    for field, (minimum, maximum) in limits.items():
        raw_value = form.get(field, "").strip()

        if not raw_value:
            raise ValueError(f"{field.upper()} is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{field.upper()} must be a number.")

        if not np.isfinite(value) or not minimum < value <= maximum:
            raise ValueError(
                f"{field.upper()} must be greater than {minimum} "
                f"and at most {maximum}."
            )

        values[field] = value

    return values


def prepare_csv(file):
    filename = secure_filename(file.filename or "")

    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    try:
        df = pd.read_csv(file, nrows=MAX_CSV_ROWS + 1)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        raise ValueError("The uploaded file is not a valid CSV.")

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    if len(df) > MAX_CSV_ROWS:
        raise ValueError(
            f"The CSV cannot contain more than {MAX_CSV_ROWS} rows."
        )

    missing = [column for column in REQUIRED_FEATURES if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(missing)
        )

    for column in REQUIRED_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[column].isna().any():
            raise ValueError(
                f"Column {column} contains missing or non-numeric values."
            )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Column {column} contains invalid numeric values."
            )

    if "Result" in df.columns:
        df["Result"] = pd.to_numeric(df["Result"], errors="coerce")

        if df["Result"].isna().any() or not df["Result"].isin([0, 1]).all():
            raise ValueError(
                "The Result column must contain only 0 or 1."
            )

    return df


# --------------------------------------------------
# VISUALIZATION HELPERS
# --------------------------------------------------

def get_img_url():
    image = io.BytesIO()
    plt.savefig(image, format="png", bbox_inches="tight")
    image.seek(0)
    encoded_image = base64.b64encode(image.getvalue()).decode("utf-8")
    plt.close()
    return encoded_image


def generate_visualizations(df):
    visual_urls = []

    if df is None or df.empty:
        return visual_urls

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    try:
        if (
            "Platelets" in numeric_df.columns
            and numeric_df["Platelets"].notna().any()
        ):
            plt.figure(figsize=(6, 4))
            sns.histplot(
                numeric_df["Platelets"].dropna(),
                kde=False,
                color="red"
            )
            plt.title("Patient Platelet Distribution")
            plt.xlabel("Platelet Count")
            plt.tight_layout()
            visual_urls.append(get_img_url())

        if (
            "WBC" in numeric_df.columns
            and "Platelets" in numeric_df.columns
        ):
            scatter_df = numeric_df.dropna(
                subset=["WBC", "Platelets"]
            )

            plt.figure(figsize=(6, 4))

            if "Result" in scatter_df.columns:
                sns.scatterplot(
                    data=scatter_df,
                    x="WBC",
                    y="Platelets",
                    hue="Result",
                    palette="viridis"
                )
            else:
                sns.scatterplot(
                    data=scatter_df,
                    x="WBC",
                    y="Platelets"
                )

            plt.title("WBC vs Platelet Count")
            plt.tight_layout()
            visual_urls.append(get_img_url())

        if numeric_df.shape[1] >= 2:
            correlation = numeric_df.corr()

            if not correlation.empty:
                plt.figure(figsize=(7, 5))
                sns.heatmap(
                    correlation,
                    annot=True,
                    cmap="RdYlGn",
                    fmt=".2f"
                )
                plt.title("Numeric Feature Correlation")
                plt.tight_layout()
                visual_urls.append(get_img_url())

    except (ValueError, TypeError, KeyError) as error:
        print(f"Visualization warning: {error}")
        plt.close("all")

    return visual_urls


def calculate_stats(df):
    total = len(df)

    positive = 0
    average_platelets = 0

    if "Result" in df.columns:
        results = pd.to_numeric(df["Result"], errors="coerce")
        positive = int((results == 1).sum())

    if "Platelets" in df.columns:
        platelets = pd.to_numeric(df["Platelets"], errors="coerce")
        average = platelets.mean()

        if pd.notna(average):
            average_platelets = round(float(average), 0)

    return {
        "total": total,
        "pos": positive,
        "avg_plt": average_platelets
    }


def predict_patient(values):
    if model is None:
        raise RuntimeError(
            "The prediction model is unavailable. Check the model file "
            "and its scikit-learn compatibility."
        )

    features = pd.DataFrame(
        [[
            values["wbc"],
            values["platelets"],
            values["hct"],
            values["hb"]
        ]],
        columns=REQUIRED_FEATURES
    )

    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(probabilities[classes.index(1)]) * 100
        else:
            probability = 0.0
    else:
        probability = None

    return prediction, probability


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not fullname or not username or not password:
            flash("Please fill in all registration fields.")
            return render_template("register.html")

        if len(fullname) > 100 or len(username) > 50:
            flash("Name or username is too long.")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.")
            return render_template("register.html")

        if len(password) > 128:
            flash("Password must not exceed 128 characters.")
            return render_template("register.html")

        password_hash = generate_password_hash(password)

        try:
            with get_db_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO users (fullname, username, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (fullname, username, password_hash)
                )

            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("That username already exists. Please choose another.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,)
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user"] = user["username"]
            session["fullname"] = user["fullname"]

            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")

    return render_template("login.html")


@app.route("/predict_single", methods=["POST"])
def predict_single():
    if "user_id" not in session:
        flash("Please log in to use predictions.")
        return redirect(url_for("login"))

    try:
        values = validate_measurements(request.form)
        prediction, probability = predict_patient(values)

        probability_text = (
            f"{probability:.1f}%"
            if probability is not None
            else "Unavailable"
        )

        result_text = (
            "Dengue POSITIVE"
            if prediction == 1
            else "Dengue NEGATIVE"
        )

        return render_template(
            "dashboard.html",
            pred_res=result_text,
            prob=probability_text,
            stats=calculate_stats(master_df),
            vis_urls=generate_visualizations(master_df)
        )

    except (ValueError, RuntimeError) as error:
        flash(str(error))
        return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        flash("Please log in to access the dashboard.")
        return redirect(url_for("login"))

    df_master = master_df.copy()
    single_res = None
    vis_urls = []

    if request.method == "POST":
        if "wbc" in request.form:
            try:
                values = validate_measurements(request.form)
                prediction, probability = predict_patient(values)

                single_res = {
                    "status": (
                        "Positive" if prediction == 1 else "Negative"
                    ),
                    "prob": (
                        round(probability, 2)
                        if probability is not None
                        else None
                    ),
                    "color": (
                        "danger" if prediction == 1 else "success"
                    )
                }

            except (ValueError, RuntimeError) as error:
                flash(str(error))

        elif "dataset" in request.files:
            uploaded_file = request.files["dataset"]

            if uploaded_file and uploaded_file.filename:
                try:
                    df_master = prepare_csv(uploaded_file)
                    flash("CSV uploaded and validated successfully.")
                except ValueError as error:
                    flash(str(error))

    stats = calculate_stats(df_master)
    vis_urls = generate_visualizations(df_master)

    return render_template(
        "dashboard.html",
        stats=stats,
        vis_urls=vis_urls,
        single_res=single_res
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(error):
    flash("File is too large. Maximum upload size is 5 MB.")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)

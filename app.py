import os
import pandas as pd
import numpy as np
import joblib  # Use joblib instead of pickle for model
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "dengue_ai_secure_key"
app.config['UPLOAD_FOLDER'] = 'uploads/'

# --- Load Model & Initial Dataset ---
# Ensure you run the generate_data.py script first!
try:
    model = joblib.load('models/dengue_model.pkl')
except FileNotFoundError:
    print("Warning: dengue_model.pkl not found. Please run generate_data.py to create it.")
    model = None  # placeholder to prevent crash

try:
    with open('dengue_dataset.pkl', 'rb') as f:
        master_df = pd.read_pickle(f)
except FileNotFoundError:
    print("Warning: dengue_dataset.pkl not found. Using empty dataset.")
    # Create empty DataFrame with required columns
    master_df = pd.DataFrame(columns=['WBC', 'Platelets', 'Hematocrit', 'Hemoglobin', 'Result'])

# Mock DB for demo
users = {"admin": "password123"}

# --- Utility: Visualization Generator ---
def generate_visualizations(df):
    vis_urls = []
    plt.switch_backend('Agg')  # Thread-safe for web apps

    # 1. Platelet Count Distribution
    if 'Platelets' in df and not df['Platelets'].empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['Platelets'], kde=True, color='red')
        plt.title('Patient Platelet Distribution')
        vis_urls.append(get_img_url())

    # 2. WBC vs Platelets Scatter (Risk Clusters)
    if 'WBC' in df and 'Platelets' in df:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            x='WBC', y='Platelets', hue='Result' if 'Result' in df else None,
            data=df, palette='viridis'
        )
        plt.title('WBC vs Platelet Count')
        vis_urls.append(get_img_url())

    # 3. Feature Correlation Heatmap
    if not df.empty:
        plt.figure(figsize=(6, 4))
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
        plt.title('CBC Parameter Correlation')
        vis_urls.append(get_img_url())

    return vis_urls

def get_img_url():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users[request.form['username']] = request.form['password']
        flash("Registration successful! Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        if u in users and users[u] == p:
            session['user'] = u
            return redirect(url_for('dashboard'))
        flash("Invalid credentials!")
    return render_template('login.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if model is None:
        flash("Model not loaded. Please generate it first.")
        return redirect(url_for('dashboard'))

    data = [float(request.form[f]) for f in ['wbc', 'platelets', 'hct', 'hb']]
    pred = model.predict([data])[0]
    prob = model.predict_proba([data])[0][1] * 100
    res_text = "Dengue POSITIVE" if pred == 1 else "Dengue NEGATIVE"
    return render_template('dashboard.html', pred_res=res_text, prob=f"{prob:.1f}%", stats={}, vis_urls=[])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    single_res = None
    vis_urls = []

    # Always start with the base dataset
    df_master = master_df.copy()

    if request.method == 'POST':
        # --- CASE A: Single Patient Prediction ---
        if 'wbc' in request.form:
            if model is None:
                flash("Model not loaded. Please generate it first.")
            else:
                wbc = float(request.form['wbc'])
                plt_count = float(request.form['platelets'])
                hct = float(request.form['hct'])
                hb = float(request.form['hb'])

                features = np.array([[wbc, plt_count, hct, hb]])
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1] * 100

                single_res = {
                    'status': "Positive" if prediction == 1 else "Negative",
                    'prob': round(probability, 2),
                    'color': "danger" if prediction == 1 else "success"
                }

        # --- CASE B: Dataset Upload for Reports ---
        elif 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename != '':
                df_uploaded = pd.read_csv(file)
                df_master = df_uploaded
                vis_urls = generate_visualizations(df_uploaded)

    # Calculate stats
    stats = {
        'total': len(df_master),
        'pos': len(df_master[df_master['Result'] == 1]) if 'Result' in df_master else 0,
        'avg_plt': round(df_master['Platelets'].mean(), 0) if 'Platelets' in df_master else 0
    }

    # Generate visuals if not generated
    if not vis_urls:
        vis_urls = generate_visualizations(df_master)

    return render_template(
        'dashboard.html',
        stats=stats,
        vis_urls=vis_urls,
        single_res=single_res
    )

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
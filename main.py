from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import pickle
import difflib
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import ast
import re
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Root@123',
    database='user_auth'
)
cursor = conn.cursor()

# Load models and data
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
dt_classifier = pickle.load(open('model/dt_classifier.pkl', 'rb'))

# Dictionaries
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76,
    'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83,
    'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95,
    'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Disease decoding dictionary
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
    30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
    5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}


# Helper functions
def match_symptom(user_input):
    matches = difflib.get_close_matches(user_input.strip().lower(), symptoms_dict.keys(), n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    # Convert to DataFrame with correct feature names (as during model training)
    input_df = pd.DataFrame([input_vector], columns=symptoms_dict.keys())

    prediction = dt_classifier.predict(input_df)[0]

    return diseases_list.get(prediction, "Unknown")


def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    diet = diets[diets['Disease'] == disease]['Diet'].values
    wrkout = workout[workout['disease'] == disease]['workout'].values
    return (
        " ".join(desc) if len(desc) else "No description available.",
        pre[0] if len(pre) else [],
        list(diet),
        list(wrkout)
    )

# Routes
@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/index')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html", username=session.get('name'))

@app.context_processor
def inject_home_link():
    home_url = url_for('index') if 'user_id' in session else url_for('index2')
    return dict(home_link=home_url)

# helper function 
def is_real_email(email):
    api_key = 'dc26baefe9364a699ae541675de5a291'  # API key
    url = f"https://emailvalidation.abstractapi.com/v1/?api_key={api_key}&email={email}"
    try:
        response = requests.get(url)
        data = response.json()
        # print(data) to debug
        return data.get("deliverability") == "DELIVERABLE"
    except Exception as e:
        print("Email check error:", e)
        return False


# Login & Registration
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'register':
            # Registration logic
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']

            # 🚫 Validate the email with Abstract API
            # Check if email is real
            if not is_real_email(email):
                flash("This email address doesn't seem to exist. Please use a valid one.", "error")
                return redirect(url_for('login'))

            cursor.execute("SELECT * FROM user WHERE email=%s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash("User already exists with this email. Please login.", "error")
                return render_template('login.html')

            hashed_password = generate_password_hash(password)
            cursor.execute("INSERT INTO user (name, email, password) VALUES (%s, %s, %s)",
                           (name, email, hashed_password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return render_template('login.html')

        elif form_type == 'login':
            # Login logic
            email = request.form['email']
            password = request.form['password']

            cursor.execute("SELECT user_id, name, password FROM user WHERE email=%s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['name'] = user[1]
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash("Invalid email or password.", "error")
                return render_template('login.html')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('index2'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        new_password = generate_password_hash(request.form['password'])

        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.execute("UPDATE user SET password = %s WHERE email = %s", (new_password, email))
            conn.commit()
            flash('Password updated successfully.')
            return redirect(url_for('login'))
        flash('Email not found.')
    return render_template('forgot-password.html')

@app.route('/diagnosis')
def diagnosis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template("diagnosis.html", symptoms=list(symptoms_dict.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    symptoms_input = request.form.get('symptoms')
    if not symptoms_input or symptoms_input.lower().strip() == "symptoms":
        return render_template('diagnosis.html',
                               message="Please provide valid symptoms.",
                               symptoms=list(symptoms_dict.keys()))

    user_symptoms = [s.strip().lower() for s in symptoms_input.split(',')]
    matched_symptoms = []
    unmatched_symptoms = []

    for symptom in user_symptoms:
        matched = match_symptom(symptom)
        if matched:
            matched_symptoms.append(matched)
        else:
            unmatched_symptoms.append(symptom)

    if not matched_symptoms:
        return render_template('diagnosis.html',
                               message="No recognizable symptoms found.",
                               symptoms=list(symptoms_dict.keys()))

    predicted_disease = get_predicted_value(matched_symptoms)
    dis_des, pre, diet_list, workout_list = helper(predicted_disease)

    # Save to history table
    cursor.execute("""
        INSERT INTO history (user_id, disease, symptoms, description, precautions, diet, workout)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        session['user_id'],
        predicted_disease,
        ', '.join(matched_symptoms),
        dis_des,
        ', '.join(pre),
        ', '.join(diet_list),
        ', '.join(workout_list)
    ))
    conn.commit()

    return render_template('diagnosis.html',
                           disease=predicted_disease,
                           description=dis_des,
                           precautions=', '.join(pre),
                           diet=', '.join(diet_list),
                           workout=', '.join(workout_list),
                           symptoms=list(symptoms_dict.keys()),
                           message=f"Ignored symptoms: {', '.join(unmatched_symptoms)}" if unmatched_symptoms else None)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cursor.execute("""
        SELECT disease, symptoms, description, precautions, diet, workout, created_at
        FROM history
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (session['user_id'],))
    rows = cursor.fetchall()

    history_data = []
    for row in rows:
        history_data.append({
            'disease': row[0],
            'symptoms': row[1],
            'description': row[2],
            'precautions': row[3],
            'diet': row[4],
            'workout': row[5],
            'date': row[6].strftime('%Y-%m-%d %H:%M:%S') if row[6] else 'N/A'
        })

    return render_template('history.html', username=session.get('name'), history=history_data)


def is_valid_email(email):
    return re.match(r'^[^@]+@[^@]+\.[^@]+$', email)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

if __name__ == '__main__':
    app.run(debug=True)

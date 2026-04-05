from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rainfall.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    prediction = db.Column(db.String(10), nullable=False)
    best_model = db.Column(db.String(50))
    best_accuracy = db.Column(db.Float)
    model_results = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')
        user = User(username=username, email=email,
                    password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash('Account created. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    history = (PredictionHistory.query
               .filter_by(user_id=session['user_id'])
               .order_by(PredictionHistory.created_at.desc())
               .limit(10).all())
    return render_template('dashboard.html', history=history, username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    location = data.get('location', '')
    lat = data.get('lat')
    lon = data.get('lon')
    if not location or lat is None or lon is None:
        return jsonify({'error': 'Location data missing'}), 400

    from ml.predictor import RainfallPredictor
    predictor = RainfallPredictor()
    result = predictor.run(lat, lon)

    if 'error' in result:
        return jsonify(result), 500

    record = PredictionHistory(
        user_id=session['user_id'],
        location=location,
        lat=lat,
        lon=lon,
        prediction=result['prediction'],
        best_model=result['best_model'],
        best_accuracy=round(result['model_results'][result['best_model']]['accuracy'], 4),
        model_results=json.dumps(result['model_results'])
    )
    db.session.add(record)
    db.session.commit()
    return jsonify(result)

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    records = (PredictionHistory.query
               .filter_by(user_id=session['user_id'])
               .order_by(PredictionHistory.created_at.desc())
               .all())
    for r in records:
        r.model_results_parsed = json.loads(r.model_results) if r.model_results else {}
    return render_template('history.html', records=records, username=session['username'])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template

# Flask uygulaması
app = Flask(__name__)

# Model ve veri yükleme
model = joblib.load('FraudDetectionModel.pkl')  # Model dosyasının yolu
df = pd.read_csv('creditcard.csv')  # Veri seti dosyasının yolu

# Time ve Amount sütunlarını ölçeklendirme
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Veriyi özellikler ve etiketler olarak ayırma
X = df.drop('Class', axis=1)
y = df['Class']

# Eğitim, validasyon ve test kümelerine ayırma
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

# Rastgele kullanıcı ID seçimi
@app.route('/get_random_user', methods=['POST'])
def get_random_user():
    random_index = np.random.choice(X_test.index)
    random_index = random_index % 42722
    print(type(random_index))
    return jsonify({'user_id': int(random_index)})


# Kullanıcı ID'ye göre tahmin
@app.route('/predict', methods=['POST'])
def predict():
    user_id = int(request.json['user_id'])
    random_sample = X_test.iloc[user_id]
    random_sample_true_label = y_test.iloc[user_id]

    random_sample_reshaped = random_sample.values.reshape(1, -1)
    random_sample_prediction = model.predict(random_sample_reshaped)[0]
    random_sample_prediction_proba = model.predict_proba(random_sample_reshaped)[0]

    result = {
        'user_id': int(user_id),
        'prediction': int(random_sample_prediction),
        'true_label': int(random_sample_true_label),
        'is_correct': bool(random_sample_prediction == random_sample_true_label),
        'probability': random_sample_prediction_proba.tolist()
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
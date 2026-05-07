from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, scaler, dan daftar nama kolom
model = joblib.load('student_performance_model.joblib')
scaler = joblib.load('student_scaler.joblib')

with open("model_columns.txt", "r") as f:
    model_columns = [line.strip() for line in f]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediksi_nilai = None
    
    if request.method == 'POST':
        # Mengambil input dari form HTML
        input_data = {}
        for col in model_columns:
            # Mengambil nilai dari form. Jika fiturnya tidak kita tampilkan di form web 
            # (agar form tidak terlalu panjang), kita beri nilai default 0
            val = request.form.get(col, 0)
            try:
                input_data[col] = [float(val)]
            except ValueError:
                input_data[col] = [0.0]

        # Ubah ke DataFrame agar sesuai dengan format input Scikit-Learn
        df_input = pd.DataFrame(input_data)
        
        # Scaling data input
        input_scaled = scaler.transform(df_input)
        
        # Prediksi
        hasil = model.predict(input_scaled)
        
        # Format hasil agar hanya 2 angka di belakang koma, dan maksimal 100
        prediksi_nilai = min(round(hasil[0], 2), 100.0)
        
    return render_template('index.html', prediksi_nilai=prediksi_nilai)

if __name__ == '__main__':
    app.run(debug=True)
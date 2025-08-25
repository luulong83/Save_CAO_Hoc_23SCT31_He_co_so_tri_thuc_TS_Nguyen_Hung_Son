from pyswip import Prolog
from flask import Flask, request, jsonify
from model import analyze_image  # Import từ model.py

app = Flask(__name__)

def integrate_ml_prolog(image_path, medical_history={}):
    # Bước 1: ML output
    ml_result = analyze_image(image_path)
    symptoms = ml_result["symptoms"]
    
    # Bước 2: Load Prolog
    prolog = Prolog()
    prolog.consult("medical_rules.pl")
    
    # Assert symptoms
    for sym in symptoms:
        prolog.assertz(f"has_symptom({sym})")
    
    # Assert từ history (ví dụ: {"fatigue": True})
    if medical_history.get("fatigue"):
        prolog.assertz("has_symptom(fatigue)")
    
    # Query
    results = list(prolog.query("diagnosis(D), treatment(D, T)"))
    if results:
        return {"diagnosis": results[0]['D'], "treatment": results[0]['T']}
    else:
        return {"diagnosis": "unknown", "treatment": "Tư vấn bác sĩ"}

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files['image']
    history = request.form.get('history', {})  # Có thể gửi JSON history
    image.save('temp.jpg')
    result = integrate_ml_prolog('temp.jpg', history)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
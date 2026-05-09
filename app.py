"""
WeldWise Flask Backend — Render-ready
--------------------------------------
pip install -r requirements.txt
Set env var: ANTHROPIC_API_KEY

Run locally:  python app.py
Render start: gunicorn app:app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, json, os, anthropic

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import joblib
def load_pkl(name):
    return joblib.load(os.path.join(BASE_DIR, name))

print("Loading models...")
reg_model         = load_pkl('weldwise_reg_model.pkl')
clf_model         = load_pkl('weldwise_clf_model.pkl')
material_encoder  = load_pkl('material_encoder.pkl')
condition_encoder = load_pkl('condition_encoder.pkl')
feature_cols      = load_pkl('feature_cols.pkl')
print("Models loaded. Columns:", feature_cols)

MATERIAL_RANGES = {
    'Mild Steel':      dict(current=(80,200),  voltage=(10,18), welding_speed=(100,350), gas_flow=(8,16),  wire_feed=(500,1800),  preheat_temp=(20,150), interpass_temp=(20,250)),
    'Stainless Steel': dict(current=(60,150),  voltage=(10,16), welding_speed=(80,300),  gas_flow=(8,16),  wire_feed=(400,1500),  preheat_temp=(20,100), interpass_temp=(20,150)),
    'Aluminum':        dict(current=(100,250), voltage=(12,20), welding_speed=(150,400), gas_flow=(10,20), wire_feed=(600,2000),  preheat_temp=(20,100), interpass_temp=(20,120)),
    'Titanium':        dict(current=(50,150),  voltage=(10,16), welding_speed=(80,250),  gas_flow=(10,20), wire_feed=(400,1200),  preheat_temp=(20,150), interpass_temp=(20,150)),
}

@app.route('/ranges', methods=['GET'])
def get_ranges():
    return jsonify(MATERIAL_RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        material       = d['material']
        current        = float(d['current'])
        voltage        = float(d['voltage'])
        welding_speed  = float(d['welding_speed'])
        gas_flow       = float(d['gas_flow'])
        wire_feed      = float(d['wire_feed'])
        preheat_temp   = float(d['preheat_temp'])
        interpass_temp = float(d['interpass_temp'])

        heat_input  = (current * voltage * 60) / (welding_speed * 1000)
        mat_encoded = material_encoder.transform([material])[0]

        # Exact column names confirmed from feature_cols.pkl
        feature_dict = {
            'Material_Encoded':     mat_encoded,
            'Current_A':            current,
            'Voltage_V':            voltage,
            'Welding_Speed_mm_min': welding_speed,
            'Gas_Flow_L_min':       gas_flow,
            'Wire_Feed_mm_min':     wire_feed,
            'Preheat_Temp_C':       preheat_temp,
            'Interpass_Temp_C':     interpass_temp,
            'Heat_Input_kJ_mm':     heat_input,
        }

        X = np.array([[feature_dict[col] for col in feature_cols]])

        reg_pred       = reg_model.predict(X)[0]
        ys, uts, elong = float(reg_pred[0]), float(reg_pred[1]), float(reg_pred[2])

        clf_pred   = clf_model.predict(X)[0]
        condition  = condition_encoder.inverse_transform([clf_pred])[0]
        confidence = round(float(max(clf_model.predict_proba(X)[0])) * 100, 1)

        return jsonify({
            'yield_strength': round(ys, 1),
            'uts':            round(uts, 1),
            'elongation':     round(elong, 2),
            'condition':      condition,
            'confidence':     confidence,
            'heat_input':     round(heat_input, 4),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        d           = request.json
        inputs      = d['inputs']
        predictions = d['predictions']
        warnings    = d.get('warnings', [])

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'ANTHROPIC_API_KEY not set'}), 500

        client    = anthropic.Anthropic(api_key=api_key)
        warn_text = f"\nWARNING — Out-of-range: {', '.join(warnings)}" if warnings else ""

        prompt = f"""You are a welding process expert analysing WeldWise ML results.

Material: {inputs['material']}
Parameters: Current {inputs['current']}A, Voltage {inputs['voltage']}V, Speed {inputs['welding_speed']}mm/min, Gas {inputs['gas_flow']}L/min, Wire {inputs['wire_feed']}mm/min, Preheat {inputs['preheat_temp']}C, Interpass {inputs['interpass_temp']}C, Heat Input {predictions['heat_input']}kJ/mm{warn_text}

ML Outputs: YS {predictions['yield_strength']}MPa, UTS {predictions['uts']}MPa, Elongation {predictions['elongation']}%, Condition {predictions['condition']} ({predictions['confidence']}% confidence)

Respond ONLY with valid JSON, no markdown or backticks:
{{"parameter_suggestions":"2-3 sentences on what to adjust and why.","property_analysis":"2-3 sentences on what YS/UTS/Elongation mean in practical terms.","application_recommendations":"2-3 sentences on suitable industries/uses and why."}}"""

        msg  = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip().replace('```json','').replace('```','').strip()
        return jsonify(json.loads(text))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

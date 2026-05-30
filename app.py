"""
WeldWise Flask Backend — Render-ready
--------------------------------------
pip install -r requirements.txt
Set env var: GROQ_API_KEY

Run locally:  python app.py
Render start: gunicorn app:app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, json, os
from groq import Groq

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

        api_key = os.environ.get('GROQ_API_KEY')
        if not api_key:
            return jsonify({'error': 'GROQ_API_KEY not set'}), 500

        client    = Groq(api_key=api_key)
        warn_text = f"\nWARNING — Out-of-range parameters: {', '.join(warnings)}" if warnings else ""

        mat   = inputs['material']
        cur   = inputs['current']
        vol   = inputs['voltage']
        spd   = inputs['welding_speed']
        gas   = inputs['gas_flow']
        wf    = inputs['wire_feed']
        pre   = inputs['preheat_temp']
        inp_t = inputs['interpass_temp']
        hi    = predictions['heat_input']
        ys    = predictions['yield_strength']
        uts   = predictions['uts']
        elong = predictions['elongation']
        cond  = predictions['condition']
        conf  = predictions['confidence']

        # Compute a base quality score to seed the model
        base_score = round(conf * (1.0 if cond == 'Good' else 0.5))

        prompt = f"""You are a welding process expert. Analyse the weld below and return ONLY a single valid JSON object. No markdown. No backticks. No explanation outside the JSON. Every field is REQUIRED — do not leave any field empty.

=== WELD DATA ===
Material: {mat}
Current: {cur}A | Voltage: {vol}V | Speed: {spd}mm/min | Gas: {gas}L/min | Wire Feed: {wf}mm/min | Preheat: {pre}C | Interpass: {inp_t}C | Heat Input: {hi}kJ/mm{warn_text}
YS: {ys}MPa | UTS: {uts}MPa | Elongation: {elong}% | Condition: {cond} | Confidence: {conf}%

=== OUTPUT FORMAT (copy this structure exactly, fill in all values) ===
{{
  "optimization_values": "• Current: [recommended A value]A\\n• Voltage: [recommended V value]V\\n• Wire Feed: [recommended value]mm/min\\n• Welding Speed: [recommended value]mm/min\\n• Gas Flow: [recommended value]L/min\\n\\n[One sentence: why these current/voltage changes help.]\\n[One sentence: effect on weld pool or penetration.]\\n[One sentence: what to watch out for.]",

  "property_analysis": "• Yield Strength ({ys}MPa): [what this value means for this material in one line]\\n• UTS ({uts}MPa): [what this value means for load-bearing in one line]\\n• Elongation ({elong}%): [what this ductility level means in one line]",

  "material_resistance": "• Heat Resistance: [specific temperature range or rating for {mat} welds]\\n• Electrical Conductivity: [conductivity level and implication for {mat}]\\n• Corrosion Resistance: [corrosion behaviour of {mat} in typical environments]\\n• Fatigue Resistance: [cyclic load behaviour at these strength levels]\\n• Wear Resistance: [surface wear characteristics for {mat}]",

  "material_grade": "• Likely Grade: [specific ASTM/ISO/AWS grade name for {mat} with these properties]\\n• Reason: [one line — which YS/UTS values point to this grade]\\n• Typical Use: [one line — what this grade is normally used for]",

  "weld_quality_score": "{base_score}",

  "application_recommendations": "• [Industry name]: [one line why this weld suits it]\\n• [Industry name]: [one line why this weld suits it]\\n• [Industry name]: [one line why this weld suits it]"
}}"""

        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.choices[0].message.content
        text = response_text.strip().replace('```json','').replace('```','').strip()
        return jsonify(json.loads(text))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

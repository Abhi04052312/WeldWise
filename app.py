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
from scipy.optimize import differential_evolution

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": [
        "https://weldwiseanalysis.netlify.app",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://*.netlify.app"
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"]
}})

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

# ── CORRECTED ranges matching the updated dataset ──────────
MATERIAL_RANGES = {
    'Mild Steel':      dict(current=(80,200),   voltage=(10,16),  welding_speed=(100,350), gas_flow=(8,15),   wire_feed=(500,1800),  preheat_temp=(20,150), interpass_temp=(20,250)),
    'Stainless Steel': dict(current=(60,144),   voltage=(10,14),  welding_speed=(80,300),  gas_flow=(8,15),   wire_feed=(400,1500),  preheat_temp=(20,100), interpass_temp=(20,150)),
    'Aluminum':        dict(current=(105,260),  voltage=(14,20),  welding_speed=(150,400), gas_flow=(10,20),  wire_feed=(600,2000),  preheat_temp=(20,100), interpass_temp=(20,120)),
    'Titanium':        dict(current=(50,120),   voltage=(10,15),  welding_speed=(80,250),  gas_flow=(12,25),  wire_feed=(400,1200),  preheat_temp=(20,150), interpass_temp=(20,150)),
}

# Ordered param list — must stay in sync with MATERIAL_RANGES keys
PARAM_ORDER = ['current', 'voltage', 'welding_speed', 'gas_flow', 'wire_feed', 'preheat_temp', 'interpass_temp']


def build_feature_vector(material, params):
    """Build the numpy feature vector from a params dict."""
    current        = params['current']
    voltage        = params['voltage']
    welding_speed  = params['welding_speed']
    gas_flow       = params['gas_flow']
    wire_feed      = params['wire_feed']
    preheat_temp   = params['preheat_temp']
    interpass_temp = params['interpass_temp']
    heat_input     = (current * voltage * 60) / (welding_speed * 1000)
    mat_encoded    = material_encoder.transform([material])[0]

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
    return np.array([[feature_dict[col] for col in feature_cols]]), heat_input


def score_params(material, params):
    """
    Returns a quality score 0–100 for a given parameter set.
    Higher is better. Used as the optimizer objective.
    """
    X, _ = build_feature_vector(material, params)

    # Classification confidence toward 'Good'
    proba      = clf_model.predict_proba(X)[0]
    clf_pred   = clf_model.predict(X)[0]
    condition  = condition_encoder.inverse_transform([clf_pred])[0]

    # Find which index corresponds to 'Good'
    good_idx   = list(condition_encoder.classes_).index('Good')
    good_proba = proba[good_idx]

    # Regression outputs — normalise each to 0–1 within material range
    reg_pred   = reg_model.predict(X)[0]
    ys, uts, el = float(reg_pred[0]), float(reg_pred[1]), float(reg_pred[2])

    r = MATERIAL_RANGES[material]
    # Use dataset output ranges for normalisation
    OUTPUT_RANGES = {
        'Mild Steel':      dict(ys=(250,400),  uts=(400,550),  el=(15,30)),
        'Stainless Steel': dict(ys=(200,480),  uts=(500,700),  el=(20,35)),
        'Aluminum':        dict(ys=(100,280),  uts=(150,290),  el=(8,18)),
        'Titanium':        dict(ys=(700,900),  uts=(800,1050), el=(10,18)),
    }
    out_r = OUTPUT_RANGES[material]

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))

    ys_norm  = norm(ys,  *out_r['ys'])
    uts_norm = norm(uts, *out_r['uts'])
    el_norm  = norm(el,  *out_r['el'])

    # Weighted score: 50% on Good probability, 50% on mechanical properties
    mech_score = (ys_norm * 0.35 + uts_norm * 0.40 + el_norm * 0.25)
    total      = good_proba * 0.50 + mech_score * 0.50
    return round(total * 100, 1)


@app.route('/ranges', methods=['GET'])
def get_ranges():
    return jsonify(MATERIAL_RANGES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        material       = d['material']
        params = {
            'current':        float(d['current']),
            'voltage':        float(d['voltage']),
            'welding_speed':  float(d['welding_speed']),
            'gas_flow':       float(d['gas_flow']),
            'wire_feed':      float(d['wire_feed']),
            'preheat_temp':   float(d['preheat_temp']),
            'interpass_temp': float(d['interpass_temp']),
        }

        X, heat_input = build_feature_vector(material, params)

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


@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Runs scipy differential_evolution against the XGBoost model to find
    the mathematically optimal parameter set.

    Accepts:
      - material: string
      - locked_params: dict of {param_name: value} for parameters to keep fixed
        e.g. {"voltage": 14, "preheat_temp": 80}
      - current_params: dict of current input values (used as fallback for locked)

    Returns:
      - optimized_params: the best parameter set found
      - optimized_score: quality score 0–100
      - optimized_prediction: full prediction on optimized params
      - improvement: score delta vs original
      - original_score: score of the original inputs
    """
    try:
        d             = request.json
        material      = d['material']
        locked_params = d.get('locked_params', {})   # {param: value}
        current_p     = d.get('current_params', {})  # original inputs

        r = MATERIAL_RANGES[material]

        # Build bounds — locked params have zero-width range (fixed point)
        free_params  = []  # params the optimizer can change
        fixed_params = {}  # locked params with their fixed values

        for p in PARAM_ORDER:
            if p in locked_params:
                fixed_params[p] = float(locked_params[p])
            else:
                free_params.append(p)

        bounds = [r[p] for p in free_params]

        def objective(x):
            # Reconstruct full param dict from free vars + fixed vars
            params = dict(fixed_params)
            for i, p in enumerate(free_params):
                params[p] = x[i]
            # Negate because differential_evolution minimises
            return -score_params(material, params)

        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=12,
            tol=0.001,
            polish=True,
        )

        # Build optimized full params dict
        opt_params = dict(fixed_params)
        for i, p in enumerate(free_params):
            opt_params[p] = round(float(result.x[i]), 2)

        opt_score = score_params(material, opt_params)

        # Get full prediction on optimized params
        X_opt, hi_opt = build_feature_vector(material, opt_params)
        reg_pred      = reg_model.predict(X_opt)[0]
        clf_pred      = clf_model.predict(X_opt)[0]
        condition     = condition_encoder.inverse_transform([clf_pred])[0]
        confidence    = round(float(max(clf_model.predict_proba(X_opt)[0])) * 100, 1)

        opt_prediction = {
            'yield_strength': round(float(reg_pred[0]), 1),
            'uts':            round(float(reg_pred[1]), 1),
            'elongation':     round(float(reg_pred[2]), 2),
            'condition':      condition,
            'confidence':     confidence,
            'heat_input':     round(hi_opt, 4),
        }

        # Original score for comparison
        if current_p:
            orig_score = score_params(material, {
                p: float(current_p.get(p, opt_params[p])) for p in PARAM_ORDER
            })
        else:
            orig_score = None

        return jsonify({
            'optimized_params':     opt_params,
            'optimized_score':      opt_score,
            'optimized_prediction': opt_prediction,
            'original_score':       orig_score,
            'improvement':          round(opt_score - orig_score, 1) if orig_score is not None else None,
            'locked_params':        list(fixed_params.keys()),
            'free_params':          free_params,
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

  "application_recommendations": [
    {{"name": "[Industry name]", "short": "[one sentence why this weld suits it]", "detail": "[two to three sentences with specific use cases, component examples, and why this material/strength profile works here]"}},
    {{"name": "[Industry name]", "short": "[one sentence why this weld suits it]", "detail": "[two to three sentences with specific use cases, component examples, and why this material/strength profile works here]"}},
    {{"name": "[Industry name]", "short": "[one sentence why this weld suits it]", "detail": "[two to three sentences with specific use cases, component examples, and why this material/strength profile works here]"}},
    {{"name": "[Industry name]", "short": "[one sentence why this weld suits it]", "detail": "[two to three sentences with specific use cases, component examples, and why this material/strength profile works here]"}}
  ]
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

"""
chatbot_cli.py - terminal version of the MediPath chatbot (no Streamlit needed)
Run:  python chatbot_cli.py
"""
import re
from mdp_engine import generate_patient_data, estimate_mdp_from_data, policy_iteration, q_table, triage

GAMMA = 0.9
P, R = estimate_mdp_from_data(generate_patient_data())
policy, V, _ = policy_iteration(P, R, GAMMA)
Q = q_table(V, P, R, GAMMA)


def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


print("MediPath 🩺  (type 'quit' to exit) — educational prototype, not medical advice")
print("Optimal policy:", policy)
while True:
    msg = input("\nYou: ").strip().lower()
    if msg in ('quit', 'exit', 'q'):
        break
    vit = dict(temperature=num(r'(\d{2}(?:\.\d)?)\s*(?:°|c\b|deg)', msg) or num(r'(?:fever|temp\w*)\D{0,10}(\d{2}(?:\.\d)?)', msg),
               heart_rate=num(r'(?:pulse|hr|heart rate)\D{0,10}(\d{2,3})', msg) or num(r'(\d{2,3})\s*bpm', msg),
               spo2=num(r'(?:oxygen|spo2|o2)\D{0,10}(\d{2,3})', msg),
               pain=num(r'pain\D{0,10}(\d{1,2})\s*/\s*10', msg))
    state, why = triage(text=msg, **vit)
    a = policy[state]
    print(f"Bot: State -> {state} ({', '.join(why)})")
    print(f"     Recommended treatment -> {a}")
    print("     Q-values:", {k: round(v, 1) for k, v in Q.loc[state].items()})
    print("     Next-state odds:", {k: f"{p*100:.0f}%" for k, p in P[state][a].items()})
    if state == 'Critical':
        print("     🚨 Real emergency? Call emergency services now.")

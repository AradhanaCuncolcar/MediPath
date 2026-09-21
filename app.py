"""
app.py - MediPath: MDP-powered treatment decision chatbot
Run:  streamlit run app.py
"""
import re
import streamlit as st
import pandas as pd
from mdp_engine import (states, actions, generate_patient_data, estimate_mdp_from_data, TRUE_P, TRUE_R,
                        policy_iteration, value_iteration, q_table, compare_policies, gamma_sensitivity,
                        triage, build_reference_graph, build_policy_graph)

st.set_page_config(page_title="MediPath – Treatment Decision Engine", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🩺 MediPath")
st.sidebar.caption("MDP decision engine for treatment planning")
gamma = st.sidebar.slider("Discount factor γ (how much future health matters)", 0.1, 0.99, 0.9, 0.01)
source = st.sidebar.radio("Model source", ["Learned from patient data", "Expert-defined model"])
n_patients = st.sidebar.slider("Synthetic patients", 100, 2000, 500, 100)
st.sidebar.warning("Educational prototype. Not medical advice.")


@st.cache_data
def load(n, src):
    df = generate_patient_data(n_patients=n)
    P, R = estimate_mdp_from_data(df) if src.startswith("Learned") else (TRUE_P, TRUE_R)
    return df, P, R


df, P, R = load(n_patients, source)
policy, V, history = policy_iteration(P, R, gamma)
vi_policy, _ = value_iteration(P, R, gamma)
Q = q_table(V, P, R, gamma)

EMOJI = {'Healthy': '🟢', 'Sick': '🟠', 'Critical': '🔴'}
ADVICE = {'No Treatment': "Monitor at home, rest, hydrate, and re-check if symptoms appear.",
          'Medication': "Start a medication course prescribed by a doctor and review in 48–72 hours.",
          'Surgery': "Escalate to emergency / surgical care immediately."}


def recommend(state, reasons):
    a = policy[state]
    q = Q.loc[state].sort_values(ascending=False)
    runner = q.index[1]
    nxt = P[state][a]
    lines = [
        f"**Assessed state:** {EMOJI[state]} **{state}**  _(based on: {', '.join(reasons)})_",
        f"**Recommended treatment:** 💊 **{a}** — {ADVICE[a]}",
        f"**Why:** it has the highest long-term expected value ({q.iloc[0]:.1f}) vs. "
        f"{runner} ({q.iloc[1]:.1f}) at γ = {gamma}.",
        "**Expected next state:** " + ", ".join(f"{EMOJI[s]} {s} {p*100:.0f}%" for s, p in nxt.items()),
    ]
    if state == 'Critical':
        lines.append("🚨 **If this is a real emergency, call your local emergency number now (998 ambulance in the UAE).**")
    return "\n\n".join(lines)


def parse_vitals(text):
    """Pull numbers like '38.5C', 'pulse 120', '130 bpm', 'oxygen 89', 'pain 7/10' out of free text."""
    t = text.lower()

    def grab(*patterns):
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                return float(m.group(1))
        return None

    return dict(temperature=grab(r'(\d{2}(?:\.\d)?)\s*(?:°|deg|c\b|celsius)', r'(?:fever|temp\w*)\D{0,10}(\d{2}(?:\.\d)?)'),
                heart_rate=grab(r'(?:hr|heart rate|pulse)\D{0,10}(\d{2,3})', r'(\d{2,3})\s*bpm'),
                spo2=grab(r'(?:spo2|oxygen|o2|saturation)\D{0,10}(\d{2,3})'),
                pain=grab(r'pain\D{0,10}(\d{1,2})\s*/\s*10', r'pain (?:is |of |level )?(\d{1,2})\b'))


tab_chat, tab_graph, tab_policy, tab_data = st.tabs(["💬 Chatbot", "🗺️ MDP graph", "📊 Policy & outcomes", "📁 Patient data"])

# ------------------------------------------------------------------ chatbot
with tab_chat:
    st.subheader("Describe how the patient is feeling")
    st.caption("Examples: “I have a fever of 38.5 and a cough” · “chest pain, oxygen 89, pulse 130” · “feeling fine, routine checkup”")
    if "msgs" not in st.session_state:
        st.session_state.msgs = [{"role": "assistant", "content":
                                  "Hi, I'm MediPath. Tell me the symptoms and any vitals (temperature, pulse, oxygen, pain 0–10) and I'll suggest the optimal treatment plan."}]

    with st.expander("Or enter vitals directly"):
        c1, c2, c3, c4 = st.columns(4)
        temp = c1.number_input("Temperature °C", 34.0, 42.0, 36.8, 0.1)
        hr = c2.number_input("Heart rate bpm", 30, 200, 75)
        spo2 = c3.number_input("SpO₂ %", 70, 100, 98)
        pain = c4.slider("Pain 0–10", 0, 10, 1)
        if st.button("Assess vitals"):
            s, why = triage(temperature=temp, heart_rate=hr, spo2=spo2, pain=pain)
            st.session_state.msgs += [{"role": "user", "content": f"Temp {temp}°C, HR {hr}, SpO₂ {spo2}%, pain {pain}/10"},
                                      {"role": "assistant", "content": recommend(s, why)}]

    for m in st.session_state.msgs:
        st.chat_message(m["role"]).markdown(m["content"])

    if prompt := st.chat_input("Type symptoms…"):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        s, why = triage(text=prompt, **parse_vitals(prompt))
        reply = recommend(s, why)
        st.session_state.msgs.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)

# ------------------------------------------------------------------ graph
with tab_graph:
    st.subheader("Optimal-policy view")
    st.graphviz_chart(build_policy_graph(P, R, policy))
    st.subheader("Reference-style flowchart (action (probability), R: reward)")
    st.graphviz_chart(build_reference_graph(P, R))

# ------------------------------------------------------------------ policy
with tab_policy:
    c1, c2, c3 = st.columns(3)
    for col, s in zip((c1, c2, c3), states):
        col.metric(f"{EMOJI[s]} {s}", policy[s], f"V = {V[s]:.1f}")
    st.success(f"Policy iteration converged in {len(history)} evaluation steps. "
               f"Value iteration agrees: {policy == vi_policy}.")
    st.markdown("**Policy iteration steps**")
    st.dataframe(pd.DataFrame([{**h['policy'], **{f"V({k})": v for k, v in h['V'].items()}} for h in history]))
    st.markdown("**Q-values: expected long-term reward of each treatment**")
    st.bar_chart(Q)
    st.dataframe(Q.round(2).style.highlight_max(axis=1, color="#c8f7c5"))
    st.markdown("**Simulated outcomes: 2000 sick patients over 30 days**")
    st.dataframe(compare_policies(policy, P, R, gamma))
    st.markdown("**Does the policy change if we care less about the future?**")
    st.dataframe(gamma_sensitivity(P, R))
    st.info("With a small γ the engine avoids costly treatments because it only values immediate reward. "
            "Surgery for critical patients only becomes optimal once long-term recovery is weighted enough (γ ≥ ~0.7).")

# ------------------------------------------------------------------ data
with tab_data:
    st.write(f"{len(df):,} treatment records · {df.patient_id.nunique()} patients")
    st.dataframe(df.head(200))
    st.markdown("**Transition matrix learned from data**")
    st.dataframe(pd.DataFrame({(s, a): P[s][a] for s in states for a in actions}).T)
    st.download_button("Download patient data (CSV)", df.to_csv(index=False), "patient_data.csv")

"""
mdp_engine.py
Healthcare Treatment MDP - core engine for the chatbot decision engine.

States  : health condition of the patient   -> Healthy, Sick, Critical
Actions : treatment options                  -> No Treatment, Medication, Surgery
P[s][a] : probability of the next health state after one treatment cycle
R[s][a] : immediate reward = quality-of-life of current state - cost/risk of treatment

Pipeline: expert model -> synthetic patient records -> P, R estimated from data
          -> policy iteration (+ value iteration check) -> simulation -> graphs
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 1. MDP components
states = ['Healthy', 'Sick', 'Critical']
actions = ['No Treatment', 'Medication', 'Surgery']

# Expert ("ground truth") transition model used to generate the synthetic hospital data.
# Reasoning behind the numbers:
#  - Healthy: doing nothing keeps you healthy; surgery on a healthy person adds risk.
#  - Sick: medication is the most effective; untreated illness can escalate.
#  - Critical: untreated critical patients rarely recover; surgery gives the best recovery odds.
TRUE_P = {
    'Healthy':  {'No Treatment': {'Healthy': 0.90, 'Sick': 0.08, 'Critical': 0.02},
                 'Medication':   {'Healthy': 0.92, 'Sick': 0.07, 'Critical': 0.01},
                 'Surgery':      {'Healthy': 0.80, 'Sick': 0.15, 'Critical': 0.05}},
    'Sick':     {'No Treatment': {'Healthy': 0.10, 'Sick': 0.60, 'Critical': 0.30},
                 'Medication':   {'Healthy': 0.60, 'Sick': 0.30, 'Critical': 0.10},
                 'Surgery':      {'Healthy': 0.50, 'Sick': 0.30, 'Critical': 0.20}},
    'Critical': {'No Treatment': {'Healthy': 0.00, 'Sick': 0.10, 'Critical': 0.90},
                 'Medication':   {'Healthy': 0.05, 'Sick': 0.35, 'Critical': 0.60},
                 'Surgery':      {'Healthy': 0.30, 'Sick': 0.45, 'Critical': 0.25}},
}

# Reward design: R(s, a) = quality of life in s  - treatment cost/risk of a
QUALITY_OF_LIFE = {'Healthy': 10, 'Sick': -2, 'Critical': -10}
TREATMENT_COST = {'No Treatment': 0, 'Medication': 2, 'Surgery': 8}
TRUE_R = {s: {a: QUALITY_OF_LIFE[s] - TREATMENT_COST[a] for a in actions} for s in states}

# How clinicians in the synthetic hospital choose treatments (behaviour policy).
# Every action has non-zero probability so every (state, action) pair appears in the data.
CLINICIAN_POLICY = {
    'Healthy':  {'No Treatment': 0.70, 'Medication': 0.25, 'Surgery': 0.05},
    'Sick':     {'No Treatment': 0.25, 'Medication': 0.55, 'Surgery': 0.20},
    'Critical': {'No Treatment': 0.15, 'Medication': 0.40, 'Surgery': 0.45},
}

# Typical vitals per state (mean, std) -> used for data + chatbot triage
VITALS = {
    'Healthy':  {'temperature': (36.8, 0.3), 'heart_rate': (72, 8),  'spo2': (98, 1.0), 'pain': (1, 1)},
    'Sick':     {'temperature': (38.2, 0.5), 'heart_rate': (95, 10), 'spo2': (95, 1.5), 'pain': (5, 1.5)},
    'Critical': {'temperature': (39.5, 0.7), 'heart_rate': (125, 12), 'spo2': (88, 3.0), 'pain': (8, 1.2)},
}


# ---------------------------------------------------------------- 2. Synthetic data
def generate_patient_data(n_patients=500, n_days=30, seed=42):
    """Simulate hospital records: one row = one treatment decision for one patient on one day."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        age = int(rng.integers(18, 90))
        state = rng.choice(states, p=[0.5, 0.35, 0.15])
        for day in range(1, n_days + 1):
            pol = CLINICIAN_POLICY[state]
            action = rng.choice(actions, p=list(pol.values()))
            probs = TRUE_P[state][action]
            next_state = rng.choice(list(probs.keys()), p=list(probs.values()))
            reward = TRUE_R[state][action] + rng.normal(0, 1.0)   # noisy observed outcome
            v = VITALS[state]
            rows.append({
                'patient_id': f'P{pid:04d}', 'day': day, 'age': age,
                'temperature': round(rng.normal(*v['temperature']), 1),
                'heart_rate': int(rng.normal(*v['heart_rate'])),
                'spo2': round(min(100, rng.normal(*v['spo2'])), 1),
                'pain_score': int(np.clip(rng.normal(*v['pain']), 0, 10)),
                'state': state, 'action': action, 'next_state': next_state,
                'treatment_cost': TREATMENT_COST[action], 'reward': round(reward, 2),
            })
            state = next_state
    return pd.DataFrame(rows)


def estimate_mdp_from_data(df):
    """Maximum-likelihood estimates: P = transition counts / visits, R = mean observed reward."""
    P, R = {}, {}
    for s in states:
        P[s], R[s] = {}, {}
        for a in actions:
            sub = df[(df.state == s) & (df.action == a)]
            counts = sub.next_state.value_counts()
            total = counts.sum()
            P[s][a] = {s2: float(round(counts.get(s2, 0) / total, 3)) for s2 in states} if total else dict(TRUE_P[s][a])
            R[s][a] = float(round(sub.reward.mean(), 2)) if total else TRUE_R[s][a]
    return P, R


# ---------------------------------------------------------------- 3. Policy iteration
def q_value(s, a, V, P, R, gamma):
    return sum(P[s][a].get(s2, 0) * (R[s][a] + gamma * V[s2]) for s2 in P[s][a])


def policy_evaluation(policy, P, R, gamma=0.9, theta=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = q_value(s, policy[s], V, P, R, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            return V


def policy_iteration(P, R, gamma=0.9):
    """Returns optimal policy, state values and a history of each improvement step."""
    policy = {s: 'No Treatment' for s in states}   # deterministic start (reproducible)
    history = []
    while True:
        V = policy_evaluation(policy, P, R, gamma)
        history.append({'policy': dict(policy), 'V': {s: float(round(V[s], 2)) for s in states}})
        stable = True
        for s in states:
            old = policy[s]
            policy[s] = max(actions, key=lambda a: q_value(s, a, V, P, R, gamma))
            if old != policy[s]:
                stable = False
        if stable:
            return policy, {s: float(v) for s, v in V.items()}, history


def value_iteration(P, R, gamma=0.9, theta=1e-6):
    """Cross-check for policy iteration: both must give the same optimal policy."""
    V = {s: 0.0 for s in states}
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(q_value(s, a, V, P, R, gamma) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = {s: max(actions, key=lambda a: q_value(s, a, V, P, R, gamma)) for s in states}
    return policy, {s: float(v) for s, v in V.items()}


def q_table(V, P, R, gamma=0.9):
    return pd.DataFrame({a: [q_value(s, a, V, P, R, gamma) for s in states] for a in actions}, index=states)


# ---------------------------------------------------------------- 4. Evaluate outcomes
def simulate_policy(policy, P, R, n_patients=2000, n_days=30, gamma=0.9, start='Sick', seed=7):
    """policy: dict state->action, or the string 'random'."""
    rng = np.random.default_rng(seed)
    ret, healthy_days, cost, end_critical = [], [], [], 0
    for _ in range(n_patients):
        s, G, h, c = start, 0.0, 0, 0
        for t in range(n_days):
            a = rng.choice(actions) if policy == 'random' else policy[s]
            G += (gamma ** t) * R[s][a]
            c += TREATMENT_COST[a]
            h += s == 'Healthy'
            nxt = P[s][a]
            s = rng.choice(list(nxt.keys()), p=np.array(list(nxt.values())) / sum(nxt.values()))
        ret.append(G); healthy_days.append(h); cost.append(c); end_critical += s == 'Critical'
    return {'avg_discounted_reward': np.mean(ret), 'pct_days_healthy': 100 * np.mean(healthy_days) / n_days,
            'avg_treatment_cost': np.mean(cost), 'pct_end_critical': 100 * end_critical / n_patients}


def compare_policies(optimal, P, R, gamma=0.9):
    candidates = {'Optimal (MDP)': optimal, 'Always No Treatment': {s: 'No Treatment' for s in states},
                  'Always Medication': {s: 'Medication' for s in states},
                  'Always Surgery': {s: 'Surgery' for s in states}, 'Random': 'random'}
    return pd.DataFrame({k: simulate_policy(v, P, R, gamma=gamma) for k, v in candidates.items()}).T.round(2)


def gamma_sensitivity(P, R, gammas=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)):
    return pd.DataFrame({g: policy_iteration(P, R, g)[0] for g in gammas}).T.rename_axis('gamma')


# ---------------------------------------------------------------- 5. Chatbot triage
CRITICAL_WORDS = ['unconscious', 'chest pain', 'can\'t breathe', 'cannot breathe', 'difficulty breathing',
                  'seizure', 'bleeding heavily', 'stroke', 'collapsed', 'severe', 'fainted', 'blue lips']
SICK_WORDS = ['fever', 'cough', 'pain', 'headache', 'vomit', 'nausea', 'infection', 'flu', 'cold',
              'sore', 'dizzy', 'rash', 'diarrhea', 'tired', 'weak', 'sick', 'ill']
HEALTHY_WORDS = ['fine', 'healthy', 'checkup', 'check-up', 'good', 'normal', 'no symptoms', 'great']


def triage(text=None, temperature=None, heart_rate=None, spo2=None, pain=None):
    """Map symptoms/vitals to an MDP state. Returns (state, list of reasons)."""
    score, reasons = 0, []
    t = (text or '').lower()
    if any(w in t for w in CRITICAL_WORDS):
        score = max(score, 2); reasons.append('red-flag symptom mentioned')
    elif any(w in t for w in SICK_WORDS):
        score = max(score, 1); reasons.append('illness symptoms mentioned')
    if temperature is not None:
        if temperature >= 39.5 or temperature < 35: score = max(score, 2); reasons.append(f'temperature {temperature}°C')
        elif temperature >= 37.8: score = max(score, 1); reasons.append(f'fever {temperature}°C')
    if heart_rate is not None:
        if heart_rate >= 120 or heart_rate < 45: score = max(score, 2); reasons.append(f'heart rate {heart_rate} bpm')
        elif heart_rate >= 100: score = max(score, 1); reasons.append(f'elevated heart rate {heart_rate} bpm')
    if spo2 is not None:
        if spo2 < 92: score = max(score, 2); reasons.append(f'low oxygen {spo2}%')
        elif spo2 < 95: score = max(score, 1); reasons.append(f'borderline oxygen {spo2}%')
    if pain is not None:
        if pain >= 8: score = max(score, 2); reasons.append(f'severe pain {pain}/10')
        elif pain >= 4: score = max(score, 1); reasons.append(f'moderate pain {pain}/10')
    if not reasons:
        reasons.append('no warning signs found')
    return states[score], reasons


# ---------------------------------------------------------------- 6. Graphs
def build_reference_graph(P, R, hide_zero=True):
    """Same style as the class reference graph: edge label = 'action (prob), R: reward'.
    Parallel edges between the same two states are merged into one edge with a multi-line label."""
    from graphviz import Digraph
    dot = Digraph(graph_attr={'rankdir': 'TB', 'nodesep': '1.2', 'ranksep': '1.4'},
                  node_attr={'shape': 'ellipse', 'fontsize': '14'},
                  edge_attr={'fontsize': '10'})
    for s in states:
        dot.node(s)
    for s in P:
        for s2 in states:
            labels = [f"{a} ({P[s][a].get(s2, 0)}),  R: {R[s][a]}"
                      for a in P[s]
                      if not (hide_zero and P[s][a].get(s2, 0) == 0)]
            if labels:
                dot.edge(s, s2, label="\n".join(labels))
    return dot


ACTION_COLORS = {'No Treatment': '#7a8691', 'Medication': '#1f78b4', 'Surgery': '#c0392b'}
STATE_COLORS = {'Healthy': '#d5f5e3', 'Sick': '#fdebd0', 'Critical': '#fadbd8'}


def build_policy_graph(P, R, policy):
    """Readable version: colour per action, optimal-policy edges bold."""
    from graphviz import Digraph
    dot = Digraph(graph_attr={'rankdir': 'LR', 'label': 'Healthcare MDP - bold edges = optimal policy',
                              'labelloc': 't', 'fontsize': '16', 'nodesep': '0.8', 'ranksep': '1.6'})
    for s in states:
        dot.node(s, f"{s}\\n★ {policy[s]}", style='filled', fillcolor=STATE_COLORS[s], shape='ellipse', fontsize='13')
    for s in states:
        for a in actions:
            best = policy[s] == a
            for s2, prob in P[s][a].items():
                if prob == 0:
                    continue
                dot.edge(s, s2, label=f"{a} ({prob}) R:{R[s][a]}", color=ACTION_COLORS[a],
                         fontcolor=ACTION_COLORS[a], penwidth='3' if best else '0.8',
                         style='solid' if best else 'dashed', fontsize='9')
    return dot

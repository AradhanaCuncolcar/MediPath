"""
run_pipeline.py - run the full healthcare MDP pipeline and save every output to ./outputs
Usage:  python run_pipeline.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mdp_engine import *

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)
GAMMA = 0.9
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)

# 1. Data
df = generate_patient_data()
df.to_csv(f'{OUT}/patient_data.csv', index=False)
print(f"Generated {len(df)} treatment records for {df.patient_id.nunique()} patients")

# 2. Estimate P and R from data
P, R = estimate_mdp_from_data(df)
json.dump({'P': P, 'R': R}, open(f'{OUT}/mdp_model.json', 'w'), indent=2)
print("\nEstimated transition probabilities P:")
for s in states:
    for a in actions:
        print(f"  {s:8s} + {a:12s} -> {P[s][a]}")
print("\nEstimated rewards R:\n", pd.DataFrame(R).T)

# 3. Policy iteration (+ value iteration check)
policy, V, history = policy_iteration(P, R, GAMMA)
vi_policy, vi_V = value_iteration(P, R, GAMMA)
print("\nPolicy iteration steps:")
for i, h in enumerate(history):
    print(f"  iter {i}: {h['policy']}  V={h['V']}")
print("\nOptimal Policy (policy iteration):", policy)
print("State Values:", {s: round(v, 2) for s, v in V.items()})
print("Optimal Policy (value iteration): ", vi_policy)
print("Both methods agree:", policy == vi_policy)

Q = q_table(V, P, R, GAMMA)
print("\nQ-values (expected long-term reward of each treatment):\n", Q.round(2))

# 4. Outcome analysis
cmp = compare_policies(policy, P, R, GAMMA)
print("\nPolicy comparison (2000 simulated sick patients, 30 days):\n", cmp)
sens = gamma_sensitivity(P, R)
print("\nDiscount factor sensitivity:\n", sens)

# 5. Graphs
build_reference_graph(P, R).render(f'{OUT}/mdp_flowchart', format='png', cleanup=True)
build_policy_graph(P, R, policy).render(f'{OUT}/mdp_policy_graph', format='png', cleanup=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
cmp['avg_discounted_reward'].plot.barh(ax=axes[0], color=['#27ae60'] + ['#95a5a6'] * 4, title='Avg discounted reward (higher = better)')
cmp['pct_days_healthy'].plot.barh(ax=axes[1], color=['#27ae60'] + ['#95a5a6'] * 4, title='% of days healthy')
cmp['pct_end_critical'].plot.barh(ax=axes[2], color=['#27ae60'] + ['#95a5a6'] * 4, title='% ending in Critical (lower = better)')
for ax in axes: ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f'{OUT}/policy_comparison.png', dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
Q.plot.bar(ax=ax, color=[ACTION_COLORS[a] for a in actions], rot=0, title='Q-values per state and treatment')
ax.axhline(0, color='k', lw=0.6); ax.set_ylabel('Expected discounted reward')
plt.tight_layout(); plt.savefig(f'{OUT}/q_values.png', dpi=130); plt.close()

# 6. Text report
with open(f'{OUT}/results.txt', 'w') as f:
    f.write(f"OPTIMAL TREATMENT POLICY (gamma={GAMMA})\n{policy}\n\nSTATE VALUES\n{V}\n\n")
    f.write(f"Q-VALUES\n{Q.round(2)}\n\nPOLICY COMPARISON\n{cmp}\n\nGAMMA SENSITIVITY\n{sens}\n")
print(f"\nAll outputs saved in ./{OUT}/")

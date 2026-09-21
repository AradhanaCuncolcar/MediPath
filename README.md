# 🩺 MediPath – MDP Chatbot Decision Engine for Healthcare

A chatbot that reads a patient's symptoms/vitals, maps them to a health state, and recommends the
**optimal treatment** found by **policy iteration** on a Markov Decision Process.

## MDP definition
| Component | Definition |
|---|---|
| States | `Healthy`, `Sick`, `Critical` |
| Actions | `No Treatment`, `Medication`, `Surgery` |
| P(s'|s,a) | Learned (MLE counts) from 15,000 synthetic patient records |
| R(s,a) | Quality of life of state (+10 / −2 / −10) − treatment cost (0 / 2 / 8) |
| γ | 0.9 (future health matters) |

## Result (γ = 0.9)
| State | Optimal treatment | V(s) |
|---|---|---|
| Healthy | No Treatment | 76.0 |
| Sick | Medication | 53.4 |
| Critical | Surgery | 29.8 |

Policy iteration converges in 2 steps; value iteration gives the same policy.

**Simulated 2000 sick patients for 30 days:** the MDP policy has the best reward (49.0), 79% healthy days,
4.8% end critical, at ~⅓ the cost of "always medicate". No treatment → 63% end critical.

**Critical thinking:** with γ = 0.1 the engine chooses *no treatment everywhere* (costs are immediate,
benefits delayed); surgery for critical patients only becomes optimal at γ ≥ 0.7.
Limitations: 3 coarse states, no terminal/death state, hand-designed rewards, synthetic data, Markov assumption.

## Files
| File | Purpose |
|---|---|
| `mdp_engine.py` | States, actions, P, R, data generator, policy/value iteration, triage, graphs |
| `run_pipeline.py` | Runs everything and writes `outputs/` |
| `app.py` | Streamlit chatbot product (chat, graph, policy analysis, data tabs) |
| `chatbot_cli.py` | Terminal chatbot (no Streamlit needed) |
| `Healthcare_MDP_Chatbot.ipynb` | Colab notebook with all code + outputs + interpretation |
| `outputs/mdp_flowchart.png` | Reference-style MDP graph (`action (prob), R: reward`) |
| `outputs/mdp_policy_graph.png` | Colour-coded graph, optimal policy in bold |
| `outputs/patient_data.csv` | Synthetic patient dataset |

## Run
```bash
pip install -r requirements.txt      # also needs Graphviz binary: apt install graphviz / brew install graphviz
python run_pipeline.py               # generate data, solve MDP, save graphs
streamlit run app.py                 # launch chatbot UI
python chatbot_cli.py                # terminal chatbot
```
On Colab: upload the notebook and Run all.

> Educational prototype – not medical advice.

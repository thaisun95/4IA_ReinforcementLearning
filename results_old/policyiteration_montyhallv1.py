import gradio as gr
import random

# --- Variables globales de session ---
session_state = {
    "winning_door": None,
    "revealed_door": None,
    "chosen_door": None,
    "remaining_door": None,
    "result": None,
    "stay_wins": 0,
    "switch_wins": 0,
    "stay_trials": 0,
    "switch_trials": 0,
}

# --- Étape 1 : choix initial du joueur ---
def choisir_porte(porte):
    session_state["winning_door"] = random.choice([0, 1, 2])
    session_state["chosen_door"] = porte

    # Monty ouvre une mauvaise porte
    non_chosen = [d for d in [0, 1, 2] if d != porte]
    if session_state["winning_door"] == porte:
        session_state["revealed_door"] = random.choice(non_chosen)
    else:
        session_state["revealed_door"] = [d for d in non_chosen if d != session_state["winning_door"]][0]

    session_state["remaining_door"] = [d for d in [0, 1, 2]
                                       if d not in [porte, session_state["revealed_door"]]][0]

    message = f"Tu as choisi la porte {porte}. Monty ouvre la porte {session_state['revealed_door']} (vide)."
    message += f"\nSouhaites-tu rester sur la porte {porte} ou changer pour la porte {session_state['remaining_door']} ?"

    return message, gr.update(visible=True)

# --- Étape 2 : décision de rester ou changer ---
def decider(strategy):
    final_choice = session_state["chosen_door"] if strategy == "stay" else session_state["remaining_door"]
    gagne = final_choice == session_state["winning_door"]

    if strategy == "stay":
        session_state["stay_trials"] += 1
        if gagne:
            session_state["stay_wins"] += 1
    else:
        session_state["switch_trials"] += 1
        if gagne:
            session_state["switch_wins"] += 1

    session_state["result"] = "🎉 Gagné !" if gagne else "💀 Perdu..."
    stats = f"""
### Résultat : {session_state["result"]}

🏠 Porte gagnante : {session_state['winning_door']}  
🎯 Ta porte finale : {final_choice}

---

📊 Statistiques :
- Stay → {session_state['stay_wins']} / {session_state['stay_trials']} gagnés ({round(session_state['stay_wins'] / session_state['stay_trials'] * 100, 2) if session_state['stay_trials'] else 0}%)
- Switch → {session_state['switch_wins']} / {session_state['switch_trials']} gagnés ({round(session_state['switch_wins'] / session_state['switch_trials'] * 100, 2) if session_state['switch_trials'] else 0}%)
    """
    return stats, gr.update(visible=False)

# --- Interface Gradio ---
with gr.Blocks() as app:
    gr.Markdown("# 🎲 Jeu Monty Hall Interactif")

    with gr.Row():
        porte_input = gr.Radio([0, 1, 2], label="Choisis une porte", info="0, 1 ou 2")
        bouton_choix = gr.Button("Valider le choix")

    message_output = gr.Textbox(label="Message", lines=3)

    with gr.Row(visible=False) as decision_zone:
        gr.Markdown("Souhaites-tu rester ou changer ?")
        bouton_stay = gr.Button("Stay (je garde ma porte)")
        bouton_switch = gr.Button("Switch (je change de porte)")

    result_output = gr.Markdown()

    bouton_choix.click(fn=choisir_porte, inputs=porte_input, outputs=[message_output, decision_zone])
    bouton_stay.click(fn=lambda: decider("stay"), inputs=[], outputs=[result_output, decision_zone])
    bouton_switch.click(fn=lambda: decider("switch"), inputs=[], outputs=[result_output, decision_zone])

# --- Lancement ---
if __name__ == "__main__":
    app.launch()

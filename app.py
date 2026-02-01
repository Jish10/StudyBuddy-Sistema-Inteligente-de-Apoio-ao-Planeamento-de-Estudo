import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from studybuddy.io import load_json, load_model as load_joblib
from studybuddy.model import Predictor
from studybuddy.planner import PlanConfig, simulate_weeks, make_week_plan
from studybuddy.db import connect, upsert_subject, list_subjects, add_progress, get_progress

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="StudyBuddy — Boost + Bandit", layout="wide")
load_css()

st.title("✨ StudyBuddy — Boost monotónico + Bandit (UCB) — estilo CS188")
st.caption("Você altera horas/presença/nota anterior → a previsão muda de forma coerente. O Bandit distribui horas entre UCs.")

with st.sidebar:
    st.header("Treino (PowerShell)")
    st.code("python -m scripts.train_boost --csv data/StudentPerformanceFactors.csv --out artifacts --max_iter 2000 --max_depth 8 --learning_rate 0.05", language="bash")
    artifacts_dir = st.text_input("Pasta artifacts", value="artifacts")
    db_path = st.text_input("DB (sqlite)", value="studybuddy.db")
    st.markdown("---")
    st.subheader("Bandit")
    ucb_c = st.slider("Exploração (UCB c)", 0.0, 5.0, 1.8, 0.1)

pred_path = Path(artifacts_dir)/"predictor.joblib"
meta_path = Path(artifacts_dir)/"meta.json"
if not pred_path.exists() or not meta_path.exists():
    st.error("Treine primeiro: artifacts/predictor.joblib e artifacts/meta.json não encontrados.")
    st.stop()

bundle = load_joblib(pred_path)
meta = load_json(meta_path)
predictor = Predictor(model=bundle["model"], features=bundle["features"], rmse_val=float(bundle.get("rmse", 10.0)))

conn = connect(db_path)

tabs = st.tabs(["🧾 Disciplinas", "📈 Previsão", "🗺️ Plano semanal", "📊 Simulação", "🗃️ Histórico"])

with tabs[0]:
    st.subheader("Disciplinas (SQLite)")
    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1:
        name = st.text_input("Nome da UC", value="IA")
    with c2:
        last = st.number_input("Última nota", 0.0, 100.0, 60.0, 1.0)
    with c3:
        diff = st.slider("Dificuldade", 1, 5, 3)
    with c4:
        prio = st.slider("Prioridade", 1, 5, 3)

    if st.button("Guardar/Atualizar UC"):
        upsert_subject(conn, name.strip(), last, diff, prio)
        st.success("Guardado.")

    rows = list_subjects(conn)
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["UC","Last","Diff","Prio"]), use_container_width=True)
    else:
        st.info("Ainda não há UCs na base de dados.")

with tabs[1]:
    st.subheader("Previsão (curva horas → nota)")
    rows = list_subjects(conn)
    if not rows:
        st.info("Crie UCs na aba Disciplinas.")
        st.stop()

    subjects = [r[0] for r in rows]
    last_scores = {r[0]: float(r[1]) for r in rows}

    sel = st.selectbox("UC", subjects)
    attendance = st.slider("Presença (%)", 0, 100, 80)
    max_h = st.slider("Máx horas/semana", 1, 40, 20)
    h_try = st.slider("Horas/semana nessa UC", 0, max_h, min(6, max_h))

    # defaults for extra features (if trained with them)
    defaults = {}
    for f in predictor.features:
        if f in ("Hours_Studied","Attendance","Previous_Scores"):
            continue
        if f == "Sleep_Hours": defaults[f] = 7.0
        elif f == "Tutoring_Sessions": defaults[f] = 0.0
        elif f == "Physical_Activity": defaults[f] = 3.0
        else: defaults[f] = 0.0

    pred = predictor.predict(Hours_Studied=float(h_try), Attendance=float(attendance), Previous_Scores=float(last_scores[sel]), **defaults)
    st.metric("Nota prevista", f"{pred:.1f}")

    xs, ys_raw, ys = predictor.predict_curve("Hours_Studied", max_h, Attendance=float(attendance), Previous_Scores=float(last_scores[sel]), **defaults)
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Horas/semana")
    plt.ylabel("Nota prevista")
    st.pyplot(fig, clear_figure=True)

with tabs[2]:
    st.subheader("Plano semanal (Bandit UCB)")
    rows = list_subjects(conn)
    if not rows:
        st.info("Crie UCs na aba Disciplinas.")
        st.stop()
    subjects = [r[0] for r in rows]
    last_scores = {r[0]: float(r[1]) for r in rows}
    difficulty = {r[0]: int(r[2]) for r in rows}
    priority = {r[0]: int(r[3]) for r in rows}

    attendance = st.slider("Presença (%)", 0, 100, 80, key="att_plan")
    total_hours = st.slider("Horas totais disponíveis/semana", 0, 80, 18, key="tot_plan")
    max_hours = st.slider("Máx horas por UC", 1, 40, 14, key="max_plan")
    weeks = st.slider("Semanas para simular", 1, 20, 6, key="weeks_plan")

    cfg = PlanConfig(weeks=int(weeks), total_hours_week=int(total_hours), max_hours_per_subject=int(max_hours), ucb_c=float(ucb_c))
    alloc = make_week_plan(predictor, subjects, float(attendance), last_scores, difficulty, priority, cfg)
    st.dataframe(pd.DataFrame([alloc]), use_container_width=True)

    if st.button("Guardar no histórico (DB)"):
        ts = int(time.time())
        defaults = {}
        for f in predictor.features:
            if f in ("Hours_Studied","Attendance","Previous_Scores"):
                continue
            if f == "Sleep_Hours": defaults[f] = 7.0
            elif f == "Tutoring_Sessions": defaults[f] = 0.0
            elif f == "Physical_Activity": defaults[f] = 3.0
            else: defaults[f] = 0.0
        for s in subjects:
            pred = predictor.predict(Hours_Studied=float(alloc.get(s,0)), Attendance=float(attendance), Previous_Scores=float(last_scores[s]), **defaults)
            add_progress(conn, ts, s, float(alloc.get(s,0)), float(pred))
        st.success("Histórico guardado.")

with tabs[3]:
    st.subheader("Simulação multi-semanas (evolução de notas)")
    rows = list_subjects(conn)
    if not rows:
        st.info("Crie UCs na aba Disciplinas.")
        st.stop()
    subjects = [r[0] for r in rows]
    last_scores = {r[0]: float(r[1]) for r in rows}
    difficulty = {r[0]: int(r[2]) for r in rows}
    priority = {r[0]: int(r[3]) for r in rows}

    attendance = st.slider("Presença (%)", 0, 100, 80, key="att_sim")
    total_hours = st.slider("Horas totais/semana", 0, 80, 18, key="tot_sim")
    max_hours = st.slider("Máx horas por UC", 1, 40, 14, key="max_sim")
    weeks = st.slider("Semanas", 1, 30, 8, key="weeks_sim")

    cfg = PlanConfig(weeks=int(weeks), total_hours_week=int(total_hours), max_hours_per_subject=int(max_hours), ucb_c=float(ucb_c))
    plans, scores = simulate_weeks(predictor, subjects, float(attendance), last_scores, difficulty, priority, cfg)

    plan_rows = []
    for i, a in enumerate(plans, start=1):
        plan_rows.append({"Semana": i, **a, "Total": sum(a.values())})
    st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)

    fig = plt.figure()
    w = np.arange(1, len(plans)+1)
    bottom = np.zeros(len(plans))
    for s in subjects:
        vals = np.array([plans[i-1].get(s,0) for i in w], dtype=float)
        plt.bar(w, vals, bottom=bottom, label=s)
        bottom += vals
    plt.xlabel("Semana")
    plt.ylabel("Horas")
    plt.legend(fontsize=8, loc="upper right")
    st.pyplot(fig, clear_figure=True)

    fig2 = plt.figure()
    for s in subjects:
        plt.plot(range(0, len(scores[s])), scores[s], label=s)
    plt.xlabel("Semana (0=baseline)")
    plt.ylabel("Nota prevista")
    plt.legend(fontsize=8, loc="lower right")
    st.pyplot(fig2, clear_figure=True)

with tabs[4]:
    st.subheader("Histórico (SQLite)")
    rows = list_subjects(conn)
    subj = st.selectbox("Filtrar por UC", ["(todas)"] + [r[0] for r in rows])
    items = get_progress(conn, None if subj=="(todas)" else subj, limit=200)
    if not items:
        st.info("Sem histórico ainda.")
    else:
        dfh = pd.DataFrame(items, columns=["ts","UC","hours","predicted"])
        dfh["ts"] = pd.to_datetime(dfh["ts"], unit="s")
        st.dataframe(dfh, use_container_width=True)



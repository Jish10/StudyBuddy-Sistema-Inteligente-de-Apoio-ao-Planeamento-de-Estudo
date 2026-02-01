# StudyBuddy — Boost (monotónico) + Bandit (UCB)

## Dataset
Coloque o CSV em:
`data/StudentPerformanceFactors.csv`

## Instalar
```powershell
python -m pip install -r requirements.txt
```

## Treinar (PowerShell - UMA LINHA)
```powershell
python -m scripts.train_boost --csv data/StudentPerformanceFactors.csv --out artifacts --max_iter 2000 --max_depth 8 --learning_rate 0.05 --min_samples_leaf 40
```

### Mais pesado
```powershell
python -m scripts.train_boost --csv data/StudentPerformanceFactors.csv --out artifacts --max_iter 6000 --max_depth 10 --learning_rate 0.03 --min_samples_leaf 20

1. Introduction

StudyBuddy is an intelligent decision-support system designed to help students plan their study time more effectively across multiple courses (units).

Instead of relying on intuition, the system uses Artificial Intelligence techniques to:

predict academic performance,

simulate the impact of study hours,

and recommend how to allocate limited study time rationally.

This project was developed in the context of an Artificial Intelligence course, inspired by concepts taught in CS188 – Introduction to Artificial Intelligence.

2. Motivation

Students often face questions such as:

Which subject should I study more?

Will studying more hours actually improve my grade?

How should I divide my time if I have many courses?

Most existing tools provide static schedules or simple averages.
StudyBuddy addresses this gap by using data-driven learning and decision-making under uncertainty.

3. Project Objectives

The main objectives of this project are:

Use real educational data to learn how study behavior affects grades

Predict future grades based on user input

Allocate study hours intelligently when time is limited

Apply AI concepts in a realistic and explainable way

4. Dataset

The system is trained using the Student Performance Factors dataset, which includes:

Hours studied

Attendance

Previous exam scores

Academic and contextual factors

Final exam results

Using a real dataset ensures that predictions are grounded in realistic academic patterns.

5. System Overview

At a high level, the system works in three stages:

Prediction
A supervised learning model estimates the expected future grade of a student based on:

weekly study hours,

attendance,

previous grade.

Decision Making
A Multi-Armed Bandit algorithm (UCB) decides how to distribute available study hours across different courses.

Visualization
A graphical interface shows:

predicted grades,

the impact of changing study hours,

a recommended study plan.

6. Artificial Intelligence Techniques Used
6.1 Supervised Learning (Prediction Model)

A regression-based model is trained to predict future exam scores.

Why this model?

Interpretable results

Stable behavior

Clear relationship between inputs and output

Key properties:

Studying more hours never decreases the predicted grade

Previous performance strongly influences predictions

Attendance acts as a moderating factor

6.2 Multi-Armed Bandit (UCB Algorithm)

To allocate study hours, the project uses a Multi-Armed Bandit approach, specifically Upper Confidence Bound (UCB).

Each course is treated as an “arm”, and each study hour is a decision.

The algorithm balances:

Exploitation: focusing on subjects with high expected improvement

Exploration: occasionally investing time in less-studied subjects

This directly reflects CS188 concepts such as:

rational agents,

utility maximization,

decision-making under uncertainty.

7. Why Not MDPs or Bayesian Networks?

Although MDPs and Bayesian Networks are powerful AI tools, they were not chosen for this project because:

The dataset does not contain explicit temporal transitions required for MDPs

There is no strong causal structure needed for Bayesian Networks

The problem is incremental and resource-based, which fits Bandits better

The Bandit approach is:

simpler,

more appropriate,

easier to explain and justify academically.

8. User Interaction

The user can input:

list of courses (units),

perceived difficulty or priority,

last known grade,

weekly study hours,

attendance level.

The system dynamically updates:

predicted grades,

recommended study hours per course,

visual charts showing the effect of study time.

9. Project Architecture

The system is organized into logical layers:

Data Layer

Dataset loading

User inputs

Model Layer

Supervised prediction model

Decision Layer

Bandit-based allocation algorithm

Application Layer

Interactive interface

Visual feedback

This modular design improves readability, extensibility, and evaluation clarity.

10. Results

Predictions respond realistically to changes in study hours

Study plans adapt dynamically to user input

The system behaves rationally and consistently

Decisions are explainable, not black-box

11. Educational Value

This project demonstrates how AI concepts can be applied to real-world problems by combining:

learning from data,

rational decision-making,

user-centered design.

It emphasizes understanding and correctness rather than unnecessary algorithmic complexity.

12. Conclusion

StudyBuddy is an intelligent study planning system that:

learns from real data,

predicts academic outcomes,

and supports better decision-making.

The project successfully applies core AI principles from CS188 in a realistic, explainable, and academically sound way.

13. How to Run (Optional)
pip install -r requirements.txt
python -m scripts.train_model
streamlit run app.py

Author

João M.
Artificial Intelligence Project
```

## Rodar UI
```powershell
python -m streamlit run app.py
```

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
```

## Rodar UI
```powershell
python -m streamlit run app.py
```

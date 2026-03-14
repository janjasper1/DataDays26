# DataDays26 Pipeline

Dieses Projekt wird zentral ueber `backend/main.py` gesteuert.

## Setup

```zsh
cd /Users/janjasper/PycharmProjects/DataDays26
python -m pip install -r requirements.txt
```

Lege den Hugging-Face-Token in `backend/.env` ab:

```dotenv
HF_TOKEN=hf_xxx
```

## Pipeline ausfuehren

Nur Training:

```zsh
cd /Users/janjasper/PycharmProjects/DataDays26
python -m backend.main --steps 1000 --lr 0.005
```

Training + Visualisierung:

```zsh
cd /Users/janjasper/PycharmProjects/DataDays26
python -m backend.main --steps 1000 --lr 0.005 --visualize --hour 8
```

## Wichtige Parameter

- `--repo-id`: Dataset-ID auf Hugging Face (Default: `Amaan/DataDays`)
- `--steps`: Anzahl SVI-Trainingsschritte
- `--lr`: Learning Rate
- `--visualize`: Aktiviert Kartenvisualisierung
- `--hour`: Stunde fuer die Visualisierung


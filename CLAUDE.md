# Project: European Football Player Scout & Transfer Intelligence

## Goal
Portfolio project for Werkstudent/internship applications in Data Analytics (Germany).
Combines ML market value prediction + AI scouting reports + Power BI dashboard.
GitHub: github.com/MuhammadAli-99/bundesliga-player-scout

Author: Muhammad Ali, TU Ilmenau (Data Analytics). This is Project 2 — Project 1
("sales-intelligence-assistant") is complete and live on GitHub as reference for
style/structure (BigQuery + Claude API + Streamlit, bilingual EN/DE).

## Tech Stack (decided, do not change without discussion)
- Python, Pandas — data pipeline
- soccerdata library — pulls FBref stats
- XGBoost / Scikit-learn — market value regression model
- Claude API (claude-sonnet-4-6) — generates scouting reports (EN/DE)
- Power BI — main dashboard (deliberately NOT Streamlit, to differentiate from Project 1)
- mplsoccer — football-specific viz (radar charts)
- Git/GitHub

## Competitions
Bundesliga, Premier League, La Liga, Serie A, Ligue 1 (Champions League and Europa
League were originally planned but NOT YET implemented — domestic leagues only so far)

## Data Sources & Key Decisions

### FBref (via soccerdata)
- `app/data_pipeline.py` pulls player stats for season "2526" (2025/26)
- Available stat_types in this soccerdata version: 'standard', 'keeper', 'shooting',
  'playing_time', 'misc' — NOT 'passing'/'possession' (those don't exist, already
  hit this error once)
- Currently merging: standard + shooting + misc = 56 columns, 2839 players
- Output: `data/all_leagues_full_stats_2324.csv` (filename says 2324 but contains
  2526 season data after the season was updated — consider renaming for clarity)

### Market values — FIFA26, NOT Transfermarkt
- soccerdata has NO Transfermarkt module in this version (checked — only ClubElo,
  ESPN, FBref, MatchHistory, SoFIFA, Sofascore, Understat, WhoScored)
- SoFIFA's read_player_ratings() returns attributes but NO value_eur column — dead end
- SOLUTION: Kaggle dataset "FC 26 (FIFA 26) Player Data" by rovnez
  (kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data)
- Manually downloaded to `data/fifa_player_values.csv` — 3204 players across our 5
  leagues, columns include: short_name, long_name, club_name, league_name, value_eur,
  wage_eur, overall, potential, age

### Season alignment was critical
- Originally tried FBref season "2324" (2023/24) merged against FIFA26 (current 2026
  rosters) — only got 37-45% match rate because ~2 years of transfers had occurred
- FIXED by switching FBref to season "2526" — jumped match rate to 83.3%
- LESSON: always check temporal alignment between merged data sources

### Player matching logic — `app/merge_data.py` (3-stage, IN PROGRESS)
Player names/club names differ between FBref and FIFA (e.g. "Joshua Kimmich" vs
"Joshua Walter Kimmich"; "Bayern Munich" vs "FC Bayern München"). Uses
unidecode + rapidfuzz.

- Stage 1: team-restricted pool (fuzz.partial_ratio on club names, threshold 70),
  player name fuzz.token_set_ratio >= 80
- Stage 2: league-wide pool (no team restriction), token_set_ratio >= 96 (strict,
  avoids false positives)
- Stage 3 (NEWLY WRITTEN, NOT YET TESTED): for still-unmatched players, search
  ENTIRE FIFA dataset (all 43 leagues, not just our 5) — catches players who
  transferred outside our 5 competitions. Uses token_set_ratio >= 85 PLUS requires
  FIFA age within ±1 year of FBref age as independent corroborating signal
  (record-linkage technique — two weak signals combine into one confident match)

Current status before Stage 3: 83.3% match rate (2364/2839), spot-checks on
Kimmich/Kane/Neuer all scored 100.0 correctly. "Iago" and "Ritsu Doan" didn't match
in earlier tests — likely genuinely absent from FIFA26 or transferred recently.

## Project Structure

bundesliga-player-scout/

├── app/

│   ├── data_pipeline.py      # FBref scraping (DONE)

│   ├── merge_data.py          # FBref + FIFA merge, 3-stage matching (Stage 3 untested)

│   ├── feature_engineering.py # EMPTY — next phase

│   ├── model.py               # EMPTY — XGBoost market value model

│   └── llm_engine.py           # EMPTY — Claude scouting reports

├── data/                       # gitignored — CSVs not committed

│   ├── all_leagues_full_stats_2324.csv  (FBref, 2839 players, 56 cols)

│   ├── fifa_player_values.csv            (Kaggle FC26, manually placed)

│   └── merged_player_data.csv            (output of merge_data.py)

├── .env, .gitignore, requirements.txt

## Immediate Next Steps
1. Run `python -m app.merge_data` (Stage 3 version) — verify match rate improves
   beyond 83.3%, sanity-check the Stage 3 sample matches (age-verified) for false
   positives
2. Phase 2 — `feature_engineering.py`: per-90 metrics, performance index,
   age-value curve, contract urgency flag
3. Phase 3 — `model.py`: XGBoost regressor predicting value_eur, evaluate MAE/R²,
   feature importance chart
4. Phase 4 — `llm_engine.py`: Claude API generates 250-word scouting report
   (overvalued/undervalued/fair) in EN or DE, following the lang_instruction
   pattern from Project 1
5. Phase 5 — Power BI dashboard (player radar, value vs predicted, league
   comparison, contract status)
6. Phase 6 — README with architecture diagram, model metrics, Power BI
   screenshots

## Conventions / Lessons from Project 1
- NEVER commit .env or API keys — already hit GitHub secret scanning blocks 3x in
  Project 1, had to filter-branch history each time. .gitignore already configured.
- Windows/PowerShell: use `New-Item file1, file2 -ItemType File -Force` not
  `echo. >`, and `venv\Scripts\activate` (PowerShell needs
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` if blocked)
- Run scripts as `python -m app.modulename` from project root (not
  `python app/modulename.py`) to avoid ModuleNotFoundError on internal imports
- Bilingual EN/DE pattern from Project 1's llm_engine.py (lang_instruction
  parameter) should be reused for scouting reports
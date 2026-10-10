# DAG-EDA: AI-Powered Exploratory Data Analysis Prototype

A student-friendly web prototype inspired by the DAG-EDA research idea. It combines:
- Python/Flask backend
- HTML templates and CSS
- pandas-based exploratory analysis
- NetworkX-compatible graph data rendered with Cytoscape.js
- SQLite persistence for sessions and explored analysis paths
- Chart.js visualizations

## Important scope note

This is a prototype, not an exact reproduction of the research paper. The suggestion engine in `analysis_engine.py` is rule-based and explainable. It does not call an LLM or train a predictive machine-learning model. Column-to-concept links are based on keyword matching, so users should review them. Correlation is not causation.

## Requirements

- Python 3.10 or newer
- Internet connection when first opening the dashboard, because Chart.js and Cytoscape.js are loaded from public CDNs. For offline use, download those libraries and serve them from `static/`.

## Run locally

1. Unzip the project.
2. Open a terminal in the `dag_eda_project` folder.
3. Create and activate a virtual environment:

   Windows:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:
   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000` in your browser.
7. Upload a CSV file and enter a question.

## Included files

- `app.py` — routes, upload handling, SQLite setup, dashboard
- `analysis_engine.py` — summary stats, candidate analysis generation, graph creation
- `templates/` — HTML pages
- `static/style.css` — page styling
- `dag_eda.db` — created automatically on first run
- `uploads/` — uploaded CSVs are stored here

## Database tables

`sessions` stores the question, dataset metadata, graph structures, and analysis suggestions.

`analysis_history` stores analysis paths the user clicks to save.

## Safety and privacy

This app is intended for local educational use. Uploaded CSV files are stored in the `uploads/` directory. Do not upload confidential or personally identifiable data. Before public deployment, add authentication, CSRF protection, upload-size limits at the server layer, stronger secret management, and a data-retention policy. Do not expose Flask's debug server to the public internet.

## Suggested future improvements

1. Connect an LLM to generate richer analysis paths, with user review.
2. Replace keyword matching with editable concept mappings or embeddings.
3. Add backtracking and comparison of paths in the intent graph.
4. Add downloadable reports and more statistical tests.
5. Add automated tests and an evaluation study with users.

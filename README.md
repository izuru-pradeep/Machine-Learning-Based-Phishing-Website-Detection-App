# Phishing Website Detector — Streamlit App

Final artefact for the "Machine Learning-Based Phishing Website Detection"
project (module 6CS007). Runs two **independent** model suites — content-based
and URL-based — and shows two separate verdicts, e.g.:

> Web content seems legitimate, URL seems phishing.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `models/` folder next to `app.py` and place your two exported
   `.pkl` files in it:
   - `rf_model.pkl` — the content-based Random Forest model, from
     `Content_model_Train_Note_Book.ipynb`.
   - `url_random_forest_model.pkl` — the URL-based model, from
     `URL_Model_Train_Note_Book.ipynb`.

   **Important:** open `URL_Model_Train_Note_Book.ipynb`, check
   `results_df` to see which of the 7 classifiers actually had the highest
   mean accuracy, and update `URL_MODEL_FILENAME` near the top of `app.py`
   to match that model's saved filename if it isn't Random Forest. Do the
   same for `CONTENT_MODEL_FILENAME` if a different content-based classifier
   turns out to outperform Random Forest once you check its results table.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- The app makes one live request to the submitted URL and reuses it for
  both the content-based parsing and the URL-based page-behaviour features
  (no double-fetching).
- If the page can't be reached, the content-based check is skipped
  (shown as "Could not analyse") while the URL-based check still runs,
  using its own built-in fallbacks — this matches the graceful-failure
  behaviour described in the Artefact Design and Test Plan.
- WHOIS lookups require outbound network access; if `whois` isn't
  installed or a lookup fails, the domain-age/expiry features fall back
  to a conservative "suspicious" value rather than crashing.

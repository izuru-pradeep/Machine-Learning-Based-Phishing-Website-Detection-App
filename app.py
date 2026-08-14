"""
Phishing Website Detection - Streamlit Web Application
========================================================

This is the final artefact for the "Machine Learning-Based Phishing Website
Detection" final year project (module 6CS007).

Per the Artefact Design and Test Plan (Section 4.3), the content-based and
URL-based models are kept as two INDEPENDENT model suites. This app never
merges them into a single label - it runs both, and reports two separate
verdicts side by side, e.g.:

    "Web content seems legitimate, URL seems phishing."

- Content-based track: 43 features parsed from the fetched page's HTML/DOM.
- URL-based track: 40 features computed directly from the URL string alone
  (no WHOIS lookup, no live page fetch required) - so it always runs, even
  when the page itself cannot be reached.

Run locally with:
    streamlit run app.py

The two trained model files must be present in the models/ folder next to
this script (see CONFIGURATION below for the exact filenames expected).
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import math
import pickle
from collections import Counter
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# The content-based notebooks disabled SSL verification when fetching pages
# (some phishing hosts use invalid/self-signed certificates). We do the same
# here, so we suppress the resulting warning noise.
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================
# Folder holding the exported .pkl model files (see the two training
# notebooks). Both files must be placed here before running the app.
MODELS_DIR = "models"

# Which trained classifier to load for each independent track. These default
# to Random Forest, since that was the strongest performer on the
# content-based track (~96.7% mean cross-validated accuracy).
CONTENT_MODEL_FILENAME = "rf_model.pkl"
URL_MODEL_FILENAME = "url_random_forest_model.pkl"

# How long to wait for a page to respond before giving up (seconds).
REQUEST_TIMEOUT = 5

PAGE_TITLE = "Phishing Website Detector"
PAGE_ICON = "🛡️"


# =============================================================================
# CONTENT-BASED FEATURE EXTRACTION (43 features)
# ------------------------------------------------------------------------
# These functions are carried over unchanged from the content-based data
# collector notebooks, so the feature values computed here line up exactly
# with the columns the content-based model (rf_model.pkl) was trained on.
# Each function inspects one aspect of the parsed HTML (a BeautifulSoup
# object).
# =============================================================================

def has_title(soup):
    if soup.title is None:
        return 0
    return 1 if len(soup.title.text) > 0 else 0


def has_input(soup):
    return 1 if len(soup.find_all("input")) else 0


def has_button(soup):
    return 1 if len(soup.find_all("button")) > 0 else 0


def has_image(soup):
    return 0 if len(soup.find_all("image")) == 0 else 1


def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
    return 0


def has_link(soup):
    return 1 if len(soup.find_all("link")) > 0 else 0


def has_password(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("name") or input_tag.get("id")) == "password":
            return 1
    return 0


def has_email_input(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("id") or input_tag.get("name")) == "email":
            return 1
    return 0


def has_hidden_element(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden":
            return 1
    return 0


def has_audio(soup):
    return 1 if len(soup.find_all("audio")) > 0 else 0


def has_video(soup):
    return 1 if len(soup.find_all("video")) > 0 else 0


def number_of_inputs(soup):
    return len(soup.find_all("input"))


def number_of_buttons(soup):
    return len(soup.find_all("button"))


def number_of_images(soup):
    image_tags = len(soup.find_all("image"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") or meta.get("name") == "image":
            count += 1
    return image_tags + count


def number_of_option(soup):
    return len(soup.find_all("option"))


def number_of_list(soup):
    return len(soup.find_all("li"))


def number_of_TH(soup):
    return len(soup.find_all("th"))


def number_of_TR(soup):
    return len(soup.find_all("tr"))


def number_of_href(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count


def number_of_paragraph(soup):
    return len(soup.find_all("p"))


def number_of_script(soup):
    return len(soup.find_all("script"))


def length_of_title(soup):
    if soup.title is None:
        return 0
    return len(soup.title.text)


def has_h1(soup):
    return 1 if len(soup.find_all("h1")) > 0 else 0


def has_h2(soup):
    return 1 if len(soup.find_all("h2")) > 0 else 0


def has_h3(soup):
    return 1 if len(soup.find_all("h3")) > 0 else 0


def length_of_text(soup):
    return len(soup.get_text())


def number_of_clickable_button(soup):
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count


def number_of_a(soup):
    return len(soup.find_all("a"))


def number_of_img(soup):
    return len(soup.find_all("img"))


def number_of_div(soup):
    return len(soup.find_all("div"))


def number_of_figure(soup):
    return len(soup.find_all("figure"))


def has_footer(soup):
    return 1 if len(soup.find_all("footer")) > 0 else 0


def has_form(soup):
    return 1 if len(soup.find_all("form")) > 0 else 0


def has_text_area(soup):
    return 1 if len(soup.find_all("textarea")) > 0 else 0


def has_iframe_content(soup):
    # Content-based iframe check (works on the parsed DOM). Named
    # "_content" to avoid clashing with any URL-based iframe helper.
    return 1 if len(soup.find_all("iframe")) > 0 else 0


def has_text_input(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "text":
            return 1
    return 0


def number_of_meta(soup):
    return len(soup.find_all("meta"))


def has_nav(soup):
    return 1 if len(soup.find_all("nav")) > 0 else 0


def has_object(soup):
    return 1 if len(soup.find_all("object")) > 0 else 0


def has_picture(soup):
    return 1 if len(soup.find_all("picture")) > 0 else 0


def number_of_sources(soup):
    return len(soup.find_all("source"))


def number_of_span(soup):
    return len(soup.find_all("span"))


def number_of_table(soup):
    return len(soup.find_all("table"))


# Fixed column order for the content-based feature set - must match the
# 'columns' list used in the data collector notebooks exactly, since that is
# the order rf_model.pkl was trained on. Verified against
# rf_model.feature_names_in_.
CONTENT_FEATURE_COLUMNS = [
    'has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link',
    'has_password', 'has_email_input', 'has_hidden_element', 'has_audio', 'has_video',
    'number_of_inputs', 'number_of_buttons', 'number_of_images', 'number_of_option',
    'number_of_list', 'number_of_th', 'number_of_tr', 'number_of_href',
    'number_of_paragraph', 'number_of_script', 'length_of_title', 'has_h1', 'has_h2',
    'has_h3', 'length_of_text', 'number_of_clickable_button', 'number_of_a',
    'number_of_img', 'number_of_div', 'number_of_figure', 'has_footer', 'has_form',
    'has_text_area', 'has_iframe', 'has_text_input', 'number_of_meta', 'has_nav',
    'has_object', 'has_picture', 'number_of_sources', 'number_of_span', 'number_of_table',
]


def extract_content_features(soup):
    """Run all 43 content-based extractor functions in the fixed column
    order above and return a single-row DataFrame ready for the content
    model's .predict()."""
    values = [
        has_title(soup), has_input(soup), has_button(soup), has_image(soup),
        has_submit(soup), has_link(soup), has_password(soup), has_email_input(soup),
        has_hidden_element(soup), has_audio(soup), has_video(soup),
        number_of_inputs(soup), number_of_buttons(soup), number_of_images(soup),
        number_of_option(soup), number_of_list(soup), number_of_TH(soup),
        number_of_TR(soup), number_of_href(soup), number_of_paragraph(soup),
        number_of_script(soup), length_of_title(soup), has_h1(soup), has_h2(soup),
        has_h3(soup), length_of_text(soup), number_of_clickable_button(soup),
        number_of_a(soup), number_of_img(soup), number_of_div(soup),
        number_of_figure(soup), has_footer(soup), has_form(soup), has_text_area(soup),
        has_iframe_content(soup), has_text_input(soup), number_of_meta(soup),
        has_nav(soup), has_object(soup), has_picture(soup), number_of_sources(soup),
        number_of_span(soup), number_of_table(soup),
    ]
    return pd.DataFrame([values], columns=CONTENT_FEATURE_COLUMNS)


# =============================================================================
# URL-BASED FEATURE EXTRACTION (40 features)
# ------------------------------------------------------------------------
# Carried over unchanged from feature_extractor.py / the URL training
# notebook, so the feature values here line up exactly with what
# url_random_forest_model.pkl was trained on (verified against
# url_model.feature_names_in_). Every feature is computed directly from the
# URL string - no WHOIS lookup or live page fetch is needed, so this track
# always runs, even when the page itself is unreachable.
# =============================================================================

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "v.gd", "x.co", "po.st", "u.to", "j.mp",
    "s.id", "rb.gy", "shorturl.at", "clck.ru", "soo.gd",
]
SHORTENING_PATTERN = re.compile("|".join(re.escape(s) for s in SHORTENING_SERVICES))

SENSITIVE_WORDS = [
    "account", "confirm", "banking", "secure", "signin", "login", "verify",
    "update", "password", "username", "billing", "security", "payment",
    "customer", "service", "verification", "limited", "access", "urgent",
    "suspend", "unlock", "recover", "wallet", "support", "identity",
    "validate", "authentication", "alert", "important",
]

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _calculate_entropy(text):
    if not text:
        return 0.0
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


def url_length(url):
    return len(url)


def is_shortened_url(url):
    return 1 if SHORTENING_PATTERN.search(url) else 0


def has_prefix_suffix(url):
    return 1 if "-" in urlparse(url).netloc else 0


def count_dots(url):
    return url.count(".")


def has_sensitive_word(url):
    domain = urlparse(url).netloc.lower()
    return 1 if any(word in domain for word in SENSITIVE_WORDS) else 0


def has_unicode_domain(url):
    domain = urlparse(url).netloc
    if any(ord(ch) > 127 for ch in domain):
        return 1
    return 0


def get_domain_length(url):
    return len(urlparse(url).netloc)


def get_path_length(url):
    return len(urlparse(url).path)


def get_query_length(url):
    return len(urlparse(url).query)


def contains_login(url):
    return 1 if 'login' in url.lower() else 0


def contains_verify(url):
    return 1 if 'verify' in url.lower() else 0


def contains_account(url):
    return 1 if 'account' in url.lower() else 0


def contains_security(url):
    return 1 if 'security' in url.lower() else 0


def contains_password(url):
    return 1 if 'password' in url.lower() else 0


def contains_payment(url):
    return 1 if 'payment' in url.lower() else 0


def has_percent_encoding(url):
    return 1 if '%' in url else 0


def has_punycode(url):
    domain = urlparse(url).netloc
    return 1 if 'xn--' in domain else 0


def url_entropy(url):
    return _calculate_entropy(url)


def domain_entropy(url):
    domain = urlparse(url).netloc
    return _calculate_entropy(domain)


def digit_ratio(url):
    domain = urlparse(url).netloc
    if not domain:
        return 0.0
    return sum(c.isdigit() for c in domain) / len(domain)


def special_char_ratio(url):
    domain = urlparse(url).netloc
    if not domain:
        return 0.0
    return sum(1 for c in domain if not c.isalnum() and c != '.') / len(domain)


def domain_hyphen_count(url):
    return urlparse(url).netloc.count('-')


def brand_similarity(url, known_brands=None):
    # Placeholder - returns 0 (not similar). Matches the value the model
    # was trained against for this column; a full implementation would
    # compare the domain to a list of known brands via string similarity.
    return 0


def digit_count(url):
    return sum(c.isdigit() for c in url)


def letter_count(url):
    return sum(c.isalpha() for c in url)


def hyphen_count(url):
    return url.count('-')


def slash_count(url):
    return url.count('/')


def underscore_count(url):
    return url.count('_')


def question_count(url):
    return url.count('?')


def equal_count(url):
    return url.count('=')


def ampersand_count(url):
    return url.count('&')


def percent_count(url):
    return url.count('%')


def at_count(url):
    return url.count('@')


def has_https(url):
    return 1 if urlparse(url).scheme == 'https' else 0


def has_ip(url):
    return 1 if IP_PATTERN.search(url) else 0


def subdomain_count(url):
    netloc = urlparse(url).netloc
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", netloc) or netloc == 'localhost':
        return 0
    if ':' in netloc:
        netloc = netloc.split(':')[0]
    parts = netloc.split('.')
    return max(0, len(parts) - 2)


def path_depth(url):
    segments = urlparse(url).path.split('/')
    return sum(1 for s in segments if s)


def query_parameter_count(url):
    query = urlparse(url).query
    if not query:
        return 0
    params = query.split('&')
    return sum(1 for p in params if p)


def has_port(url):
    return 1 if urlparse(url).port is not None else 0


def has_fragment(url):
    return 1 if urlparse(url).fragment else 0


# Column order matches url_random_forest_model.pkl's feature_names_in_
# exactly - do not reorder.
URL_FEATURE_COLUMNS = [
    "URL_Length", "Path_Depth", "Has_IP", "Is_Shortened",
    "Has_Prefix_Suffix", "Dot_Count", "Has_Sensitive_Word", "Has_Unicode_Domain",
    "Domain_Length", "Path_Length", "Query_Length",
    "Digit_Count", "Letter_Count", "Hyphen_Count", "Slash_Count",
    "Underscore_Count", "Question_Count", "Equal_Count", "Ampersand_Count",
    "Percent_Count", "At_Count",
    "Has_HTTPS", "Subdomain_Count", "Query_Parameter_Count", "Has_Port", "Has_Fragment",
    "Contains_Login", "Contains_Verify", "Contains_Account",
    "Contains_Security", "Contains_Password", "Contains_Payment",
    "Has_Percent_Encoding", "Has_Punycode", "URL_Entropy", "Domain_Entropy",
    "Domain_Digit_Ratio", "Domain_Special_Char_Ratio", "Domain_Hyphen_Count", "Brand_Similarity",
]


def ensure_trailing_slash(url: str) -> str:
    """Add a trailing slash to the URL's path if it doesn't already end with
    one - matches how URLs were normalised before feature extraction during
    training."""
    if not url.endswith('/'):
        url += '/'
    return url


def extract_url_features(url):
    """Extract the 40-feature vector for a single URL, in
    URL_FEATURE_COLUMNS order, and return it as a single-row DataFrame ready
    for the URL model's .predict(). Pure string processing - no network
    call, so this always succeeds even when the page is unreachable."""
    url = ensure_trailing_slash(url)

    lexical = [
        url_length(url), path_depth(url), has_ip(url), is_shortened_url(url),
        has_prefix_suffix(url), count_dots(url), has_sensitive_word(url),
        has_unicode_domain(url), get_domain_length(url), get_path_length(url),
        get_query_length(url),
    ]
    characters = [
        digit_count(url), letter_count(url), hyphen_count(url), slash_count(url),
        underscore_count(url), question_count(url), equal_count(url),
        ampersand_count(url), percent_count(url), at_count(url),
    ]
    structure = [
        has_https(url), subdomain_count(url), query_parameter_count(url),
        has_port(url), has_fragment(url),
    ]
    suspicious_patterns = [
        contains_login(url), contains_verify(url), contains_account(url),
        contains_security(url), contains_password(url), contains_payment(url),
    ]
    obfuscation = [
        has_percent_encoding(url), has_punycode(url), url_entropy(url), domain_entropy(url),
    ]
    domain = [
        digit_ratio(url), special_char_ratio(url), domain_hyphen_count(url), brand_similarity(url),
    ]

    values = lexical + characters + structure + suspicious_patterns + obfuscation + domain
    return pd.DataFrame([values], columns=URL_FEATURE_COLUMNS)


# =============================================================================
# MODEL LOADING
# ------------------------------------------------------------------------
# st.cache_resource keeps the models loaded in memory across user
# interactions, rather than re-reading the .pkl files from disk on every
# button click.
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model(filename):
    """Load a single pickled scikit-learn model from the models/ folder.
    Returns None (rather than raising) if the file is missing, so the UI can
    show a friendly setup message instead of crashing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_prediction_details(model, features_df):
    """Run a model's prediction and, where possible, two different
    probability readings:
    - confidence: how sure the model is of the class it actually predicted
      (shown on that model's own verdict card, as before).
    - phishing_probability: the probability of the PHISHING class
      specifically, regardless of which class was predicted. This is the
      directional 0-1 score the combined risk score is built from - using
      `confidence` there would be wrong, since a model that's 90% sure a
      site is legitimate and one that's 90% sure it's phishing both report
      0.9 confidence despite pointing in opposite directions.
    Falls back gracefully for classifiers that don't expose predict_proba
    (e.g. LinearSVC), rather than assuming every model supports it."""
    prediction = model.predict(features_df)[0]
    confidence = None
    phishing_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]
        confidence = float(np.max(probabilities))
        classes = list(model.classes_)
        if 1 in classes:
            phishing_probability = float(probabilities[classes.index(1)])
    return prediction, confidence, phishing_probability


# =============================================================================
# COMBINED RISK SCORE
# ------------------------------------------------------------------------
# The two verdict cards stay independent (per the Artefact Design and Test
# Plan, Section 4.3) - but a raw side-by-side view can leave a non-technical
# user unsure what to actually do. This section folds both models'
# phishing-probabilities into one 0-100% risk score, weighted by each
# model's own measured accuracy, and maps that score to plain-English
# guidance.
# =============================================================================

# Weights = each model's own cross-validated accuracy from its training
# notebook, so the more reliable track counts for more in the blend.
CONTENT_MODEL_ACCURACY = 0.967  # content-based RF, ~96.7% mean CV accuracy (see CONFIGURATION note above)
URL_MODEL_ACCURACY = 0.967      # TODO: replace with the URL model's own measured CV accuracy from
                                 # URL_Model_Train_Note_Book.ipynb - currently a placeholder equal to
                                 # the content model's, since no figure for the URL track was supplied.


def compute_combined_risk(content_phishing_prob, url_phishing_prob):
    """Blend both models' phishing-probabilities into a single 0-100 risk
    percentage, weighted by each model's own accuracy.

    Returns (percentage, unverified):
    - percentage: the combined risk score.
    - unverified: True when the content check couldn't run (page
      unreachable), meaning the score reflects the URL track alone and
      should be shown with an explicit caveat.
    """
    if content_phishing_prob is None:
        return url_phishing_prob * 100, True

    total_weight = CONTENT_MODEL_ACCURACY + URL_MODEL_ACCURACY
    combined = (
        content_phishing_prob * CONTENT_MODEL_ACCURACY
        + url_phishing_prob * URL_MODEL_ACCURACY
    ) / total_weight
    return combined * 100, False


# Five risk bands, each with a short label and concrete prevention steps for
# the user to follow at that risk level.
RISK_BANDS = [
    {
        "label": "Very Low Risk",
        "css_class": "verdict-safe",
        "icon": "✅",
        "tips": [
            "Both checks agree this site looks legitimate.",
            "Normal browsing caution still applies — never share passwords over email or chat.",
        ],
    },
    {
        "label": "Low Risk",
        "css_class": "verdict-safe",
        "icon": "🙂",
        "tips": [
            "Only minor flags detected.",
            "Double-check the domain spelling before entering any sensitive info.",
        ],
    },
    {
        "label": "Uncertain",
        "css_class": "verdict-unknown",
        "icon": "⚠️",
        "tips": [
            "Signals are mixed or weak — the checks aren't confident either way.",
            "Don't log in or enter payment details on this page.",
            "Reach the site via a bookmark or official app instead of this link.",
        ],
    },
    {
        "label": "High Risk",
        "css_class": "verdict-danger",
        "icon": "⛔",
        "tips": [
            "Strong phishing indicators detected.",
            "Don't click any links or download anything from this page.",
            "Close the tab and verify through the organisation's official channel.",
        ],
    },
    {
        "label": "Very High Risk",
        "css_class": "verdict-danger",
        "icon": "🚨",
        "tips": [
            "Treat this as phishing.",
            "Don't enter any data — close the page immediately.",
            "Consider reporting the link (e.g. to Google Safe Browsing or your email provider).",
        ],
    },
]


def get_risk_band(percentage):
    """Map a 0-100 risk percentage to one of the five RISK_BANDS entries."""
    index = min(int(percentage // 20), len(RISK_BANDS) - 1)
    return RISK_BANDS[index]


# =============================================================================
# URL NORMALISATION + PAGE FETCHING
# =============================================================================

def normalize_url(raw_url: str) -> str:
    """Add a scheme if the user typed a bare domain, e.g. 'example.com' ->
    'http://example.com'. Both extractors rely on urlparse() splitting the
    URL into scheme/netloc/path correctly, which requires a scheme."""
    raw_url = raw_url.strip()
    if not raw_url:
        return raw_url
    if "://" not in raw_url:
        raw_url = "http://" + raw_url
    return raw_url


def fetch_page(url):
    """Attempt a single live fetch of the target URL. Returns the response
    object, or None if the page could not be reached. Only the content-based
    track needs this - the URL-based track never depends on it."""
    try:
        return requests.get(url, timeout=REQUEST_TIMEOUT, verify=False)
    except Exception:
        return None


# =============================================================================
# STREAMLIT UI
# =============================================================================

def inject_custom_css():
    st.markdown("""
        <style>
        .main > div { padding-top: 1.5rem; }

        .hero {
            background: linear-gradient(135deg, #16324F 0%, #2C5F8A 100%);
            padding: 2rem 2rem 1.6rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.8rem;
        }
        .hero h1 { margin: 0; font-size: 1.9rem; }
        .hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        .verdict-card {
            border-radius: 14px;
            padding: 1.3rem 1.4rem;
            height: 100%;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .verdict-card h3 { margin-top: 0; margin-bottom: 0.3rem; font-size: 1rem; color: #555; }
        .verdict-label { font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0; }
        .verdict-safe { background: #EAF7EC; border-left: 6px solid #3A7D44; }
        .verdict-safe .verdict-label { color: #3A7D44; }
        .verdict-danger { background: #FCEAEA; border-left: 6px solid #B33A3A; }
        .verdict-danger .verdict-label { color: #B33A3A; }
        .verdict-unknown { background: #FFF6E5; border-left: 6px solid #B8860B; }
        .verdict-unknown .verdict-label { color: #B8860B; }

        .summary-banner {
            padding: 1rem 1.3rem;
            border-radius: 12px;
            background: #F5F8FC;
            border: 1px solid #DDE7F0;
            font-size: 1.05rem;
            margin: 1.2rem 0;
        }
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


def render_verdict_card(column, title, description, prediction, confidence):
    with column:
        if prediction is None:
            css_class, icon, label = "verdict-unknown", "❓", "Could not analyse"
        elif prediction == 1:
            css_class, icon, label = "verdict-danger", "⚠️", "Phishing"
        else:
            css_class, icon, label = "verdict-safe", "✅", "Legitimate"

        confidence_html = ""
        if confidence is not None:
            confidence_html = f"<div style='color:#666;font-size:0.9rem;'>Confidence: {confidence:.0%}</div>"

        st.markdown(f"""
            <div class="verdict-card {css_class}">
                <h3>{title}</h3>
                <div class="verdict-label">{icon} {label}</div>
                {confidence_html}
                <div style="color:#666;font-size:0.85rem;margin-top:0.5rem;">{description}</div>
            </div>
        """, unsafe_allow_html=True)


def verdict_word(prediction):
    if prediction is None:
        return "unknown"
    return "phishing" if prediction == 1 else "legitimate"


def render_combined_risk(percentage, unverified):
    """Render the combined 0-100% risk score, its band label, and the
    prevention tips for that band. Shown below the two independent verdict
    cards, not in place of them."""
    band = get_risk_band(percentage)

    unverified_html = ""
    if unverified:
        unverified_html = (
            "<div style='color:#B8860B;font-size:0.85rem;margin-top:0.5rem;'>"
            "⚠️ Content check unavailable (page unreachable) — this score reflects "
            "the URL-based check only. Treat it with extra caution and verify the "
            "site independently before relying on it."
            "</div>"
        )

    tips_html = "".join(f"<li>{tip}</li>" for tip in band["tips"])

    st.markdown(f"""
        <div class="verdict-card {band['css_class']}" style="margin-top:1rem;">
            <h3>Overall Risk</h3>
            <div class="verdict-label">{band['icon']} {percentage:.0f}% — {band['label']}</div>
            {unverified_html}
            <ul style="color:#555;font-size:0.9rem;margin-top:0.6rem;">{tips_html}</ul>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
    inject_custom_css()

    # ---- Header ----
    st.markdown(f"""
        <div class="hero">
            <h1>{PAGE_ICON} Phishing Website Detector</h1>
            <p>Two independent machine-learning checks - one on the page's content,
            one on the URL itself - shown side by side rather than merged into a
            single answer.</p>
        </div>
    """, unsafe_allow_html=True)

    # ---- Sidebar: project info ----
    with st.sidebar:
        st.subheader("About this tool")
        st.write(
            "This is a final-year project artefact. It runs two independently "
            "trained classifier suites:"
        )
        st.markdown("- **Content-based** — 43 features from the page's HTML/DOM")
        st.markdown("- **URL-based** — 40 features from the URL's structure alone "
                     "(no live fetch needed, so it always runs)")
        st.write(
            "The two verdicts are shown separately on purpose - a disagreement "
            "between them is useful information, not an error. An overall risk "
            "score below combines both into one plain-English recommendation."
        )
        st.caption("For academic demonstration only - not a substitute for "
                   "professional security judgement.")

    # ---- Load models once ----
    content_model = load_model(CONTENT_MODEL_FILENAME)
    url_model = load_model(URL_MODEL_FILENAME)

    if content_model is None or url_model is None:
        st.warning(
            f"Model file(s) not found in `{MODELS_DIR}/`. Expected "
            f"`{CONTENT_MODEL_FILENAME}` and `{URL_MODEL_FILENAME}`. "
            "Export these from the training notebooks and place them in the "
            "models/ folder next to app.py."
        )

    # ---- URL input ----
    raw_url = st.text_input("Enter a URL to check", placeholder="https://example.com/login")
    analyse_clicked = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if analyse_clicked and raw_url:
        if content_model is None or url_model is None:
            st.stop()

        url = normalize_url(raw_url)

        with st.spinner("Fetching the page and running both checks..."):
            # Only the content-based track needs a live fetch.
            response = fetch_page(url)

            # --- Content-based track ---
            content_prediction, content_confidence, content_phishing_prob = None, None, None
            if response is not None:
                soup = BeautifulSoup(response.content, "html.parser")
                content_features = extract_content_features(soup)
                content_prediction, content_confidence, content_phishing_prob = get_prediction_details(
                    content_model, content_features
                )

            # --- URL-based track (always runs - pure string processing,
            # no dependency on the page being reachable) ---
            url_features = extract_url_features(url)
            url_prediction, url_confidence, url_phishing_prob = get_prediction_details(
                url_model, url_features
            )

            # --- Combined risk score (weighted by each model's accuracy) ---
            risk_percentage, risk_unverified = compute_combined_risk(
                content_phishing_prob, url_phishing_prob
            )

        if response is None:
            st.info(
                "The page could not be reached, so the content-based check "
                "was skipped. The URL-based check still ran, since it only "
                "needs the URL string itself."
            )

        # ---- Two independent verdict cards ----
        col1, col2 = st.columns(2)
        render_verdict_card(
            col1, "Web Content", "Based on the page's HTML/DOM structure.",
            content_prediction, content_confidence,
        )
        render_verdict_card(
            col2, "URL", "Based on the URL's structure alone.",
            url_prediction, url_confidence,
        )

        # ---- Combined plain-English summary ----
        summary = f"Web content seems {verdict_word(content_prediction)}, URL seems {verdict_word(url_prediction)}."
        st.markdown(f'<div class="summary-banner">{summary}</div>', unsafe_allow_html=True)

        if content_prediction is not None and url_prediction is not None and content_prediction != url_prediction:
            st.caption(
                "The two checks disagree here - that's a signal worth treating "
                "with extra caution, not a bug in the tool."
            )

        # ---- Overall risk score + prevention guidance ----
        render_combined_risk(risk_percentage, risk_unverified)


if __name__ == "__main__":
    main()

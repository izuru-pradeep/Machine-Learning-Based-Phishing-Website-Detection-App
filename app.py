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
import time
import pickle
from datetime import datetime
from urllib.parse import urlparse

import requests
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# WHOIS lookups (domain age / expiry) - only needed for the URL-based track.
# Wrapped in a try/except import so the app still loads (with a warning)
# even if the 'whois' package isn't installed in a given environment.
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

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
# content-based track (~96.7% mean cross-validated accuracy). UPDATE
# URL_MODEL_FILENAME after running URL_Model_Train_Note_Book.ipynb and
# checking which classifier actually had the highest mean accuracy there -
# the filenames follow the "url_<classifier_name>_model.pkl" convention set
# in that notebook (e.g. url_random_forest_model.pkl, url_svm_model.pkl, ...).
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
# with the columns the content-based models were trained on. Each function
# inspects one aspect of the parsed HTML (a BeautifulSoup object).
# =============================================================================

def has_title(soup):
    # 1 if the page has a non-empty <title> tag, else 0.
    if soup.title is None:
        return 0
    return 1 if len(soup.title.text) > 0 else 0


def has_input(soup):
    # 1 if the page has at least one <input> element (e.g. a login field).
    return 1 if len(soup.find_all("input")) else 0


def has_button(soup):
    # 1 if the page has at least one <button> element.
    return 1 if len(soup.find_all("button")) > 0 else 0


def has_image(soup):
    # 1 if the page has at least one legacy <image> tag.
    return 0 if len(soup.find_all("image")) == 0 else 1


def has_submit(soup):
    # 1 if any <input> is specifically a submit button.
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
    return 0


def has_link(soup):
    # 1 if the page declares at least one <link> element (e.g. stylesheet).
    return 1 if len(soup.find_all("link")) > 0 else 0


def has_password(soup):
    # 1 if any <input> looks like a password field, by type/name/id.
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("name") or input_tag.get("id")) == "password":
            return 1
    return 0


def has_email_input(soup):
    # 1 if any <input> looks like an email field, by type/id/name.
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") or input_tag.get("id") or input_tag.get("name")) == "email":
            return 1
    return 0


def has_hidden_element(soup):
    # 1 if any <input> is a hidden field (often used to pass hidden data).
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden":
            return 1
    return 0


def has_audio(soup):
    # 1 if the page embeds an <audio> element.
    return 1 if len(soup.find_all("audio")) > 0 else 0


def has_video(soup):
    # 1 if the page embeds a <video> element.
    return 1 if len(soup.find_all("video")) > 0 else 0


def number_of_inputs(soup):
    # Total count of <input> elements.
    return len(soup.find_all("input"))


def number_of_buttons(soup):
    # Total count of <button> elements.
    return len(soup.find_all("button"))


def number_of_images(soup):
    # Total count of legacy <image> tags, plus <meta> tags that describe an image.
    image_tags = len(soup.find_all("image"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") or meta.get("name") == "image":
            count += 1
    return image_tags + count


def number_of_option(soup):
    # Total count of <option> elements (dropdown choices).
    return len(soup.find_all("option"))


def number_of_list(soup):
    # Total count of <li> list items.
    return len(soup.find_all("li"))


def number_of_TH(soup):
    # Total count of <th> table header cells.
    return len(soup.find_all("th"))


def number_of_TR(soup):
    # Total count of <tr> table rows.
    return len(soup.find_all("tr"))


def number_of_href(soup):
    # Count of <link> elements that declare an href attribute.
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count


def number_of_paragraph(soup):
    # Total count of <p> paragraph elements.
    return len(soup.find_all("p"))


def number_of_script(soup):
    # Total count of <script> elements.
    return len(soup.find_all("script"))


def length_of_title(soup):
    # Character length of the page's <title> text.
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
    # Character length of all visible text on the page.
    return len(soup.get_text())


def number_of_clickable_button(soup):
    # Count of <button> elements explicitly typed as "button" (clickable, not submit/reset).
    count = 0
    for button in soup.find_all("button"):
        if button.get("type") == "button":
            count += 1
    return count


def number_of_a(soup):
    # Total count of <a> anchor (hyperlink) elements.
    return len(soup.find_all("a"))


def number_of_img(soup):
    # Total count of <img> elements.
    return len(soup.find_all("img"))


def number_of_div(soup):
    # Total count of <div> elements.
    return len(soup.find_all("div"))


def number_of_figure(soup):
    # Total count of <figure> elements.
    return len(soup.find_all("figure"))


def has_footer(soup):
    return 1 if len(soup.find_all("footer")) > 0 else 0


def has_form(soup):
    return 1 if len(soup.find_all("form")) > 0 else 0


def has_text_area(soup):
    return 1 if len(soup.find_all("textarea")) > 0 else 0


def has_iframe_content(soup):
    # Content-based iframe check (works on the parsed DOM). Named
    # "_content" here to avoid clashing with the URL-based page-behaviour
    # has_iframe() further down, which works on the raw response text.
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
# the order the models were trained on.
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
# URL-BASED FEATURE EXTRACTION (14 features)
# ------------------------------------------------------------------------
# Carried over unchanged from URL_Feature_Extractor_Note_Book.ipynb, so the
# feature values here line up exactly with what the URL-based models were
# trained on.
# =============================================================================

# Known URL-shortening services. A match here is a mild phishing signal,
# since shorteners hide the real destination domain from the person clicking.
SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "tiny.cc",
    "lnkd.in", "db.tt", "qr.ae", "v.gd", "x.co", "po.st", "u.to", "j.mp",
    "s.id", "rb.gy", "shorturl.at", "clck.ru", "soo.gd",
]
SHORTENING_PATTERN = re.compile("|".join(re.escape(s) for s in SHORTENING_SERVICES))

# Words that commonly show up in phishing domains/paths as social-engineering
# bait (e.g. "verify-account-support.com").
SENSITIVE_WORDS = [
    "account", "confirm", "banking", "secure", "signin", "login", "verify",
    "update", "password", "username", "billing", "security", "payment",
    "customer", "service", "verification", "limited", "access", "urgent",
    "suspend", "unlock", "recover", "wallet", "support", "identity",
    "validate", "authentication", "alert", "important",
]

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def url_length(url):
    # Feature 1: raw character length of the URL.
    return len(url)


def url_depth(url):
    # Feature 2: number of non-empty '/' separated path segments.
    segments = urlparse(url).path.split('/')
    return sum(1 for s in segments if len(s) > 0)


def has_ip_address(url):
    # Feature 3: does the URL contain a raw IPv4 address instead of a domain?
    return 1 if IP_PATTERN.search(url) else 0


def has_at_symbol(url):
    # Feature 4: presence of '@' in the URL (used to disguise the real host).
    return 1 if "@" in url else 0


def is_shortened_url(url):
    # Feature 5: does the URL match a known link-shortening service?
    return 1 if SHORTENING_PATTERN.search(url) else 0


def has_prefix_suffix(url):
    # Feature 6: hyphen in the domain (e.g. "paypal-secure-login.com").
    return 1 if "-" in urlparse(url).netloc else 0


def count_dots(url):
    # Feature 7: total number of '.' characters in the URL.
    return url.count(".")


def has_sensitive_word(url):
    # Feature 8: does the domain contain a common phishing "bait" word?
    domain = urlparse(url).netloc.lower()
    return 1 if any(word in domain for word in SENSITIVE_WORDS) else 0


def has_unicode_domain(url):
    # Feature 9: punycode-encoded ("xn--") or raw non-ASCII domain -
    # both used in homograph (look-alike) domain attacks.
    domain = urlparse(url).netloc
    if "xn--" in domain:
        return 1
    if any(ord(ch) > 127 for ch in domain):
        return 1
    return 0


def _first_date(value):
    # Helper: WHOIS date fields can be None, a single datetime, a
    # date-formatted string, or a list of candidate dates - normalise to one
    # datetime (or None).
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None
    return value


def domain_age_flag(domain_info):
    # Feature 10: flags a domain whose creation-to-expiry window is under 6
    # months. Falls back to 1 (suspicious) if WHOIS data is unavailable.
    if domain_info is None:
        return 1
    created = _first_date(getattr(domain_info, "creation_date", None))
    expires = _first_date(getattr(domain_info, "expiration_date", None))
    if created is None or expires is None:
        return 1
    age_days = abs((expires - created).days)
    return 1 if (age_days / 30) < 6 else 0


def domain_expiry_flag(domain_info):
    # Feature 11: flags a domain expiring in under 6 months from today.
    if domain_info is None:
        return 1
    expires = _first_date(getattr(domain_info, "expiration_date", None))
    if expires is None:
        return 1
    days_left = (expires - datetime.now()).days
    return 1 if (days_left / 30) < 6 else 0


def safe_whois_lookup(url):
    # Wraps whois.whois() in a try/except: parked, suspended, or
    # privacy-protected domains commonly raise here rather than returning
    # usable data.
    if not WHOIS_AVAILABLE:
        return None
    try:
        return whois.whois(urlparse(url).netloc)
    except Exception:
        return None


def has_iframe(response):
    # Feature 12: does the fetched page contain an <iframe>?
    if response is None:
        return 1
    return 1 if re.search(r"<iframe", response.text, re.IGNORECASE) else 0


def has_mouseover(response):
    # Feature 13: does the page use an onmouseover script (status-bar spoofing)?
    if response is None:
        return 1
    return 1 if re.search(r"onmouseover", response.text, re.IGNORECASE) else 0


def excessive_redirects(response):
    # Feature 14: did the request go through more than 2 redirect hops?
    if response is None:
        return 1
    return 1 if len(response.history) > 2 else 0


URL_FEATURE_COLUMNS = [
    "URL_Length", "URL_Depth", "Has_IP", "Has_At_Symbol", "Is_Shortened",
    "Has_Prefix_Suffix", "Dot_Count", "Has_Sensitive_Word", "Has_Unicode_Domain",
    "Domain_Age_Flag", "Domain_Expiry_Flag",
    "Has_Iframe", "Has_Mouseover", "Excessive_Redirects",
]


def extract_url_features(url, response):
    """Run all 14 URL-based extractor functions in the fixed column order
    above and return a single-row DataFrame ready for the URL model's
    .predict(). Reuses the already-fetched `response` (may be None) for the
    page-behaviour features, rather than fetching the page a second time."""
    lexical = [
        url_length(url), url_depth(url), has_ip_address(url), has_at_symbol(url),
        is_shortened_url(url), has_prefix_suffix(url), count_dots(url),
        has_sensitive_word(url), has_unicode_domain(url),
    ]
    domain_info = safe_whois_lookup(url)
    domain_based = [domain_age_flag(domain_info), domain_expiry_flag(domain_info)]
    behaviour = [has_iframe(response), has_mouseover(response), excessive_redirects(response)]
    return pd.DataFrame([lexical + domain_based + behaviour], columns=URL_FEATURE_COLUMNS)


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


def get_verdict_and_confidence(model, features_df):
    """Run a model's prediction and, where possible, a confidence score.
    Falls back gracefully for classifiers that don't expose predict_proba
    (e.g. LinearSVC), rather than assuming every model supports it."""
    prediction = model.predict(features_df)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]
        confidence = float(np.max(probabilities))
    return prediction, confidence


# =============================================================================
# PAGE FETCHING
# =============================================================================

def fetch_page(url):
    """Attempt a single live fetch of the target URL. Returns the response
    object, or None if the page could not be reached. This one response is
    reused for BOTH the content-based parsing and the URL-based
    page-behaviour features, rather than fetching twice."""
    try:
        return requests.get(url, timeout=REQUEST_TIMEOUT, verify=False)
    except Exception:
        return None


# =============================================================================
# STREAMLIT UI
# =============================================================================

def inject_custom_css():
    # Custom styling for a cleaner look than Streamlit's defaults: a
    # gradient header banner and colour-coded verdict cards.
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
    """Render one of the two independent verdict cards. prediction is None
    when that track could not be evaluated (e.g. page unreachable)."""
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
    # Turns a 0/1/None prediction into the word used in the summary sentence.
    if prediction is None:
        return "unknown"
    return "phishing" if prediction == 1 else "legitimate"


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
        st.markdown("- **URL-based** — 14 features from the URL's structure, "
                     "domain registration, and page behaviour")
        st.write(
            "The two verdicts are shown separately on purpose - a disagreement "
            "between them is useful information, not an error."
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
    url = st.text_input("Enter a URL to check", placeholder="https://example.com/login")
    analyse_clicked = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if analyse_clicked and url:
        if content_model is None or url_model is None:
            st.stop()

        with st.spinner("Fetching the page and running both checks..."):
            # Single live fetch, reused for both the content parser and the
            # URL-based page-behaviour features (avoids fetching twice).
            response = fetch_page(url)

            # --- Content-based track ---
            content_prediction, content_confidence = None, None
            if response is not None:
                soup = BeautifulSoup(response.content, "html.parser")
                content_features = extract_content_features(soup)
                content_prediction, content_confidence = get_verdict_and_confidence(
                    content_model, content_features
                )

            # --- URL-based track (always runs - it has its own per-feature
            # fallbacks even when the page itself is unreachable) ---
            url_features = extract_url_features(url, response)
            url_prediction, url_confidence = get_verdict_and_confidence(
                url_model, url_features
            )

        if response is None:
            st.info(
                "The page could not be reached, so the content-based check "
                "was skipped. The URL-based check still ran, using its "
                "built-in fallbacks for the page-behaviour features."
            )

        # ---- Two independent verdict cards ----
        col1, col2 = st.columns(2)
        render_verdict_card(
            col1, "Web Content", "Based on the page's HTML/DOM structure.",
            content_prediction, content_confidence,
        )
        render_verdict_card(
            col2, "URL", "Based on the URL's structure, domain, and behaviour.",
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


if __name__ == "__main__":
    main()

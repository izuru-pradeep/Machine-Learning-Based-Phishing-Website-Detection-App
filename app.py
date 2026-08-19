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
- URL-based track: 38 features computed directly from the URL string alone
  (no WHOIS lookup, no live page fetch required) - so it always runs, even
  when the page itself cannot be reached.
- Each track's verdict includes an expandable "Why this result?" panel
  showing the top 8 features that most influenced that specific prediction,
  via SHAP (SHapley Additive exPlanations) - not a fixed heuristic, but the
  model's own actual reasoning for this particular input.

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
import shap
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
# URL-BASED FEATURE EXTRACTION (38 features)
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
#
# NOTE: Has_Unicode_Domain and Brand_Similarity were removed from this list
# (2026-08-19) because the currently deployed url_random_forest_model.pkl
# was never retrained after those two columns were added to the feature
# pipeline - it raises "Feature names unseen at fit time" for both. If you
# retrain the model on the extended feature set, add them back here (and to
# extract_url_features() and URL_FEATURE_DISPLAY below) in their original
# positions: Has_Unicode_Domain right after Has_Sensitive_Word, and
# Brand_Similarity at the end.
URL_FEATURE_COLUMNS = [
    "URL_Length", "Path_Depth", "Has_IP", "Is_Shortened",
    "Has_Prefix_Suffix", "Dot_Count", "Has_Sensitive_Word",
    "Domain_Length", "Path_Length", "Query_Length",
    "Digit_Count", "Letter_Count", "Hyphen_Count", "Slash_Count",
    "Underscore_Count", "Question_Count", "Equal_Count", "Ampersand_Count",
    "Percent_Count", "At_Count",
    "Has_HTTPS", "Subdomain_Count", "Query_Parameter_Count", "Has_Port", "Has_Fragment",
    "Contains_Login", "Contains_Verify", "Contains_Account",
    "Contains_Security", "Contains_Password", "Contains_Payment",
    "Has_Percent_Encoding", "Has_Punycode", "URL_Entropy", "Domain_Entropy",
    "Domain_Digit_Ratio", "Domain_Special_Char_Ratio", "Domain_Hyphen_Count",
]


def ensure_trailing_slash(url: str) -> str:
    """Add a trailing slash to the URL's path if it doesn't already end with
    one - matches how URLs were normalised before feature extraction during
    training."""
    if not url.endswith('/'):
        url += '/'
    return url


def extract_url_features(url):
    """Extract the 38-feature vector for a single URL, in
    URL_FEATURE_COLUMNS order, and return it as a single-row DataFrame ready
    for the URL model's .predict(). Pure string processing - no network
    call, so this always succeeds even when the page is unreachable."""
    url = ensure_trailing_slash(url)

    lexical = [
        url_length(url), path_depth(url), has_ip(url), is_shortened_url(url),
        has_prefix_suffix(url), count_dots(url), has_sensitive_word(url),
        get_domain_length(url), get_path_length(url),
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
        digit_ratio(url), special_char_ratio(url), domain_hyphen_count(url),
    ]

    values = lexical + characters + structure + suspicious_patterns + obfuscation + domain
    return pd.DataFrame([values], columns=URL_FEATURE_COLUMNS)


# =============================================================================
# EXPLAINABILITY (SHAP)
# ------------------------------------------------------------------------
# For a single prediction, shap.TreeExplainer gives an exact, mathematically
# faithful contribution for every feature toward the phishing class - the
# base rate plus the sum of all contributions equals the model's own
# predicted probability. This is the model's actual reasoning for THIS
# input, not a fixed heuristic, which is why the wording below is built
# from the sign of each contribution rather than a hard-coded assumption
# about what "should" be suspicious.
#
# Each dict below maps a technical feature column name to (plain-English
# label, value-formatting function), used only to phrase the value shown
# next to each contribution - it does not affect the model or the score.
# =============================================================================

URL_FEATURE_DISPLAY = {
    "URL_Length": ("URL Length", lambda v: f"{int(v)} characters"),
    "Path_Depth": ("Path Depth", lambda v: f"{int(v)} folder levels deep"),
    "Has_IP": ("IP Address Instead of Domain", lambda v: "Yes" if v else "No"),
    "Is_Shortened": ("Known URL Shortener", lambda v: "Yes" if v else "No"),
    "Has_Prefix_Suffix": ("Hyphen in Domain", lambda v: "Yes" if v else "No"),
    "Dot_Count": ("Number of Dots", lambda v: f"{int(v)}"),
    "Has_Sensitive_Word": ("Sensitive Keyword in Domain", lambda v: "Yes" if v else "No"),
    "Domain_Length": ("Domain Length", lambda v: f"{int(v)} characters"),
    "Path_Length": ("Path Length", lambda v: f"{int(v)} characters"),
    "Query_Length": ("Query String Length", lambda v: f"{int(v)} characters"),
    "Digit_Count": ("Digits in URL", lambda v: f"{int(v)}"),
    "Letter_Count": ("Letters in URL", lambda v: f"{int(v)}"),
    "Hyphen_Count": ("Hyphens in URL", lambda v: f"{int(v)}"),
    "Slash_Count": ("Slashes in URL", lambda v: f"{int(v)}"),
    "Underscore_Count": ("Underscores in URL", lambda v: f"{int(v)}"),
    "Question_Count": ("Question Marks in URL", lambda v: f"{int(v)}"),
    "Equal_Count": ("Equals Signs in URL", lambda v: f"{int(v)}"),
    "Ampersand_Count": ("Ampersands in URL", lambda v: f"{int(v)}"),
    "Percent_Count": ("Percent Signs in URL", lambda v: f"{int(v)}"),
    "At_Count": ("'@' Symbols in URL", lambda v: f"{int(v)}"),
    "Has_HTTPS": ("HTTPS Encryption", lambda v: "Used" if v else "Not used"),
    "Subdomain_Count": ("Number of Subdomains", lambda v: f"{int(v)}"),
    "Query_Parameter_Count": ("Query Parameters", lambda v: f"{int(v)}"),
    "Has_Port": ("Non-Standard Port", lambda v: "Yes" if v else "No"),
    "Has_Fragment": ("URL Fragment (#)", lambda v: "Present" if v else "Absent"),
    "Contains_Login": ("Contains 'login'", lambda v: "Yes" if v else "No"),
    "Contains_Verify": ("Contains 'verify'", lambda v: "Yes" if v else "No"),
    "Contains_Account": ("Contains 'account'", lambda v: "Yes" if v else "No"),
    "Contains_Security": ("Contains 'security'", lambda v: "Yes" if v else "No"),
    "Contains_Password": ("Contains 'password'", lambda v: "Yes" if v else "No"),
    "Contains_Payment": ("Contains 'payment'", lambda v: "Yes" if v else "No"),
    "Has_Percent_Encoding": ("Percent-Encoded Characters", lambda v: "Yes" if v else "No"),
    "Has_Punycode": ("Punycode Domain (xn--)", lambda v: "Yes" if v else "No"),
    "URL_Entropy": ("URL Randomness (Entropy)", lambda v: f"{v:.2f}"),
    "Domain_Entropy": ("Domain Randomness (Entropy)", lambda v: f"{v:.2f}"),
    "Domain_Digit_Ratio": ("Proportion of Digits in Domain", lambda v: f"{v:.0%}"),
    "Domain_Special_Char_Ratio": ("Proportion of Special Characters in Domain", lambda v: f"{v:.0%}"),
    "Domain_Hyphen_Count": ("Hyphens in Domain", lambda v: f"{int(v)}"),
}

CONTENT_FEATURE_DISPLAY = {
    "has_title": ("Has a Page Title", lambda v: "Yes" if v else "No"),
    "has_input": ("Has Input Fields", lambda v: "Yes" if v else "No"),
    "has_button": ("Has Buttons", lambda v: "Yes" if v else "No"),
    "has_image": ("Has Legacy <image> Tag", lambda v: "Yes" if v else "No"),
    "has_submit": ("Has a Submit Button", lambda v: "Yes" if v else "No"),
    "has_link": ("Has <link> Elements (e.g. stylesheets)", lambda v: "Yes" if v else "No"),
    "has_password": ("Has a Password Field", lambda v: "Yes" if v else "No"),
    "has_email_input": ("Has an Email Input Field", lambda v: "Yes" if v else "No"),
    "has_hidden_element": ("Has Hidden Form Fields", lambda v: "Yes" if v else "No"),
    "has_audio": ("Has Audio Content", lambda v: "Yes" if v else "No"),
    "has_video": ("Has Video Content", lambda v: "Yes" if v else "No"),
    "number_of_inputs": ("Number of Input Fields", lambda v: f"{int(v)}"),
    "number_of_buttons": ("Number of Buttons", lambda v: f"{int(v)}"),
    "number_of_images": ("Number of Images", lambda v: f"{int(v)}"),
    "number_of_option": ("Number of Dropdown Options", lambda v: f"{int(v)}"),
    "number_of_list": ("Number of List Items", lambda v: f"{int(v)}"),
    "number_of_th": ("Number of Table Header Cells", lambda v: f"{int(v)}"),
    "number_of_tr": ("Number of Table Rows", lambda v: f"{int(v)}"),
    "number_of_href": ("Number of Linked Stylesheets", lambda v: f"{int(v)}"),
    "number_of_paragraph": ("Number of Paragraphs", lambda v: f"{int(v)}"),
    "number_of_script": ("Number of Scripts", lambda v: f"{int(v)}"),
    "length_of_title": ("Page Title Length", lambda v: f"{int(v)} characters"),
    "has_h1": ("Has an H1 Heading", lambda v: "Yes" if v else "No"),
    "has_h2": ("Has an H2 Heading", lambda v: "Yes" if v else "No"),
    "has_h3": ("Has an H3 Heading", lambda v: "Yes" if v else "No"),
    "length_of_text": ("Amount of Visible Text", lambda v: f"{int(v)} characters"),
    "number_of_clickable_button": ("Number of Clickable Buttons", lambda v: f"{int(v)}"),
    "number_of_a": ("Number of Hyperlinks", lambda v: f"{int(v)}"),
    "number_of_img": ("Number of Images (<img>)", lambda v: f"{int(v)}"),
    "number_of_div": ("Number of <div> Elements", lambda v: f"{int(v)}"),
    "number_of_figure": ("Number of Figures", lambda v: f"{int(v)}"),
    "has_footer": ("Has a Footer", lambda v: "Yes" if v else "No"),
    "has_form": ("Has a Form", lambda v: "Yes" if v else "No"),
    "has_text_area": ("Has a Text Area", lambda v: "Yes" if v else "No"),
    "has_iframe": ("Has an Embedded iframe", lambda v: "Yes" if v else "No"),
    "has_text_input": ("Has a Text Input Field", lambda v: "Yes" if v else "No"),
    "number_of_meta": ("Number of Meta Tags", lambda v: f"{int(v)}"),
    "has_nav": ("Has a Navigation Menu", lambda v: "Yes" if v else "No"),
    "has_object": ("Has an <object> Element", lambda v: "Yes" if v else "No"),
    "has_picture": ("Has a <picture> Element", lambda v: "Yes" if v else "No"),
    "number_of_sources": ("Number of Media Sources", lambda v: f"{int(v)}"),
    "number_of_span": ("Number of <span> Elements", lambda v: f"{int(v)}"),
    "number_of_table": ("Number of Tables", lambda v: f"{int(v)}"),
}

TOP_N_CONTRIBUTING_FEATURES = 8


@st.cache_resource(show_spinner=False)
def build_explainer(_model, cache_key):
    """Build (and cache) a SHAP TreeExplainer for a model. cache_key
    distinguishes the content vs URL model in Streamlit's cache, since the
    model object itself (_model, leading underscore) is excluded from
    hashing - sklearn estimators aren't reliably hashable."""
    return shap.TreeExplainer(_model)


def _phishing_class_contributions(shap_output):
    """Extract the per-feature contribution toward the phishing (class 1)
    prediction, handling both SHAP API return shapes: older versions return
    a list of one array per class, newer versions return a single
    (n_samples, n_features, n_classes) array."""
    if isinstance(shap_output, list):
        return np.array(shap_output[1])[0]
    arr = np.array(shap_output)
    if arr.ndim == 3:
        return arr[0, :, 1]
    return arr[0]


def get_top_contributing_features(explainer, features_df, feature_display, top_n=TOP_N_CONTRIBUTING_FEATURES):
    """Return the top_n features that most influenced this specific
    prediction, ranked by absolute SHAP contribution. Each entry reports
    the plain-English label, the formatted value for this input, and
    whether it pushed the verdict toward phishing or legitimate - derived
    from the sign of the model's own contribution, not a fixed rule."""
    shap_output = explainer.shap_values(features_df)
    contributions = _phishing_class_contributions(shap_output)

    results = []
    for name, contribution, raw_value in zip(features_df.columns, contributions, features_df.iloc[0]):
        label, formatter = feature_display.get(name, (name, lambda v: str(v)))
        results.append({
            "label": label,
            "value_text": formatter(raw_value),
            "contribution": float(contribution),
            "direction": "phishing" if contribution > 0 else "legitimate",
        })

    results.sort(key=lambda item: -abs(item["contribution"]))
    return results[:top_n]


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
        "description": (
            "Both the content-based and URL-based checks are consistent with a "
            "legitimate website. No significant phishing indicators were detected."
        ),
        "actions": [
            "Standard browsing precautions still apply — never share credentials "
            "in response to an unsolicited email or message.",
            "Confirm the domain matches the organisation you intend to visit "
            "before entering any information.",
            "Keep your browser and security software up to date.",
        ],
    },
    {
        "label": "Low Risk",
        "css_class": "verdict-safe",
        "icon": "🙂",
        "description": (
            "The site is predominantly consistent with legitimate characteristics, "
            "though minor anomalies were present in one or both checks."
        ),
        "actions": [
            "Examine the URL closely for unusual subdomains, misspellings, or "
            "extra characters before proceeding.",
            "Confirm the connection uses HTTPS and a valid certificate before "
            "entering any information.",
            "Avoid submitting sensitive details until you have independently "
            "confirmed the site's authenticity.",
        ],
    },
    {
        "label": "Moderate Risk",
        "css_class": "verdict-unknown",
        "icon": "⚠️",
        "description": (
            "The two checks produced mixed or inconclusive signals. This does not "
            "confirm phishing, but the result should not be treated as trustworthy."
        ),
        "actions": [
            "Do not enter login credentials, payment details, or personal "
            "information on this page.",
            "Navigate to the organisation's website directly — via a bookmark or "
            "search engine — rather than through this link.",
            "If the link was received unexpectedly, verify its legitimacy with "
            "the sender through a separate communication channel.",
        ],
    },
    {
        "label": "High Risk",
        "css_class": "verdict-danger",
        "icon": "⛔",
        "description": (
            "A majority of the indicators assessed are consistent with known "
            "phishing characteristics."
        ),
        "actions": [
            "Close this page and do not interact with any of its content.",
            "Do not click embedded links or download any attachments from this "
            "source.",
            "Report the URL to your organisation's IT/security team or your "
            "email provider.",
        ],
    },
    {
        "label": "Very High Risk",
        "css_class": "verdict-danger",
        "icon": "🚨",
        "description": (
            "Both checks strongly agree on indicators typically associated with "
            "phishing websites."
        ),
        "actions": [
            "Do not enter any information on this page under any circumstances.",
            "Close the page immediately. If it was reached via email, delete or "
            "report the message as phishing rather than replying.",
            "Report the URL to a phishing-reporting service (e.g. Google Safe "
            "Browsing or a national CERT) and warn anyone else who may have "
            "received the same link.",
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
            color: #000000;
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

        # Single-line on purpose - see the note in render_combined_risk()
        # about blank lines breaking Streamlit's HTML rendering.
        card_html = (
            f'<div class="verdict-card {css_class}">'
            f'<h3>{title}</h3>'
            f'<div class="verdict-label">{icon} {label}</div>'
            f'{confidence_html}'
            f'<div style="color:#666;font-size:0.85rem;margin-top:0.5rem;">{description}</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


def verdict_word(prediction):
    if prediction is None:
        return "unknown"
    return "phishing" if prediction == 1 else "legitimate"


def render_why_expander(column, top_features):
    """Render an expandable 'Why this result?' panel under a verdict card,
    listing the top SHAP-contributing features for that specific prediction
    and which way each one pushed the verdict. Rendered as one markdown
    call per row (rather than one large multi-line block) so there's no
    blank line for Streamlit's markdown parser to misread as a code block."""
    if not top_features:
        return
    with column:
        with st.expander("Why this result?"):
            for item in top_features:
                if item["direction"] == "phishing":
                    arrow, color, direction_word = "🔺", "#B33A3A", "phishing"
                else:
                    arrow, color, direction_word = "🔻", "#3A7D44", "legitimate"
                row_html = (
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                    f'padding:0.35rem 0;border-bottom:1px solid #eee;font-size:0.85rem;gap:0.5rem;">'
                    f'<span>{arrow} <strong>{item["label"]}:</strong> {item["value_text"]}</span>'
                    f'<span style="color:{color};white-space:nowrap;">→ {direction_word}</span>'
                    f'</div>'
                )
                st.markdown(row_html, unsafe_allow_html=True)
            st.caption(
                "Ranked by SHAP contribution - each feature's actual influence on "
                "this specific prediction, not a fixed rule."
            )


def render_combined_risk(percentage, unverified):
    """Render the combined 0-100% risk score, its band label, a plain-English
    description of what that band means, and the recommended actions for
    that risk level. Shown below the two independent verdict cards, not in
    place of them."""
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

    actions_html = "".join(f"<li>{action}</li>" for action in band["actions"])

    # Built as a single line (no embedded newlines/leading whitespace) on
    # purpose - a blank line inside an indented multi-line f-string here
    # gets misread by Streamlit's markdown parser as the start of a code
    # block, which renders the HTML after it as literal text instead of
    # markup. This has no such blank-line risk.
    card_html = (
        f'<div class="verdict-card {band["css_class"]}" style="margin-top:1rem;">'
        f'<h3>Overall Risk</h3>'
        f'<div class="verdict-label">{band["icon"]} {percentage:.0f}% — {band["label"]}</div>'
        f'{unverified_html}'
        f'<div style="color:#444;font-size:0.9rem;margin-top:0.6rem;line-height:1.5;">{band["description"]}</div>'
        f'<div style="color:#333;font-size:0.85rem;font-weight:600;margin-top:0.8rem;">Recommended actions</div>'
        f'<ul style="color:#555;font-size:0.9rem;margin-top:0.3rem;line-height:1.6;">{actions_html}</ul>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


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
        st.markdown("- **URL-based** — 38 features from the URL's structure alone "
                     "(no live fetch needed, so it always runs)")
        st.write(
            "The two verdicts are shown separately on purpose - a disagreement "
            "between them is useful information, not an error. An overall risk "
            "score below combines both into one plain-English recommendation. "
            "Each verdict also has a 'Why this result?' panel showing the top "
            "factors behind that specific prediction."
        )
        st.caption("For academic demonstration only - not a substitute for "
                   "professional security judgement.")

    # ---- Load models once ----
    content_model = load_model(CONTENT_MODEL_FILENAME)
    url_model = load_model(URL_MODEL_FILENAME)

    # ---- Build SHAP explainers once (cached) alongside the models ----
    content_explainer = build_explainer(content_model, "content") if content_model is not None else None
    url_explainer = build_explainer(url_model, "url") if url_model is not None else None

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
            content_top_features = None
            if response is not None:
                soup = BeautifulSoup(response.content, "html.parser")
                content_features = extract_content_features(soup)
                content_prediction, content_confidence, content_phishing_prob = get_prediction_details(
                    content_model, content_features
                )
                content_top_features = get_top_contributing_features(
                    content_explainer, content_features, CONTENT_FEATURE_DISPLAY
                )

            # --- URL-based track (always runs - pure string processing,
            # no dependency on the page being reachable) ---
            url_features = extract_url_features(url)
            url_prediction, url_confidence, url_phishing_prob = get_prediction_details(
                url_model, url_features
            )
            url_top_features = get_top_contributing_features(
                url_explainer, url_features, URL_FEATURE_DISPLAY
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
        render_why_expander(col1, content_top_features)
        render_verdict_card(
            col2, "URL", "Based on the URL's structure alone.",
            url_prediction, url_confidence,
        )
        render_why_expander(col2, url_top_features)

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

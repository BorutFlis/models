"""LIFE-Preserved input preparation, Selenium transport and resumable batches.

Only predictor values are sent to the public calculator. Record IDs stay local.
No browser or data file is opened at import time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

URL = "https://tessa-r.shinyapps.io/shiny/"
INPUT_IDS = {
    "Age": "age", "Sex": "sex", "Diabetes mellitus": "diabetesDiagnosis",
    "Atrial fibrillation": "af", "Current smoking": "isSmoking",
    "Previous HF hospitalization": "prev_hosp", "COPD": "copd",
    "NYHA class": "dummynyha34", "Ischaemic heart disease": "ihd",
    "NT-proBNP": "ntprobnp", "BMI": "bmi", "Heart rate": "heartrate",
    "Haemoglobin": "hb", "eGFR": "gfr",
}
RADIO_OPTIONS = {
    "sex": {"Male": "1", "Female": "0"},
    "isSmoking": {"Current smoker": "1", "Never or former smoker": "0"},
    "dummynyha34": {"III or IV": "1", "I or II": "0"},
    **{key: {"Yes": "1", "No": "0"} for key in (
        "prev_hosp", "prev_hosp6m", "diabetesDiagnosis", "copd", "ihd", "af"
    )},
    "hb_unit": {"mmol/L": "mmolL", "g/dL": "gdL"},
}
TREATMENTS = {
    "SGLT2-inhibitors": "SGLT2", "Finerenone": "Fine",
    "GLP-1 receptor agonists": "GLP1", "Tirzepatide": "Tirze",
}
# Explicit reference assumptions, not claimed population means. Numerical
# missingness can instead use means fitted on a supplied reference cohort.
DEFAULTS = {
    "age": 70.0, "sex": "0", "isSmoking": "0", "dummynyha34": "0",
    "prev_hosp": "0", "prev_hosp6m": "0", "diabetesDiagnosis": "0",
    "copd": "0", "ihd": "0", "af": "0", "bmi": 25.0,
    "ntprobnp": 300.0, "heartrate": 70.0, "gfr": 75.0, "hb": 14.0,
}
BOUNDS = {"age": (40, 89), "bmi": (18, 55), "ntprobnp": (0, 20000),
          "heartrate": (40, 120), "gfr": (15, 90), "hb": (3, 11)}
TABS = {
    "10-year risk": ("currentTenYearRisk", "tenYearRiskWithTreatment", "tenyearARR", "resultBarChart"),
    "Lifetime risk": ("currentLifetimeRisk80", "lifetimeRiskWithTreatment", "lifetimeARR", "resultBarChartLife"),
    "HF hospitalization-free life expectancy": (
        "medianLifeExp", "medianLifeExpWithTreatment", "Benefit", "resultPolygon"
    ),
}


def attach_identifiers(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with selectors; retain all original source mappings."""
    result = mapping.copy()
    result["shiny_input_id"] = result["LIFE_Preserved_variable"].map(INPUT_IDS)
    result["shiny_input_type"] = result.shiny_input_id.map(
        lambda key: "radio" if key in RADIO_OPTIONS else "number" if pd.notna(key) else None
    )
    result["shiny_input_selector"] = [
        f'input[name="{key}"]' if kind == "radio" else
        f'input[id="{key}"]' if pd.notna(key) else None
        for key, kind in zip(result.shiny_input_id, result.shiny_input_type)
    ]
    return result


def encode_radio(field: str, value) -> str:
    """Encode labels or site-coded 0/1; source Sex 0=M/1=F is handled upstream."""
    text = str(value).strip().lower()
    labels = {label.lower(): code for label, code in RADIO_OPTIONS[field].items()}
    if text in labels:
        return labels[text]
    if field == "sex":
        labels.update({"m": "1", "f": "0"})
    elif field == "dummynyha34":
        labels.update({"i": "0", "ii": "0", "iii": "1", "iv": "1",
                       "2": "0", "3": "1", "4": "1"})
    else:
        labels.update({"y": "1", "n": "0", "true": "1", "false": "0"})
    labels.update({"0": "0", "0.0": "0", "1": "1", "1.0": "1"})
    if text not in labels:
        raise ValueError(f"Unsupported category for {field}: {value!r}")
    return labels[text]


def prepare_profile(row: Mapping, mapping: pd.DataFrame, *, defaults=None,
                    hb_unit="gdL") -> dict:
    """Resolve site IDs, LIFE labels or mapped source columns in that order.

    Direct site IDs use site encodings. Numeric mapped source Sex follows the
    CSV's M=0/F=1 convention. No implicit haemoglobin unit conversion occurs.
    """
    if hb_unit not in {"gdL", "mmolL"}:
        raise ValueError("hb_unit must be gdL or mmolL")
    reference = {**DEFAULTS, **(defaults or {})}
    if hb_unit == "mmolL" and (not defaults or "hb" not in defaults):
        reference["hb"] = DEFAULTS["hb"] / 1.6114
    values, imputed, clipped = {}, {}, {}
    by_id = {INPUT_IDS.get(item.LIFE_Preserved_variable): item
             for item in mapping.itertuples() if item.LIFE_Preserved_variable in INPUT_IDS}
    for field in DEFAULTS:
        item = by_id.get(field)
        source = getattr(item, "high_risk_HES_schema_attribute", None)
        label = getattr(item, "LIFE_Preserved_variable", "")
        candidates = [field, label]
        if field == "ihd":
            candidates.append("Ischemic heart disease")
        if isinstance(source, str) and "|" not in source:
            candidates.append(source)
        selected = next((key for key in candidates
                         if key in row and pd.notna(row[key])), None)
        if selected is None:
            value = reference[field]
            imputed[field] = value
        else:
            value = row[selected]
            if field == "sex" and selected == source and str(value) in {"0", "1", "0.0", "1.0"}:
                value = "1" if float(value) == 0 else "0"
        if field in RADIO_OPTIONS:
            value = encode_radio(field, value)
        else:
            value = float(value)
            lower, upper = BOUNDS[field]
            # The supplied haemoglobin limits are in mmol/L. Keep the
            # equivalent limits when the form is configured for g/dL.
            if field == "hb" and hb_unit == "gdL":
                lower, upper = lower * 1.6114, upper * 1.6114
            if field in {"age", "bmi", "gfr", "hb", "ntprobnp"} and math.isfinite(value):
                bounded = min(max(value, lower), upper)
                if bounded != value:
                    clipped[field] = {"original": value, "submitted": bounded}
                value = bounded
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{field} outside website range [{lower}, {upper}]: {value}")
        values[field] = value
    values["hb_unit"] = hb_unit
    return {"inputs": values, "imputed": imputed, "clipped": clipped}


def fit_reference_defaults(frame: pd.DataFrame, mapping: pd.DataFrame, *, hb_unit="gdL") -> dict:
    """Fit numeric means and categorical modes from an explicit reference cohort.

    Use a training/reference cohort when avoiding evaluation-cohort leakage.
    Fully unavailable fields retain the documented reference assumptions.
    """
    prepared = [prepare_profile(row, mapping, hb_unit=hb_unit) for _, row in frame.iterrows()]
    result = dict(DEFAULTS)
    if hb_unit == "mmolL":
        result["hb"] /= 1.6114
    for field in DEFAULTS:
        observed = [p["inputs"][field] for p in prepared if field not in p["imputed"]]
        if observed:
            series = pd.Series(observed)
            result[field] = str(series.mode().iloc[0]) if field in RADIO_OPTIONS else float(series.mean())
    return result


def create_driver(*, headless=True):
    """Create a Chrome driver separately from prediction functions."""
    from selenium import webdriver
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


def interact_when_ready(driver, locator, action, description, timeout=60, *,
                        allow_modal=False, require_visible=True):
    """Retry idempotent actions on freshly located controls during UI updates."""
    from selenium.common.exceptions import (
        ElementClickInterceptedException, ElementNotInteractableException,
        NoSuchElementException, StaleElementReferenceException,
    )
    from selenium.webdriver.support.ui import WebDriverWait

    def attempt(d):
        if not allow_modal and any(e.is_displayed() for e in d.find_elements("css selector", ".modal, .modal-backdrop")):
            return False
        element = d.find_element(*locator)
        if not element.is_enabled() or (require_visible and not element.is_displayed()):
            return False
        if require_visible:
            d.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        action(element)
        return True

    return WebDriverWait(driver, timeout, poll_frequency=0.2, ignored_exceptions=(
        NoSuchElementException, StaleElementReferenceException,
        ElementNotInteractableException, ElementClickInterceptedException,
    )).until(attempt, message=f"Could not {description}; locator={locator!r}")


def open_calculator(driver, timeout=60):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    driver.get(URL)
    wait = WebDriverWait(driver, timeout)
    wait.until(lambda d: d.execute_script(
        "return !!(window.Shiny && Shiny.shinyapp && Shiny.shinyapp.$socket "
        "&& Shiny.shinyapp.$socket.readyState === 1 && document.querySelector('#age'));"
    ))
    # The introductory notice is informational and closes with Escape.
    interact_when_ready(driver, (By.ID, "shiny-modal"),
                        lambda element: element.send_keys(Keys.ESCAPE),
                        "dismiss introductory modal", timeout, allow_modal=True)
    wait.until(lambda d: not any(e.is_displayed() for e in d.find_elements(
        By.CSS_SELECTOR, ".modal, .modal-backdrop")), "Introductory modal or backdrop did not close")


def fill_form(driver, inputs: Mapping, treatments=(), timeout=60):
    """Set every predictor and clear treatments left over from an earlier row."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    for field in BOUNDS:
        def set_number(element):
            # Clear on every attempt so a partial send_keys cannot duplicate text.
            element.clear()
            element.send_keys(str(inputs[field]), Keys.TAB)
        interact_when_ready(driver, (By.ID, field), set_number,
                            f"fill numeric input {field}", timeout)
    for field in RADIO_OPTIONS:
        def select_radio(target):
            if not target.is_selected():
                driver.execute_script("arguments[0].click();", target)
        interact_when_ready(
            driver, (By.CSS_SELECTOR, f'input[name="{field}"][value="{inputs[field]}"]'),
            select_radio, f"select radio input {field}", timeout, require_visible=False)
    # Shiny numeric inputs debounce changes; allow their final events to flush.
    time.sleep(0.6)
    # Treatment UI is reactive to the BMI and may replace its input elements.
    wait = WebDriverWait(driver, timeout)
    wait.until(lambda d: d.execute_script("return !document.documentElement.classList.contains('shiny-busy')"))
    requested = {TREATMENTS.get(value, value) for value in treatments}
    if not requested <= set(TREATMENTS.values()):
        raise ValueError(f"Unknown treatment: {requested - set(TREATMENTS.values())}")
    available = {e.get_attribute("value") for e in driver.find_elements(By.CSS_SELECTOR, 'input[name="treatment"]')}
    if not requested <= available:
        raise ValueError(f"Treatment unavailable for this profile: {requested - available}")
    for value in available:
        def set_treatment(element):
            if element.is_selected() != (value in requested):
                driver.execute_script("arguments[0].click();", element)
        interact_when_ready(
            driver, (By.CSS_SELECTOR, f'input[name="treatment"][value="{value}"]'),
            set_treatment, f"set treatment {value}", timeout, require_visible=False)
    # Confirm actual DOM values before triggering calculation.
    wait.until(lambda d: d.execute_script("""
        return Object.entries(arguments[0]).every(([key,value]) => {
          const numeric = document.querySelector('input[id="'+key+'"]');
          const radio = document.querySelector('input[name="'+key+'"]:checked');
          return numeric ? Number(numeric.value) === Number(value) : radio && radio.value === String(value);
        });
    """, dict(inputs)))


def parse_result(value: str, label: str) -> dict:
    """Preserve unavailable calculator values as JSON null, never as zero."""
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?|NA)\s*(%|years?)\s*", value)
    if not match:
        raise ValueError(f"Unexpected calculator output: {value!r}")
    missing = match[1] == "NA"
    return {"value": None if missing else float(match[1]),
            "unit": "%" if match[2] == "%" else "years",
            "label": label, "text": value,
            "status": "unavailable" if missing else "ok"}


def collect_results(driver, output_dir, timeout=60) -> dict:
    """Activate all tabs, capture nine labelled values, three PNGs and HTML."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    wait, metrics, plots = WebDriverWait(driver, timeout), {}, {}
    for title, ids in TABS.items():
        interact_when_ready(
            driver, (By.XPATH, f'//a[@data-toggle="tab" and normalize-space(.)="{title}"]'),
            lambda element: element.click(), f"open results tab {title}", timeout)
        for output_id in ids:
            def ready(d):
                element = d.find_element(By.ID, output_id)
                if "recalculating" in (element.get_attribute("class") or ""):
                    return False
                if element.find_elements(By.CSS_SELECTOR, ".shiny-output-error") or "shiny-output-error" in (element.get_attribute("class") or ""):
                    raise ValueError(f"Calculator error in {output_id}: {element.text}")
                return element if element.find_elements(By.CSS_SELECTOR, "h3, img") else False
            element = wait.until(ready)
            if output_id == ids[-1]:
                image = element.find_element(By.TAG_NAME, "img")
                wait.until(lambda d: d.execute_script("return arguments[0].complete && arguments[0].naturalWidth > 0", image))
                destination = directory / f"{output_id}.png"
                src = image.get_attribute("src")
                if src.startswith("data:image/png;base64,"):
                    destination.write_bytes(base64.b64decode(src.split(",", 1)[1]))
                else:
                    image.screenshot(str(destination))
                plots[output_id] = str(destination.resolve())
            else:
                metrics[output_id] = parse_result(element.find_element(By.TAG_NAME, "h3").text,
                                                   element.find_element(By.TAG_NAME, "p").text)
    snapshot = directory / "assessment.html"
    snapshot.write_text(driver.page_source, encoding="utf-8")
    return {"metrics": metrics, "plots": plots, "html": str(snapshot.resolve())}


def predict_profile(driver, profile: dict, output_dir, *, treatments=(), timeout=60) -> dict:
    """Use an injected driver; fresh page prevents stale output across profiles."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    open_calculator(driver, timeout)
    fill_form(driver, profile["inputs"], treatments, timeout)
    # Arm a completion observer before click. Do not rely on scores changing:
    # identical consecutive profiles legitimately produce identical scores.
    driver.execute_script("""
        window.__lifeComplete = false;
        $(document).off('shiny:value.life').on('shiny:value.life', function(event) {
          if (event.name === 'currentTenYearRisk' && event.value &&
              String(event.value.html || event.value).includes('%')) window.__lifeComplete = true;
        });
    """)
    # Do not retry a submitted click inside the same assessment: that could
    # trigger duplicate calculations. Wait for clickability, then click once.
    from selenium.webdriver.support import expected_conditions as EC
    submit = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.ID, "submit")), "Calculate risk button not ready")
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", submit)
    submit.click()
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return window.__lifeComplete === true"))
    return {**profile, "treatments": list(treatments), **collect_results(driver, output_dir, timeout)}


def run_batch(frame: pd.DataFrame, mapping: pd.DataFrame, predictor: Callable, output_dir,
              *, defaults=None, hb_unit="gdL", treatments=(), retries=2, delay=1.0,
              resume=True) -> pd.DataFrame:
    """Predict sequentially with injectable transport and a JSON checkpoint per row.

    predictor(profile, directory, treatments=...) returns a result dictionary.
    Resume requires matching ID, inputs, treatment scenario and schema version.
    Errors are recorded per row; failed rows are retried on a subsequent run.
    """
    if not frame.index.is_unique:
        raise ValueError("Batch index must uniquely identify rows")
    if retries < 0 or delay < 0:
        raise ValueError("retries and delay must be nonnegative")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    treatments = tuple(treatments)
    records = []
    for row_id, row in frame.iterrows():
        profile, key = None, None
        try:
            profile = prepare_profile(row, mapping, defaults=defaults, hb_unit=hb_unit)
            identity = {"version": 1, "row_id": str(row_id), "profile": profile,
                        "treatments": sorted(treatments)}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            row_dir = directory / key
            checkpoint = row_dir / "result.json"
            if resume and checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved["status"] == "ok" and all(Path(p).exists() for p in saved["plots"].values()) and Path(saved["html"]).exists():
                    records.append(saved)
                    continue
            for attempt in range(retries + 1):
                try:
                    result = predictor(profile, row_dir, treatments=treatments)
                    record = {**result, "row_id": str(row_id), "key": key, "status": "ok",
                              "timestamp": datetime.now(timezone.utc).isoformat()}
                    break
                except Exception:
                    if attempt == retries:
                        raise
                    time.sleep(delay * (2 ** attempt))
        except Exception as error:
            record = {"row_id": str(row_id), "key": key, "status": "error",
                      "error": f"{type(error).__name__}: {error}", "profile": profile}
        if key:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(checkpoint)
        records.append(record)
        time.sleep(delay)
    summary = pd.json_normalize(records)
    summary.to_csv(directory / "predictions.csv", index=False)
    return summary

import sys
import traceback
from pathlib import Path
from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tmp' / 'life_deps'))
sys.path.insert(0, str(root))
import pandas as pd
from disease_progression.scripts.life_preserved import create_driver, predict_profile, run_batch, INPUT_IDS

mapping = pd.DataFrame([{'LIFE_Preserved_variable': label, 'high_risk_HES_schema_attribute': None}
                        for label in INPUT_IDS])
driver = create_driver()
def predict(profile, directory, **kwargs):
    try:
        return predict_profile(driver, profile, directory, timeout=20, **kwargs)
    except Exception:
        traceback.print_exc()
        print(driver.current_url, driver.title, flush=True)
        print(driver.execute_script("return {shiny:!!window.Shiny,app:!!window.Shiny?.shinyapp,socket:!!window.Shiny?.shinyapp?.$socket,ready:window.Shiny?.shinyapp?.$socket?.readyState,modal:!!document.querySelector('#shiny-modal'),inputs:document.querySelector('#age')?.value};"), flush=True)
        driver.save_screenshot(str(root / 'tmp' / 'life_failure.png'))
        raise
try:
    summary = run_batch(pd.DataFrame({'age': [70, 70]}, index=['synthetic-a', 'synthetic-b']),
                        mapping, predict, root / 'tmp' / 'life_smoke',
                        retries=0, delay=1, resume=False)
    print(summary[['row_id', 'status']].to_string(index=False))
    if summary.status.ne('ok').any():
        print(summary.get('error').to_string(index=False))
        driver.save_screenshot(str(root / 'tmp' / 'life_failure.png'))
        raise SystemExit(1)
    print(summary.filter(regex='metrics.*value').to_string(index=False))
    treated = run_batch(pd.DataFrame({'age': [70]}, index=['synthetic-treated']),
                       mapping, predict, root / 'tmp' / 'life_smoke', treatments=['SGLT2'],
                       retries=0, delay=0, resume=False)
    print(treated[['row_id', 'status']].to_string(index=False))
    print(treated.filter(regex='metrics.*value').to_string(index=False))
    if treated.status.ne('ok').any():
        print(treated.get('error').to_string(index=False))
        raise SystemExit(1)
    driver.save_screenshot(str(root / 'tmp' / 'life_success.png'))
finally:
    driver.quit()

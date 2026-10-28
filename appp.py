"""
AI-Powered Loan Decision & Explanation System
Executive Underwriting, RAG & Risk Simulation Platform
Entrypoint: streamlit run appp.py
"""
import sys
import os
from pathlib import Path

# Add credit-risk-app directory to sys.path
APP_DIR = Path(__file__).resolve().parent / "credit-risk-app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Change working directory so relative assets and templates resolve
os.chdir(str(APP_DIR))

# Execute complete credit-risk-app suite
target_script = APP_DIR / "app.py"
with open(target_script, "r", encoding="utf-8") as f:
    app_code = f.read()

# Strip any standalone launcher from target script
if 'if __name__ == "__main__":' in app_code:
    app_code = app_code.split('if __name__ == "__main__":')[0]

exec(compile(app_code, str(target_script), "exec"))

if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

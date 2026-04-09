# streamlit_app.py
#
# Entry point for Streamlit Cloud deployment.
# Streamlit Cloud runs this file from the repo root.
# The salary_predictor package is installed via requirements.txt (-e .)
# so all imports resolve correctly.

from salary_predictor.dashboard.app import main

main()

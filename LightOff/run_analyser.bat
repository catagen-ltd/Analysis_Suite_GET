@echo off
cd /d "C:\PythonAnalyser\Python Analysis Suite\LO Analyser"
call ".venv\Scripts\activate"
powershell -Command "python 'LO_Analyser_1.0_MKS.py'"
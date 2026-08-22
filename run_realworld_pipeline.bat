@echo off
setlocal
cd /d "%~dp0"

echo === KMVision Real-World Pipeline ===
echo.
echo Step 1: PMC scrape + image download
python real_dataset\run_collection.py --km-only
if errorlevel 1 exit /b 1

echo.
echo Step 2: Manual labeling (interactive — run until queue empty)
echo   python real_dataset\labeler.py --type km
echo.
echo Step 3: Fine-tune macro model on labeled real-world KM charts
python train_realworld.py --chart-type km --max-global-steps 500 --output-dir checkpoints/realworld_macro_km
if errorlevel 1 (
  echo Fine-tune skipped or failed — label more charts first.
  exit /b 0
)

echo.
echo Step 4: Evaluate real-world checkpoint
python eval_realworld.py --checkpoint checkpoints/realworld_macro_km/final --output-dir evaluation/results/realworld_macro

echo.
echo Pipeline complete. Next: generate real-world tiles and fine-tune Stage 2.
exit /b 0

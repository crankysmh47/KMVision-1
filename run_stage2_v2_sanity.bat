@echo off
setlocal
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset

echo === Stage 2 v2: regenerate capped tiles ===
python scripts\generate_stage2_tiles.py --dataset_root "%DATASET%" --source testing --max_charts 12000 --clear_output --seed 42
if errorlevel 1 exit /b 1

echo === Stage 2 v2: 500-step sanity training (fresh init) ===
python train_stage2.py --dataset_root "%DATASET%" --output_dir checkpoints/stage2_v2 --max_global_steps 500 --no_auto_resume
if errorlevel 1 exit /b 1

echo === Stage 2 v2: sanity JSON check (5 holdout tiles) ===
python scripts\stage2_sanity_check.py --checkpoint checkpoints/stage2_v2/final --max-samples 5
if errorlevel 1 exit /b 1

echo === Stage 2 v2: spot eval (10 tiles) ===
python eval_stage2.py --checkpoint checkpoints/stage2_v2/final --max-samples 10 --seed 0

echo Done.

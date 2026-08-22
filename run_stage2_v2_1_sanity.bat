@echo off
setlocal
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset

echo === Stage 2 v2.1: normalized local tiles ===
python scripts\generate_stage2_tiles.py --dataset_root "%DATASET%" --source testing --max_charts 12000 --clear_output --seed 42 --coordinate-space normalized_local
if errorlevel 1 exit /b 1

echo === Stage 2 v2.1: 500-step sanity (fresh LoRA, prefix-masked loss) ===
python train_stage2.py --dataset_root "%DATASET%" --output_dir checkpoints/stage2_v2_1 --max_global_steps 500 --no_auto_resume
if errorlevel 1 exit /b 1

echo === Sanity JSON check (10 tiles, prefix forced) ===
python scripts\stage2_sanity_check.py --checkpoint checkpoints/stage2_v2_1/final --max-samples 10 --force-json-prefix

echo Done. If sanity passes, run full train: python train_stage2.py --max_global_steps 3000 --no_auto_resume

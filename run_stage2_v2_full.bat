@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset
set LOG=logs\stage2_v2_full_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG=%LOG: =0%
mkdir logs 2>nul

echo [%date% %time%] Stage 2 v2 full pipeline started >> "%LOG%"
echo Log: %LOG%

echo === Resume training 500 -^> 3000 === >> "%LOG%"
python train_stage2.py ^
  --dataset_root "%DATASET%" ^
  --output_dir checkpoints/stage2_v2 ^
  --max_global_steps 3000 ^
  --auto_resume >> "%LOG%" 2>&1
if errorlevel 1 (
  echo TRAIN FAILED exit=!errorlevel! >> "%LOG%"
  exit /b 1
)

echo === Sanity check (10 holdout tiles) === >> "%LOG%"
python scripts\stage2_sanity_check.py ^
  --checkpoint checkpoints/stage2_v2/final ^
  --max-samples 10 >> "%LOG%" 2>&1

echo === Holdout eval (150 tiles) === >> "%LOG%"
python eval_stage2.py ^
  --checkpoint checkpoints/stage2_v2/final ^
  --max-samples 150 ^
  --seed 0 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo EVAL FAILED exit=!errorlevel! >> "%LOG%"
  exit /b 1
)

echo [%date% %time%] Pipeline complete. Results: evaluation\results\stage2_v2_holdout\ >> "%LOG%"
echo Done. See %LOG%

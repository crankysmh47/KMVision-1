@echo off
setlocal
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset
set LOG=logs\stage2_regen_train1_%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set LOG=%LOG: =0%

echo === Regenerate Stage 2 tiles from train_1 (KM pool) ===
echo Log: %LOG%

python scripts\generate_stage2_tiles.py ^
  --dataset_root "%DATASET%" ^
  --source train_1 ^
  --max_charts 12500 ^
  --output_dir "%DATASET%\stage2_train1" ^
  --holdout_dir "%DATASET%\stage2_train1_holdout" ^
  --clear_output ^
  > "%LOG%.log" 2>&1

if errorlevel 1 (
  echo Tile regen FAILED. See %LOG%.log
  exit /b 1
)

echo Tile regen complete. Output: %DATASET%\stage2_train1
exit /b 0

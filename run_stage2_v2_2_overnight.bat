@echo off

setlocal

cd /d "%~dp0"



set DATASET=C:\sem4\KMVision-1 Data\dataset

set LOG=logs\stage2_v2_2_overnight_%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%

set LOG=%LOG: =0%



echo === Stage 2 v2.2: 500-step sanity (flat_xy targets) ===

python train_stage2.py --dataset_root "%DATASET%" --output_dir checkpoints/stage2_v2_1 --max_global_steps 500 --no_auto_resume

if errorlevel 1 exit /b 1



echo === Sanity JSON check (10 tiles, prefix forced) ===

python scripts\stage2_sanity_check.py --checkpoint checkpoints/stage2_v2_1/final --max-samples 10 --force-json-prefix

if errorlevel 1 (

  echo Sanity FAILED - not starting full train.

  exit /b 1

)



echo === Sanity PASSED: 3000-step full train ===

python train_stage2.py --dataset_root "%DATASET%" --output_dir checkpoints/stage2_v2_1 --max_global_steps 3000 --no_auto_resume

if errorlevel 1 exit /b 1



echo === Holdout eval (150 tiles, prefix forced) ===

python eval_stage2.py --checkpoint checkpoints/stage2_v2_1/final --force-json-prefix --max-samples 150



echo Done.



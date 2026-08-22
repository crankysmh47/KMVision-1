@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset
set CKPT=checkpoints/stage2_v2_1
set LOG=logs\stage2_v2_2_train10k_%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set LOG=%LOG: =0%

echo === Continue Stage 2 v2.2: 3000 -^> 10000 steps ===
python train_stage2.py --dataset_root "%DATASET%" --output_dir "%CKPT%" --max_global_steps 10000 --checkpoint_every 2500 --auto_resume >> "%LOG%.log" 2>&1
if errorlevel 1 exit /b 1

for %%S in (2500 5000 7500) do (
  python eval_stage2.py --checkpoint "%CKPT%\step_00%%S" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_v2_2_10k/step_%%S >> "%LOG%.log" 2>&1
)
python eval_stage2.py --checkpoint "%CKPT%\final" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_v2_2_10k/final >> "%LOG%.log" 2>&1

for /f "delims=" %%F in ('dir /b /o-d evaluation\results\stage2_v2_2_10k\final\eval_*.jsonl 2^>nul') do (
  python eval_e2e.py --stage2-jsonl evaluation/results/stage2_v2_2_10k/final/%%F --max-charts 12 --output-dir evaluation/results/e2e_v2_2_10k >> "%LOG%.log" 2>&1
  goto :done
)
:done
echo Complete. Log: %LOG%.log
exit /b 0

@echo off
setlocal
cd /d "%~dp0"

set DATASET=C:\sem4\KMVision-1 Data\dataset
set CKPT=checkpoints/stage2_train1_10k
set LOG=logs\stage2_train1_10k_%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set LOG=%LOG: =0%

echo === Stage 2 train_1 tiles: 10k steps with checkpoint evals ===
echo Checkpoint dir: %CKPT%
echo Log: %LOG%.log

python train_stage2.py ^
  --dataset_root "%DATASET%" ^
  --image_dir "%DATASET%\stage2_train1\images\km" ^
  --label_dir "%DATASET%\stage2_train1\labels\km" ^
  --output_dir "%CKPT%" ^
  --max_global_steps 10000 ^
  --checkpoint_every 2500 ^
  --no_auto_resume ^
  >> "%LOG%.log" 2>&1
if errorlevel 1 exit /b 1

echo === Eval @ 2500 ===
python eval_stage2.py --checkpoint "%CKPT%\step_002500" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_train1_10k/step_2500 >> "%LOG%.log" 2>&1

echo === Eval @ 5000 ===
python eval_stage2.py --checkpoint "%CKPT%\step_005000" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_train1_10k/step_5000 >> "%LOG%.log" 2>&1

echo === Eval @ 7500 ===
python eval_stage2.py --checkpoint "%CKPT%\step_007500" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_train1_10k/step_7500 >> "%LOG%.log" 2>&1

echo === Eval @ 10000 (final) ===
python eval_stage2.py --checkpoint "%CKPT%\final" --force-json-prefix --max-samples 150 --output-dir evaluation/results/stage2_train1_10k/final >> "%LOG%.log" 2>&1

echo === E2E eval on final checkpoint ===
for /f "delims=" %%F in ('dir /b /o-d evaluation\results\stage2_train1_10k\final\eval_*.jsonl 2^>nul') do (
  python eval_e2e.py --stage2-jsonl evaluation/results/stage2_train1_10k/final/%%F --max-charts 12 --output-dir evaluation/results/e2e_train1_10k >> "%LOG%.log" 2>&1
  goto :e2e_done
)
:e2e_done

echo Done. See %LOG%.log
exit /b 0

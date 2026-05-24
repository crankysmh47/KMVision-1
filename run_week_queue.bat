@echo off
setlocal enabledelayedexpansion

REM KMVision-1 week-long experiment queue (chained checkpoints + metric gates).
REM Execute from repo root:  run_week_queue.bat
REM On gate failure the queue stops (use check_eval_gate.py --warn-only to override manually).

cd /d "%~dp0"
set REPO=%CD%
set LOGDIR=%REPO%\logs\week_queue
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set INIT_PHASE_B=checkpoints\phase_b\final
set RUN1_OUT=checkpoints\phase_c_run1_minified
set RUN2_OUT=checkpoints\phase_c_run2_chatml
set RUN3_OUT=checkpoints\phase_c_run3_low_lr
set EVAL1_DIR=evaluation\results\run1_minified
set EVAL2_DIR=evaluation\results\run2_chatml
set EVAL3_DIR=evaluation\results\run3_low_lr
set DATASET_ROOT=C:\sem4\KMVision-1 Data\dataset

echo [%date% %time%] Week queue starting >> "%LOGDIR%\queue.log"

REM --- Step 0: compress labels (skip if gate already passes) ---
echo === STEP 0: compress gate (pre-check) ===
python scripts\check_compress_gate.py >> "%LOGDIR%\00_compress_gate.log" 2>&1
if errorlevel 1 (
  echo === STEP 0a: compress_labels ===
  python scripts\compress_labels.py --input-dir "%DATASET_ROOT%\train_1\labels" --output-dir "%DATASET_ROOT%\labels_compressed" >> "%LOGDIR%\00_compress_labels.log" 2>&1
  if errorlevel 1 goto :failed
  echo === STEP 0b: compress gate ===
  python scripts\check_compress_gate.py >> "%LOGDIR%\00_compress_gate.log" 2>&1
  if errorlevel 1 goto :failed
) else (
  echo Compressed labels already pass token budget; skipping compress_labels.
)

if not exist "%INIT_PHASE_B%\adapter_model.safetensors" (
  echo FATAL: Phase B init checkpoint missing at %INIT_PHASE_B%
  goto :failed
)

REM --- Run 1: minified JSON, init from Phase B ---
echo === RUN 1: minified (init %INIT_PHASE_B%) ===
python train_phase_c.py --subset_size 30000 --learning_rate 5e-5 --output_dir %RUN1_OUT% --init_checkpoint %INIT_PHASE_B% >> "%LOGDIR%\01_train_run1.log" 2>&1
if errorlevel 1 goto :failed

python scripts\verify_checkpoint.py %RUN1_OUT%\final >> "%LOGDIR%\01_train_run1.log" 2>&1
if errorlevel 1 goto :failed

echo === EVAL 1 ===
python eval_inference.py --checkpoint %RUN1_OUT%/final --category km --max-samples 12 --output-dir %EVAL1_DIR% >> "%LOGDIR%\01_eval_run1.log" 2>&1
if errorlevel 1 goto :failed

python scripts\check_eval_gate.py --stage run1 --results-dir %EVAL1_DIR% >> "%LOGDIR%\01_eval_gate.log" 2>&1
if errorlevel 1 goto :gate_failed

REM --- Run 2: ChatML, chained from Run 1 ---
echo === RUN 2: ChatML (init %RUN1_OUT%\final) ===
python train_phase_c.py --subset_size 30000 --learning_rate 5e-5 --output_dir %RUN2_OUT% --init_checkpoint %RUN1_OUT%/final --use_chatml >> "%LOGDIR%\02_train_run2.log" 2>&1
if errorlevel 1 goto :failed

python scripts\verify_checkpoint.py %RUN2_OUT%\final >> "%LOGDIR%\02_train_run2.log" 2>&1
if errorlevel 1 goto :failed

echo === EVAL 2 ===
python eval_inference.py --checkpoint %RUN2_OUT%/final --category km --max-samples 12 --output-dir %EVAL2_DIR% >> "%LOGDIR%\02_eval_run2.log" 2>&1
if errorlevel 1 goto :failed

python scripts\check_eval_gate.py --stage run2 --results-dir %EVAL2_DIR% --previous-summary %EVAL1_DIR%\latest_summary.json >> "%LOGDIR%\02_eval_gate.log" 2>&1
if errorlevel 1 goto :gate_failed

REM --- Run 3: ChatML low LR, chained from Run 2 ---
echo === RUN 3: ChatML low LR (init %RUN2_OUT%\final) ===
python train_phase_c.py --subset_size 30000 --learning_rate 1e-5 --output_dir %RUN3_OUT% --init_checkpoint %RUN2_OUT%/final --use_chatml >> "%LOGDIR%\03_train_run3.log" 2>&1
if errorlevel 1 goto :failed

python scripts\verify_checkpoint.py %RUN3_OUT%\final >> "%LOGDIR%\03_train_run3.log" 2>&1
if errorlevel 1 goto :failed

echo === EVAL 3 ===
python eval_inference.py --checkpoint %RUN3_OUT%/final --category km --max-samples 12 --output-dir %EVAL3_DIR% >> "%LOGDIR%\03_eval_run3.log" 2>&1
if errorlevel 1 goto :failed

python scripts\check_eval_gate.py --stage run3 --results-dir %EVAL3_DIR% --previous-summary %EVAL2_DIR%\latest_summary.json >> "%LOGDIR%\03_eval_gate.log" 2>&1
if errorlevel 1 goto :gate_failed

echo [%date% %time%] Week queue completed OK >> "%LOGDIR%\queue.log"
echo All runs and gates passed.
exit /b 0

:gate_failed
echo [%date% %time%] Week queue stopped: eval gate failed >> "%LOGDIR%\queue.log"
echo Eval gate failed. See gate_*.json in evaluation/results/ and logs in %LOGDIR%
echo To continue anyway, re-run the next train step manually after reviewing metrics.
exit /b 1

:failed
echo [%date% %time%] Week queue FAILED >> "%LOGDIR%\queue.log"
echo A step failed. Check logs in %LOGDIR%
exit /b 1

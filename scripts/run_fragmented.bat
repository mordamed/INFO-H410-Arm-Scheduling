@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo  ARM32 Scheduler: Resilient Fragmented Execution
echo ========================================================
echo.

:: Configuration
set K_VAL=50
set EPISODES_VAL=5000
set SEEDS=42 43 44
set SIZES=10 30 50
set METHODS=bayesian csp mdp
set LOG_FILE=experiments\live_benchmark.log

:: Verification
C:\Python312\python.exe -c "import torch; print(f'GPU Check: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
echo.

:: Incremental Loop
for %%N in (%SIZES%) do (
    for %%S in (%SEEDS%) do (
        echo [EXECUTING] Size=%%N, Seed=%%S...
        echo.
        
        @REM We call the main script for just ONE combination of (Size, Seed) at a time.
        @REM Internal checkpointing will handle resume within DQN.
        @REM Benchmark resume will handle skipping already finished jobs.
        
        C:\Python312\python.exe experiments\run_all.py --k !K_VAL! --sizes %%N --seeds %%S --episodes !EPISODES_VAL! --methods %METHODS% --log !LOG_FILE! --resume
        
        if !errorlevel! neq 0 (
            echo.
            echo [WARNING] Job Size=%%N Seed=%%S failed or was interrupted.
            echo The system will attempt to continue.
            echo.
        )
    )
)

echo.
echo ========================================================
echo  Fragmented Execution Complete!
echo  Check experiments\results for high-resolution plots.
echo ========================================================
pause

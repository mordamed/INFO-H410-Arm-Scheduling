@echo off
echo ========================================================
echo  ARM32 Scheduler: Setup GPU Environment (Global Install)
echo ========================================================
echo.

@REM :: 1. Verify that C:\Python312 exists
@REM if not exist "C:\Python312\python.exe" (
@REM     echo [ERROR] Official Python 3.12 was not found at C:\Python312\python.exe.
@REM     pause
@REM     exit /b
@REM )

@REM :: 2. Install Project Dependencies globally to bypass WDAC path blocking
@REM echo Installing PyTorch with CUDA 12.4 support into C:\Python312...
@REM C:\Python312\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

@REM echo.
@REM echo Installing project requirements...
@REM C:\Python312\python.exe -m pip install -r requirements.txt
@REM C:\Python312\python.exe -m pip install -e .

:: 3. Verify GPU
echo.
echo ========================================================
echo GPU Verification:
C:\Python312\python.exe -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ========================================================
echo.

:: 4. Run the benchmark with logging enabled
echo Running benchmark (k=50) - output will be tracked in the terminal
echo AND simultaneously saved to experiments\live_benchmark.log ...
echo.

C:\Python312\python.exe experiments\run_all.py --k 50 --sizes 10 30 50 --seeds 42 43 44 --episodes 5000 --methods bayesian csp mdp --log experiments\live_benchmark.log

echo.
echo Benchmark Complete! Results are in experiments\results\
pause

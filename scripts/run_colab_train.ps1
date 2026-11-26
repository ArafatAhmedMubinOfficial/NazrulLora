# End-to-end training & evaluation launcher via Colab CLI
param(
    [string]$SessionName = "nazrul-train",
    [string]$Gpu = "T4",
    [double]$Stage1Minutes = 27.0,
    [double]$Stage2Minutes = 17.0,
    [int]$Stage1MaxSteps = 320,
    [int]$Stage2MaxSteps = 360
)

$ErrorActionPreference = "Stop"

Write-Host "Provisioning Colab session '$SessionName' ($Gpu)..."
colab new -s $SessionName --gpu $Gpu

Write-Host "Uploading project scripts..."
colab exec -s $SessionName --timeout 60 "import os; os.makedirs('/content/nazrul_ai/src', exist_ok=True)"
colab upload -s $SessionName src/prepare_data.py /content/nazrul_ai/src/prepare_data.py
colab upload -s $SessionName src/train.py /content/nazrul_ai/src/train.py
colab upload -s $SessionName src/eval_model.py /content/nazrul_ai/src/eval_model.py
colab upload -s $SessionName src/chat.py /content/nazrul_ai/src/chat.py

Write-Host "Building Nazrul + Basic Bangla dataset on Colab..."
echo "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False); import runpy; sys.argv=['prepare_data.py', '--out-dir', '/content/nazrul_ai/data/processed']; runpy.run_path('/content/nazrul_ai/src/prepare_data.py', run_name='__main__')" | colab exec -s $SessionName --timeout 600

Write-Host "Running 2-stage training (Stage1=${Stage1Minutes}m / ${Stage1MaxSteps} steps, Stage2=${Stage2Minutes}m / ${Stage2MaxSteps} steps)..."
echo "import subprocess, sys; subprocess.run([sys.executable, '/content/nazrul_ai/src/train.py', '--data-dir', '/content/nazrul_ai/data/processed', '--out-dir', '/content/nazrul_ai/outputs', '--stage1-minutes', '$Stage1Minutes', '--stage2-minutes', '$Stage2Minutes', '--stage1-max-steps', '$Stage1MaxSteps', '--stage2-max-steps', '$Stage2MaxSteps'], check=True)" | colab exec -s $SessionName --timeout 3600

Write-Host "Evaluating trained model..."
echo "import subprocess, sys; subprocess.run([sys.executable, '/content/nazrul_ai/src/eval_model.py', '--adapter-dir', '/content/nazrul_ai/outputs/nazrul-lora', '--val-jsonl', '/content/nazrul_ai/data/processed/cpt_val.jsonl', '--out-file', '/content/nazrul_ai/outputs/eval_report.json'], check=True)" | colab exec -s $SessionName --timeout 600

Write-Host "Downloading trained LoRA adapter and reports..."
New-Item -ItemType Directory -Force -Path outputs/nazrul-lora | Out-Null
colab download -s $SessionName /content/nazrul_ai/outputs/eval_report.json outputs/eval_report.json
colab download -s $SessionName /content/nazrul_ai/outputs/training_summary.json outputs/training_summary.json
colab download -s $SessionName /content/nazrul_ai/data/processed/dataset_stats.json outputs/dataset_stats.json
colab exec -s $SessionName --timeout 120 "import shutil; shutil.make_archive('/content/nazrul_lora', 'zip', '/content/nazrul_ai/outputs/nazrul-lora')"
colab download -s $SessionName /content/nazrul_lora.zip outputs/nazrul-lora.zip
Expand-Archive -Force outputs/nazrul-lora.zip outputs/nazrul-lora

Write-Host "Stopping Colab session '$SessionName'..."
colab stop -s $SessionName
Write-Host "Done! Trained adapter saved in outputs/nazrul-lora"

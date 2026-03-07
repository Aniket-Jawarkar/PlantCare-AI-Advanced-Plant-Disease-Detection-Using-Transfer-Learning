# run_pipeline.ps1
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
Write-Host "Starting PlantCare-AI ML Pipeline Automation..." -ForegroundColor Green

Write-Host "`n[1/5] Running Data Preprocessing..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\1_Data_Collection_and_Preprocessing"
python preprocessing.py

Write-Host "`n[2/5] Running Model Building..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\2_Model_Building"
python build_model.py

Write-Host "`n[3/5] Running Model Training..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\3_Model_Training"
python train_model.py

Write-Host "`n[4/5] Running Model Evaluation..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\4_Model_Evaluation_and_Testing"
python evaluate_model.py

Write-Host "`n[5/5] Launching the Web Application..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\5_Application_Building"

# Give the app a moment to start, then open the browser
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:5000"
} | Out-Null

python app.py

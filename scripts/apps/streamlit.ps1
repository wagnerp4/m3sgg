# VidSgg Streamlit App Launcher
# This script starts the Streamlit web application for VidSgg

Write-Host "Starting VidSgg Streamlit Application..." -ForegroundColor Green
Write-Host "=================================="

# Check if streamlit is installed
try {
    streamlit --version
} catch {
    Write-Host "Error: Streamlit is not installed or not available in PATH" -ForegroundColor Red
    Write-Host "Please install it with: pip install streamlit" -ForegroundColor Yellow
    exit 1
}

# Check if streamlit.py exists in scripts/apps directory
if (-not (Test-Path "scripts/apps/streamlit.py")) {
    Write-Host "Error: streamlit.py not found in scripts/apps directory" -ForegroundColor Red
    exit 1
}

Write-Host "Launching Streamlit app on http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Start streamlit
streamlit run scripts/apps/streamlit.py

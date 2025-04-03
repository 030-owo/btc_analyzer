@echo off
chcp 65001
cls
echo ====================================
echo    啟動加密貨幣分析服務
echo ====================================

:: 獲取本機 IP 地址
echo 正在獲取網絡信息...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| find "IPv4"') do (
    set ip=%%a
    goto :found_ip
)
:found_ip
set ip=%ip:~1%

:: 啟動 Flask 應用
echo [1/1] 正在啟動 Flask 應用程序...
start "Flask App" cmd /k "python app.py"

:: 等待 Flask 應用啟動
echo 等待服務啟動...
timeout /t 3

echo.
echo ====================================
echo    服務已成功啟動！
echo ====================================
echo.
echo 請注意：
echo 1. 在同一個網絡下的設備可以通過以下地址訪問：
echo    http://%ip%:5000
echo 2. 手機必須連接到同一個 WiFi 網絡
echo 3. 關閉此窗口不會停止服務運行
echo.
echo 按任意鍵關閉此窗口...
pause > nul 
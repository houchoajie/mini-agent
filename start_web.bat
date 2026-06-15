@echo off
echo ====================================
echo   Mini Agent Web Server 启动脚本
echo ====================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未找到 Python
    echo 请先安装 Python 3.13 或更高版本
    pause
    exit /b 1
)

echo ✅ Python 版本检查通过
echo.

REM 安装依赖
echo 📦 正在安装依赖...
pip install fastapi uvicorn openai python-dotenv PyMuPDF python-docx fpdf2 -q
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit /b 1
)

echo ✅ 依赖安装完成
echo.

REM 检查 .env 文件
if not exist .env (
    echo ⚠️  警告：.env 文件不存在
    echo 请复制 .env.example 为 .env 并配置 API Key
    echo.
    echo 即将创建 .env 文件...
    copy .env.example .env >nul
    echo ✅ 已创建 .env 文件，请编辑并填入你的 API Key
    pause
    exit /b 1
)

echo ✅ .env 文件存在
echo.

REM 启动 Web 服务
echo 🚀 正在启动 Web 服务...
echo 📍 访问地址: http://127.0.0.1:8002
echo.

REM 使用 conda 的 Python（venv 没有安装 fastapi）
D:\install\jetbrains\anaconda\python.exe web_server.py

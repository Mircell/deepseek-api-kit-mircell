@echo off
:: تغییر مسیر به پوشه‌ی خود فایل (تا مسیرهای نسبی درست کار کنند)
cd /d "%~dp0"

:: فعال‌سازی محیط مجازی (با call تا متغیرها حفظ شوند)
call ".venv\Scripts\activate.bat"

:: اجرای سرور Uvicorn برای VS Code Chat Proxy
uvicorn vscode_chat.main:app --host 127.0.0.1 --port 8001

:: در صورت بروز خطا، پنجره بسته نمی‌شود تا پیام خطا را ببینید
pause

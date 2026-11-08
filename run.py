"""Start the local application: .venv/bin/python run.py."""
import os
from pathlib import Path
import socket
import threading
import time
import webbrowser


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    os.environ.setdefault("U2NET_HOME", str(root / "models/rembg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(root / ".cache/numba"))
    port = int(os.environ.get("AVTO_PORT", "7860"))
    url = f"http://127.0.0.1:{port}"
    with socket.socket() as check:
        if check.connect_ex(("127.0.0.1", port)) == 0:
            print(f"Порт {port} уже занят. Если AVTO запущен, открой {url}.")
            print("Для другого порта: AVTO_PORT=7861 .venv/bin/python run.py")
            raise SystemExit(1)

    if os.environ.get("AVTO_NO_BROWSER") != "1":
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()
    print(f"\nAVTO — оценка автомобиля по фото\nОткрой {url}\nДля остановки нажми Ctrl+C.\n", flush=True)
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, reload=False, log_level="info")

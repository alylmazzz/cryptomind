#!/usr/bin/env python3
"""
AGI Trader Web Dashboard başlatıcı.

  python serve.py                 # http://127.0.0.1:8000
  python serve.py --port 8080 --host 0.0.0.0

GÜVENLİK: Dashboard varsayılan olarak yalnızca yerel makineye (127.0.0.1) açılır.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _port_free(host: str, port: int) -> bool:
    """Port bağlanabilir (boş) mı? uvicorn ile AYNI davranış için SO_REUSEADDR YOK
    (Windows'ta SO_REUSEADDR kullanımdaki portu 'boş' gösterip yanıltır)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host not in ("", "0.0.0.0") else "127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_free_port(host: str, start: int, tries: int = 25) -> int:
    """start'tan itibaren ilk BOŞ portu döndür. 8000 WSL/başka uygulama tarafından
    tutuluyorsa otomatik 8001, 8002… dener — böylece başlatıcı her zaman çalışır."""
    for offset in range(tries):
        cand = start + offset
        if _port_free(host, cand):
            return cand
    # hiçbiri boş değilse OS'a seçtir (port 0 = otomatik boş port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host if host not in ("", "0.0.0.0") else "127.0.0.1", 0))
        free = s.getsockname()[1]
    return free


def _open_browser_when_ready(host: str, port: int):
    """Sunucu portu dinlemeye başlayınca tarayıcıyı (tercihen Chrome) aç."""
    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{target}:{port}"
    # port hazır olana kadar bekle (en fazla ~60 sn)
    for _ in range(120):
        try:
            with socket.create_connection((target, port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    time.sleep(0.6)  # uvicorn route'ları bağlasın

    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for cp in chrome_paths:
        if os.path.exists(cp):
            try:
                subprocess.Popen([cp, "--new-window", url])
                print(f"  [✓] Chrome açıldı: {url}")
                return
            except Exception:
                pass
    try:
        webbrowser.open(url)
        print(f"  [✓] Varsayılan tarayıcı açıldı: {url}")
    except Exception:
        print(f"  [!] Tarayıcı otomatik açılamadı. Elle açın: {url}")


def main():
    p = argparse.ArgumentParser(description="AGI Trader Dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", default=None)
    p.add_argument("--no-open", action="store_true", help="Tarayıcıyı otomatik açma")
    args = p.parse_args()

    import uvicorn
    from agi_trader.config import load_config
    from agi_trader.server import create_app

    # İstenen port doluysa (ör. 8000'i WSL tutuyorsa) otomatik boş porta geç
    port = args.port
    if not _port_free(args.host, port):
        new_port = _find_free_port(args.host, port + 1)
        print(f"  [!] Port {port} dolu (baska uygulama/WSL kullaniyor) "
              f"-> otomatik {new_port} portuna geciliyor.", flush=True)
        port = new_port

    app = create_app(load_config(args.config))
    print(f"\n  AGI Trader Dashboard -> http://{args.host}:{port}\n", flush=True)

    if not args.no_open:
        threading.Thread(target=_open_browser_when_ready, args=(args.host, port),
                         daemon=True).start()

    try:
        uvicorn.run(app, host=args.host, port=port, log_level="warning")
    except OSError as e:
        # nadir yarış durumu: port seçildikten sonra kapıldıysa bir kez daha dene
        fallback = _find_free_port(args.host, port + 1)
        print(f"  [!] {port} bağlanamadı ({e}); {fallback} deneniyor…")
        if not args.no_open:
            threading.Thread(target=_open_browser_when_ready, args=(args.host, fallback),
                             daemon=True).start()
        uvicorn.run(app, host=args.host, port=fallback, log_level="warning")


if __name__ == "__main__":
    main()

"""
Otopilot / borsa bağlantısı API'si — `/account/trading/*`

Halka açık panel (`/api/*`) GET-only ve salt-okunur KALIR. Bu uçlar hesap
uygulamasının parçasıdır: oturum + CSRF + hız sınırı ister ve nginx'te
`/cryptomind/account/` konumundan yayımlanır.

Uçlar
  GET  /account/trading/catalog      borsalar, anahtar durumu, koşucu özeti, sunucu kapıları
  GET  /account/trading/state        bir borsanın koşucu durumu (pozisyon, işlem, karar izi)
  GET  /account/trading/readiness    canlıya geçiş kanıtı (paper self-test)
  POST /account/trading/start        koşucuyu kur + başlat (paper | testnet | live)
  POST /account/trading/stop
  POST /account/trading/close_all
  POST /account/trading/resume       HALT'tan elle çıkış
  POST /account/trading/params       strateji/zincir parametrelerini güncelle
  POST /account/trading/reset        paper defterini sıfırla
  POST /account/trading/remove       koşucuyu kaldır (açık pozisyon yoksa)

CANLI MOD ŞARTLARI (hepsi birden, biri eksikse 403 ve eksik listesi):
  1. kasa hazır + oturum + CSRF
  2. bu borsada İŞLEM kapsamlı anahtar (para çekme KAPALI doğrulanmış)
  3. sunucu operatör kapısı: execution.mode=live + allow_live + CRYPTOMIND_LIVE_CONFIRM=EVET
  4. kullanıcının onay cümlesi: "CANLI İŞLEMİ ONAYLIYORUM"
  5. paper kanıtı: bu stratejide ≥ N sanal işlem ve net > 0
  6. anahtar izinleri o an yeniden doğrulanır
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Body, Cookie, Header, Query, Request
from fastapi.responses import JSONResponse

from ..auto import live_runner as LR
from ..risk.live_guard import audit, live_enabled
from ..strategies import video_scalp as VS
from . import auth as A
from . import secure_keys as V


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?"))


# ccxt `urls.test` varlığına göre ölçüldü (2026-09); sınıfları çalışma anında
# kurmak 10 borsa × büyük describe() sözlüğü = gereksiz RSS.
_SANDBOX_TABLE = {"binance": True, "bybit": True, "okx": True, "kraken": False,
                  "coinbase": False, "kucoin": False, "gateio": None, "bitget": False,
                  "mexc": False, "htx": False}


def _sandbox_supported(ccxt_id: str) -> Optional[bool]:
    return _SANDBOX_TABLE.get(ccxt_id)


def _state_file(output_dir: str, uid: int, ex: str) -> Optional[Dict]:
    p = LR.state_path(output_dir, uid, ex)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_trading_router(registry: LR.RunnerRegistry, output_dir: str = "runs",
                          server_config=None) -> APIRouter:
    r = APIRouter()
    _sandbox_cache: Dict[str, Optional[bool]] = {}

    def _session(tok: Optional[str]) -> Optional[Dict]:
        return A.read_session(tok)

    def _guard(req: Request, tok, csrf_cookie, csrf_header):
        u = _session(tok)
        if not u:
            return None, JSONResponse(status_code=401,
                                      content={"error": "oturum yok veya süresi dolmuş"})
        if not A.check_csrf(csrf_cookie, csrf_header):
            return None, JSONResponse(status_code=403, content={"error": "CSRF jetonu geçersiz"})
        if not A.allow_api(f"u{u['id']}"):
            return None, JSONResponse(status_code=429,
                                      content={"error": "hız sınırı — dakikada 10 istek"})
        return u, None

    def _runner_or_404(uid: int, ex: str):
        run = registry.get(uid, ex)
        if run is None:
            return None, JSONResponse(status_code=404,
                                      content={"error": f"{ex} için kurulu koşucu yok"})
        return run, None

    # ------------------------------------------------------------ katalog
    @r.get("/account/trading/catalog")
    def catalog(cm_session: Optional[str] = Cookie(default=None)):
        from ..providers import exchanges
        u = _session(cm_session)
        ok, why = V.vault_available()
        le = live_enabled(server_config) if server_config is not None else \
            {"live": False, "missing": ["sunucu konfigürasyonu yok"], "note": "KAĞIT MOD"}
        saved = {k["provider"] for k in (V.list_secrets(u["id"], output_dir) if u else [])}
        summary = registry.summary(u["id"]) if u else {}
        out = []
        for p in exchanges():
            if p.ccxt_id not in _sandbox_cache:
                _sandbox_cache[p.ccxt_id] = _sandbox_supported(p.ccxt_id)
            has = p.id in saved
            out.append({
                "id": p.id, "name": p.name, "ccxt_id": p.ccxt_id,
                "fields": [{"key": f.key, "label": f.label, "secret": f.secret}
                           for f in p.fields],
                "passphrase": any(f.key.endswith("_PASSWORD") for f in p.fields),
                "signup_url": p.signup_url, "note": p.note,
                "sandbox": _sandbox_cache[p.ccxt_id],
                "has_key": has,
                "scope": (V.key_scope(u["id"], p.id, output_dir) if (u and has) else None),
                "runner": summary.get(p.id),
                "saved_state": bool(u and LR.state_path(output_dir, u["id"], p.id).exists()),
            })
        return {"logged_in": bool(u), "email": (u or {}).get("email"),
                "vault_ready": ok, "vault_note": why,
                "live_gate": le, "confirm_phrase": LR.CONFIRM_PHRASE,
                "strategy": VS.describe(),
                "defaults": LR.RunnerConfig().to_dict(),
                "exchanges": out}

    @r.get("/account/trading/state")
    def state(exchange: str = Query(default="binance"),
              cm_session: Optional[str] = Cookie(default=None)):
        u = _session(cm_session)
        if not u:
            return JSONResponse(status_code=401, content={"error": "oturum yok"})
        run = registry.get(u["id"], exchange)
        if run is None:
            saved = _state_file(output_dir, u["id"], exchange)
            return {"configured": False, "exchange": exchange,
                    "saved": bool(saved),
                    "saved_mode": (saved or {}).get("config", {}).get("mode"),
                    "saved_positions": len((saved or {}).get("positions", []))}
        return run.full_state()

    @r.get("/account/trading/readiness")
    def readiness(exchange: str = Query(default="binance"),
                  cm_session: Optional[str] = Cookie(default=None)):
        u = _session(cm_session)
        if not u:
            return JSONResponse(status_code=401, content={"error": "oturum yok"})
        run = registry.get(u["id"], exchange)
        if run is not None:
            return run.readiness()
        saved = _state_file(output_dir, u["id"], exchange) or {}
        h = saved.get("paper_history", [])
        cfg = LR.RunnerConfig.from_dict(saved.get("config"))
        net = round(sum(x.get("net_pnl", 0.0) for x in h), 2)
        return {"ok": (not cfg.require_paper_proof) or (len(h) >= cfg.paper_proof_trades and net > 0),
                "paper_trades": len(h), "paper_net": net,
                "required_trades": cfg.paper_proof_trades, "required": cfg.require_paper_proof,
                "missing": ([] if len(h) >= cfg.paper_proof_trades else
                            [f"paper işlem {len(h)}/{cfg.paper_proof_trades}"])
                           + ([] if net > 0 else [f"paper net {net:+.2f} USDT ≤ 0"])}

    # ------------------------------------------------------------ başlat
    @r.post("/account/trading/start")
    def start(req: Request, body: Dict = Body(...),
              cm_session: Optional[str] = Cookie(default=None),
              cm_csrf: Optional[str] = Cookie(default=None),
              x_csrf_token: Optional[str] = Header(default=None)):
        u, err = _guard(req, cm_session, cm_csrf, x_csrf_token)
        if err:
            return err
        ok, why = V.vault_available()
        if not ok:
            return JSONResponse(status_code=503, content={"error": why})
        ex = str(body.get("exchange") or "binance").strip().lower()
        mode = str(body.get("mode") or "paper").strip().lower()
        if mode not in ("paper", "testnet", "live"):
            return JSONResponse(status_code=400, content={"error": "mode paper|testnet|live"})
        from ..providers import exchanges
        prov = next((p for p in exchanges() if p.id == ex), None)
        if prov is None:
            return JSONResponse(status_code=400, content={"error": f"bilinmeyen borsa: {ex}"})

        cfg = LR.RunnerConfig.from_dict({**(body.get("config") or {}),
                                         "exchange_id": ex, "mode": mode})

        creds = None
        if mode in ("testnet", "live"):
            creds = V.exchange_creds(u["id"], ex, output_dir)
            if not creds:
                return JSONResponse(status_code=400, content={
                    "error": f"{prov.name} için kayıtlı anahtar yok — önce İŞLEM kapsamlı "
                             f"anahtar kaydedin"})
            scope = V.key_scope(u["id"], ex, output_dir)
            if scope != V.SCOPE_TRADE:
                return JSONResponse(status_code=403, content={
                    "error": f"{prov.name} anahtarı SADECE-OKUMA kapsamında — otopilot için "
                             f"'İŞLEM' kapsamıyla yeniden kaydedin"})

        blockers = []
        if mode == "live":
            le = live_enabled(server_config) if server_config is not None else \
                {"live": False, "missing": ["sunucu konfigürasyonu yok"]}
            if not le["live"]:
                blockers.append("sunucu operatör kapısı kapalı: " + ", ".join(le["missing"]))
            if str(body.get("confirm_phrase") or "").strip() != LR.CONFIRM_PHRASE:
                blockers.append(f"onay cümlesi gerekli: \"{LR.CONFIRM_PHRASE}\"")
            rd = readiness(ex, cm_session)
            if isinstance(rd, dict) and not rd.get("ok"):
                blockers.append("paper kanıtı eksik: " + ", ".join(rd.get("missing", [])))
            if not blockers:
                perm = V.exchange_permissions(prov.ccxt_id, creds["apiKey"], creds["secret"],
                                              creds.get("password"), require_trade=True)
                if not perm.get("ok"):
                    blockers.append("anahtar yeniden doğrulanamadı: " + perm.get("reason", "?"))
            if blockers:
                audit("LIVE_START_BLOCKED", {"user": u["email"], "exchange": ex,
                                             "blockers": blockers}, output_dir)
                return JSONResponse(status_code=403, content={"error": "canlı mod açılamadı",
                                                              "blockers": blockers})

        prev = registry.get(u["id"], ex)
        saved = _state_file(output_dir, u["id"], ex)
        if prev is not None:
            prev.save()
            saved = _state_file(output_dir, u["id"], ex) or saved
            if prev.positions and prev.cfg.mode != mode:
                return JSONResponse(status_code=409, content={
                    "error": f"{prev.cfg.mode} modunda {len(prev.positions)} açık pozisyon var — "
                             f"mod değiştirmeden önce kapatın (close_all)"})
        restore = None
        if saved:
            if (saved.get("config") or {}).get("mode") == mode:
                restore = saved                      # aynı mod: defter devam eder
            else:
                restore = {"paper_history": saved.get("paper_history", [])}   # kanıt taşınır
        try:
            run = registry.create(u["id"], cfg, creds=creds, restore=restore)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"koşucu kurulamadı: "
                                                                   f"{type(e).__name__}: {e}"})
        res = run.start()
        audit("TRADING_START", {"user": u["email"], "exchange": ex, "mode": mode,
                                "symbols": cfg.symbols, "capital": cfg.capital_usdt,
                                "max_order": cfg.max_order_usdt}, output_dir)
        return {"ok": bool(res.get("ok")), **res, "state": run.full_state()}

    # ------------------------------------------------------------ diğer
    def _simple(action: str):
        def handler(req: Request, body: Dict = Body(default={}),
                    cm_session: Optional[str] = Cookie(default=None),
                    cm_csrf: Optional[str] = Cookie(default=None),
                    x_csrf_token: Optional[str] = Header(default=None)):
            u, err = _guard(req, cm_session, cm_csrf, x_csrf_token)
            if err:
                return err
            ex = str(body.get("exchange") or "binance").strip().lower()
            run, err = _runner_or_404(u["id"], ex)
            if err:
                return err
            if action == "stop":
                out = run.stop("kullanıcı")
            elif action == "close_all":
                out = run.close_all("MANUEL")
            elif action == "resume":
                out = run.resume()
            elif action == "params":
                out = run.update_params(body.get("params"), body.get("chain"))
            elif action == "reset":
                out = run.reset_paper()
            elif action == "remove":
                if run.positions:
                    return JSONResponse(status_code=409, content={
                        "error": "açık pozisyon varken kaldırılamaz — önce close_all"})
                registry.remove(u["id"], ex)
                try:
                    LR.state_path(output_dir, u["id"], ex).unlink()
                except Exception:
                    pass
                out = {"ok": True}
            else:
                out = {"ok": False}
            audit(f"TRADING_{action.upper()}", {"user": u["email"], "exchange": ex,
                                                "ok": bool(out.get("ok"))}, output_dir)
            if action == "remove":
                return out
            return {**out, "state": run.full_state()}
        return handler

    for act in ("stop", "close_all", "resume", "params", "reset", "remove"):
        r.add_api_route(f"/account/trading/{act}", _simple(act), methods=["POST"])

    return r

"""
Hesap ve anahtar yönetimi API'si (FAZ 7).

Halka açık panel (`public_api.py`) SALT-OKUNUR ve GET-only kalır. Anahtar
yönetimi AYRI bir uygulamadır ve nginx'te ayrı bir konumda yayımlanır
(`/cryptomind/account/`), böylece panelin saldırı yüzeyi büyümez.

Uçlar
  GET  /account/status        kasa durumu + oturum (anahtar İÇERMEZ)
  GET  /account/providers     sağlayıcı kataloğu (ücretsiz/ücretli, kayıt linki)
  POST /account/login         oturum aç (hız sınırlı)
  POST /account/logout
  POST /account/register      İLK kullanıcı (bootstrap) veya yönetici
  GET  /account/keys          kayıtlı anahtarların MASKELİ listesi
  POST /account/keys          anahtar kaydet (borsa ise izin doğrulaması yapılır)
  POST /account/keys/delete   anahtar sil
  POST /account/keys/test     bağlantı testi (anahtarı ifşa etmeden)

GÜVENLİK SÖZLEŞMESİ
  • Hiçbir uç düz metin anahtar DÖNDÜRMEZ — yalnız "••••1234".
  • Tüm POST'lar CSRF jetonu ister (çift gönderim) ve oturum gerektirir.
  • Borsa anahtarı SADECE-OKUMA değilse KAYDEDİLMEZ.
  • Her yazma işlemi `runs/audit.log`'a (anahtar değeri OLMADAN) yazılır.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from fastapi import Body, Cookie, FastAPI, Header, Request
from fastapi.responses import JSONResponse

from ..risk.live_guard import audit
from . import auth as A
from . import secure_keys as V


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?"))


def create_account_app(config=None, output_dir: str = "runs",
                       secure_cookies: bool = True,
                       cookie_path: str = A.COOKIE_PATH) -> FastAPI:
    from .safe_json import SafeJSONResponse
    app = FastAPI(title="CryptoMind Account API", docs_url=None, redoc_url=None, default_response_class=SafeJSONResponse)

    # ------------------------------------------------------------- yardımcılar
    def _session(tok: Optional[str]) -> Optional[Dict]:
        return A.read_session(tok)

    def _guard(req: Request, tok: Optional[str], csrf_cookie: Optional[str],
               csrf_header: Optional[str], need_csrf: bool = True):
        """Oturum + CSRF + hız sınırı. Hata varsa JSONResponse döner."""
        u = _session(tok)
        if not u:
            return None, JSONResponse(status_code=401,
                                      content={"error": "oturum yok veya süresi dolmuş"})
        if need_csrf and not A.check_csrf(csrf_cookie, csrf_header):
            return None, JSONResponse(status_code=403,
                                      content={"error": "CSRF jetonu geçersiz"})
        if not A.allow_api(f"u{u['id']}"):
            return None, JSONResponse(status_code=429,
                                      content={"error": "hız sınırı — dakikada 10 istek"})
        return u, None

    # ------------------------------------------------------------------ durum
    @app.get("/account/status")
    def status(cm_session: Optional[str] = Cookie(default=None)):
        ok, why = V.vault_available()
        u = _session(cm_session)
        return {"vault_ready": ok, "vault_note": why,
                "logged_in": bool(u),
                "email": (u or {}).get("email"),
                "user_count": V.user_count(output_dir),
                "bootstrap_needed": V.user_count(output_dir) == 0}

    @app.get("/account/providers")
    def providers():
        """Sağlayıcı kataloğu — panel bu listeden otomatik üretilir."""
        from ..providers import cred_schema
        try:
            return {"providers": cred_schema()}
        except Exception as e:
            return {"providers": [], "error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------ giriş
    @app.post("/account/register")
    def register(req: Request, body: Dict = Body(...),
                 cm_session: Optional[str] = Cookie(default=None)):
        """İlk kullanıcı serbest (bootstrap); sonrakiler yalnız yönetici tarafından."""
        n = V.user_count(output_dir)
        if n > 0:
            u = _session(cm_session)
            if not u or u.get("role") != "admin":
                return JSONResponse(status_code=403,
                                    content={"error": "yeni kullanıcıyı yalnız yönetici ekler"})
        if not A.allow_login(_client_ip(req)):
            return JSONResponse(status_code=429, content={"error": "çok fazla deneme"})
        try:
            out = V.create_user(str(body.get("email", "")), str(body.get("password", "")),
                                output_dir, role="admin" if n == 0 else "user")
        except V.VaultError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        audit("ACCOUNT_REGISTER", {"email": out["email"], "role": out["role"]}, output_dir)
        return {"ok": True, **out}

    @app.post("/account/login")
    def login(req: Request, body: Dict = Body(...)):
        u, err = A.login(str(body.get("email", "")), str(body.get("password", "")),
                         _client_ip(req), output_dir)
        if not u:
            audit("ACCOUNT_LOGIN_FAIL", {"ip": _client_ip(req)}, output_dir)
            return JSONResponse(status_code=401, content={"error": err})
        try:
            tok = A.issue_session(u)
        except V.VaultError as e:
            return JSONResponse(status_code=503, content={"error": str(e)})
        csrf = A.issue_csrf()
        r = JSONResponse(content={"ok": True, "email": u["email"], "role": u["role"],
                                  "csrf": csrf})
        r.set_cookie(A.SESSION_COOKIE, tok,
                     **A.cookie_kwargs(secure_cookies, cookie_path))
        # CSRF çerezi JS tarafından OKUNABİLİR olmalı (başlığa konacak)
        ck = A.cookie_kwargs(secure_cookies, cookie_path)
        ck["httponly"] = False
        r.set_cookie(A.CSRF_COOKIE, csrf, **ck)
        audit("ACCOUNT_LOGIN", {"email": u["email"]}, output_dir)
        return r

    @app.post("/account/logout")
    def logout():
        r = JSONResponse(content={"ok": True})
        r.delete_cookie(A.SESSION_COOKIE, path=cookie_path)
        r.delete_cookie(A.CSRF_COOKIE, path=cookie_path)
        return r

    # ---------------------------------------------------------------- anahtar
    @app.get("/account/keys")
    def list_keys(cm_session: Optional[str] = Cookie(default=None)):
        u = _session(cm_session)
        if not u:
            return JSONResponse(status_code=401, content={"error": "oturum yok"})
        keys = V.list_secrets(u["id"], output_dir)
        # Borsa anahtarlarının KAPSAMI (read/trade) — değer değil, etiket
        scopes = {}
        for p in {k["provider"] for k in keys}:
            try:
                scopes[p] = V.key_scope(u["id"], p, output_dir)
            except Exception:
                scopes[p] = V.SCOPE_READ
        return {"keys": keys, "scopes": scopes}

    @app.post("/account/keys")
    def save_keys(req: Request, body: Dict = Body(...),
                  cm_session: Optional[str] = Cookie(default=None),
                  cm_csrf: Optional[str] = Cookie(default=None),
                  x_csrf_token: Optional[str] = Header(default=None)):
        u, err = _guard(req, cm_session, cm_csrf, x_csrf_token)
        if err:
            return err
        provider = str(body.get("provider", "")).strip()
        fields: Dict = body.get("fields") or {}
        if not provider or not isinstance(fields, dict) or not fields:
            return JSONResponse(status_code=400,
                                content={"error": "provider ve fields gerekli"})

        ok, why = V.vault_available()
        if not ok:
            return JSONResponse(status_code=503, content={"error": why})

        # --- borsa anahtarıysa ÖNCE izinleri doğrula ---
        # scope=read  → sadece-okuma zorunlu (emir izni olan anahtar RET)
        # scope=trade → otopilot için; emir izni gerekli, para çekme YİNE RET,
        #               kullanıcı 'para çekme kapalı' onay kutusunu işaretlemiş olmalı
        perm = None
        ex_id = str(body.get("exchange_id", "")).strip()
        scope = str(body.get("scope") or V.SCOPE_READ).strip().lower()
        if scope not in (V.SCOPE_READ, V.SCOPE_TRADE):
            return JSONResponse(status_code=400, content={"error": "scope read|trade olmalı"})
        if ex_id:
            api_key = str(fields.get("apiKey") or fields.get("api_key") or "")
            sec = str(fields.get("secret") or fields.get("apiSecret") or "")
            pwd = fields.get("password")
            if not api_key or not sec:
                return JSONResponse(status_code=400,
                                    content={"error": "borsa için apiKey ve secret gerekli"})
            if scope == V.SCOPE_TRADE and not bool(body.get("withdraw_disabled_ack")):
                return JSONResponse(status_code=400, content={
                    "error": "İşlem kapsamı için 'para çekme izni KAPALI' onayı gerekli "
                             "(withdraw_disabled_ack)"})
            perm = V.exchange_permissions(ex_id, api_key, sec, pwd,
                                          require_trade=(scope == V.SCOPE_TRADE))
            if not perm.get("ok"):
                audit("ACCOUNT_KEY_REJECTED",
                      {"user": u["email"], "provider": provider, "scope": scope,
                       "reason": perm.get("reason")}, output_dir)
                return JSONResponse(status_code=400,
                                    content={"error": perm.get("reason", "izin doğrulanamadı"),
                                             "permissions": perm})

        saved = []
        for f, v in fields.items():
            if v is None or str(v).strip() == "":
                continue
            try:
                saved.append(V.put_secret(u["id"], provider, str(f), str(v), output_dir))
            except V.VaultError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})
        if ex_id:
            try:
                V.set_meta(u["id"], provider, "scope", scope, output_dir)
            except V.VaultError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        # DİKKAT: audit'e anahtar DEĞERİ değil yalnız alan adları yazılır
        audit("ACCOUNT_KEY_SAVED",
              {"user": u["email"], "provider": provider, "scope": scope if ex_id else None,
               "fields": [s["field"] for s in saved]}, output_dir)
        return {"ok": True, "saved": saved, "permissions": perm,
                "scope": scope if ex_id else None}

    @app.post("/account/keys/delete")
    def delete_keys(req: Request, body: Dict = Body(...),
                    cm_session: Optional[str] = Cookie(default=None),
                    cm_csrf: Optional[str] = Cookie(default=None),
                    x_csrf_token: Optional[str] = Header(default=None)):
        u, err = _guard(req, cm_session, cm_csrf, x_csrf_token)
        if err:
            return err
        provider = str(body.get("provider", "")).strip()
        field = body.get("field")
        if not provider:
            return JSONResponse(status_code=400, content={"error": "provider gerekli"})
        n = V.delete_secret(u["id"], provider, field, output_dir)
        audit("ACCOUNT_KEY_DELETED",
              {"user": u["email"], "provider": provider, "field": field, "n": n}, output_dir)
        return {"ok": True, "deleted": n}

    @app.post("/account/keys/test")
    def test_key(req: Request, body: Dict = Body(...),
                 cm_session: Optional[str] = Cookie(default=None),
                 cm_csrf: Optional[str] = Cookie(default=None),
                 x_csrf_token: Optional[str] = Header(default=None)):
        """Kayıtlı anahtarla bağlantı testi — anahtar yanıtta İFŞA EDİLMEZ."""
        u, err = _guard(req, cm_session, cm_csrf, x_csrf_token)
        if err:
            return err
        provider = str(body.get("provider", "")).strip()
        ex_id = str(body.get("exchange_id", "")).strip()
        creds = V.get_provider_creds(u["id"], provider, output_dir)
        if not creds:
            return JSONResponse(status_code=404,
                                content={"error": "bu sağlayıcı için kayıtlı anahtar yok"})
        if ex_id:
            scope = V.key_scope(u["id"], provider, output_dir)
            perm = V.exchange_permissions(
                ex_id, creds.get("apiKey", creds.get("api_key", "")),
                creds.get("secret", creds.get("apiSecret", "")), creds.get("password"),
                require_trade=(scope == V.SCOPE_TRADE))
            return {"ok": bool(perm.get("ok")), "permissions": perm, "scope": scope}
        return {"ok": True, "note": "anahtar kayıtlı — sağlayıcıya özel test yok",
                "fields": sorted(creds.keys())}

    return app

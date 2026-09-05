# -*- coding: utf-8 -*-
"""FAZ 7 — kasa, kimlik doğrulama ve hesap API'si güvenlik testleri.

Bu katman GERÇEK API ANAHTARI tutar. Test edilmemiş bir şifreleme, şifrelenmemiş
bir anahtar demektir. Kritik davranışlar burada kilitlenir:
  • ana anahtar yoksa kasa KAPALI (fail closed)
  • şifre metni ana anahtar olmadan çözülemez
  • kullanıcılar birbirinin anahtarını çözemez
  • düz metin HTTP yanıtına ASLA sızmaz
  • CSRF'siz / oturumsuz yazma reddedilir
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_trader.server import secure_keys as V  # noqa: E402
from agi_trader.server import auth as A  # noqa: E402

pytest.importorskip("cryptography", reason="kasa cryptography olmadan çalışmaz")
MASTER = "test-master-key-" + "x" * 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, MASTER)
    A.reset_rate_limits()
    yield


@pytest.fixture
def vault(tmp_path):
    return str(tmp_path)


# ─────────────────────────── kasa durumu ────────────────────────────
def test_ana_anahtar_yoksa_kasa_kapali(monkeypatch, vault):
    monkeypatch.delenv(V.MASTER_ENV, raising=False)
    ok, why = V.vault_available()
    assert ok is False and V.MASTER_ENV in why


def test_kisa_ana_anahtar_reddedilir(monkeypatch):
    monkeypatch.setenv(V.MASTER_ENV, "kisa")
    with pytest.raises(V.VaultError):
        V.master_key()


def test_kasa_kapaliyken_anahtar_kaydedilmez(monkeypatch, vault):
    u = V.create_user("a@b.com", "parola-uzun-12345", vault)
    uid = V.verify_user("a@b.com", "parola-uzun-12345", vault)["id"]
    monkeypatch.delenv(V.MASTER_ENV, raising=False)
    with pytest.raises(V.VaultLocked):
        V.put_secret(uid, "binance", "apiKey", "SECRET", vault)


# ─────────────────────────── kullanıcı ──────────────────────────────
def test_kullanici_olustur_ve_dogrula(vault):
    V.create_user("x@y.com", "cok-guclu-parola-1", vault)
    assert V.verify_user("x@y.com", "cok-guclu-parola-1", vault) is not None
    assert V.verify_user("x@y.com", "yanlis-parola-123", vault) is None
    assert V.verify_user("yok@y.com", "cok-guclu-parola-1", vault) is None


def test_kisa_parola_reddedilir(vault):
    with pytest.raises(V.VaultError):
        V.create_user("k@y.com", "kisa123", vault)


def test_ayni_eposta_iki_kez_olmaz(vault):
    V.create_user("d@y.com", "cok-guclu-parola-1", vault)
    with pytest.raises(V.VaultError):
        V.create_user("d@y.com", "baska-parola-12345", vault)


def test_parola_duz_metin_saklanmaz(vault):
    import sqlite3
    V.create_user("p@y.com", "gizli-parola-12345", vault)
    raw = Path(V._db_path(vault)).read_bytes()
    assert b"gizli-parola-12345" not in raw


# ─────────────────────────── şifreleme ──────────────────────────────
def test_anahtar_sifreli_saklanir_ve_cozulur(vault):
    V.create_user("e@y.com", "cok-guclu-parola-1", vault)
    uid = V.verify_user("e@y.com", "cok-guclu-parola-1", vault)["id"]
    V.put_secret(uid, "binance", "apiKey", "AAAABBBBCCCC1234", vault)
    # düz metin diskte OLMAMALI
    raw = Path(V._db_path(vault)).read_bytes()
    assert b"AAAABBBBCCCC1234" not in raw
    # ama çözülebilmeli
    assert V.get_secret(uid, "binance", "apiKey", vault) == "AAAABBBBCCCC1234"


def test_maske_yalniz_son_4_hane(vault):
    V.create_user("m@y.com", "cok-guclu-parola-1", vault)
    uid = V.verify_user("m@y.com", "cok-guclu-parola-1", vault)["id"]
    V.put_secret(uid, "coingecko", "COINGECKO_API_KEY", "SUPERGIZLIANAHTAR9876", vault)
    lst = V.list_secrets(uid, vault)
    assert lst[0]["masked"] == "••••9876"
    assert "SUPERGIZLI" not in str(lst)


def test_baska_ana_anahtarla_cozulemez(vault, monkeypatch):
    V.create_user("z@y.com", "cok-guclu-parola-1", vault)
    uid = V.verify_user("z@y.com", "cok-guclu-parola-1", vault)["id"]
    V.put_secret(uid, "binance", "apiKey", "GIZLI-DEGER-1234", vault)
    monkeypatch.setenv(V.MASTER_ENV, "BASKA-ana-anahtar-" + "y" * 40)
    with pytest.raises(V.VaultError):
        V.get_secret(uid, "binance", "apiKey", vault)


def test_kullanicilar_birbirinin_anahtarini_cozemez(vault):
    V.create_user("u1@y.com", "cok-guclu-parola-1", vault)
    V.create_user("u2@y.com", "cok-guclu-parola-2", vault)
    a = V.verify_user("u1@y.com", "cok-guclu-parola-1", vault)["id"]
    b = V.verify_user("u2@y.com", "cok-guclu-parola-2", vault)["id"]
    V.put_secret(a, "binance", "apiKey", "KULLANICI-A-ANAHTARI", vault)
    # b kendi kasasında böyle bir kayıt görmez
    assert V.get_secret(b, "binance", "apiKey", vault) is None
    assert V.list_secrets(b, vault) == []


def test_silme(vault):
    V.create_user("s@y.com", "cok-guclu-parola-1", vault)
    uid = V.verify_user("s@y.com", "cok-guclu-parola-1", vault)["id"]
    V.put_secret(uid, "binance", "apiKey", "AAA1111", vault)
    V.put_secret(uid, "binance", "secret", "BBB2222", vault)
    assert V.delete_secret(uid, "binance", "apiKey", vault) == 1
    assert len(V.list_secrets(uid, vault)) == 1
    assert V.delete_secret(uid, "binance", None, vault) == 1
    assert V.list_secrets(uid, vault) == []


# ─────────────────────────── oturum ─────────────────────────────────
def test_oturum_jetonu_dogrulanir():
    tok = A.issue_session({"id": 7, "email": "a@b.com", "role": "admin"})
    s = A.read_session(tok)
    assert s and s["id"] == 7 and s["role"] == "admin"


def test_kurcalanmis_jeton_reddedilir():
    tok = A.issue_session({"id": 7, "email": "a@b.com", "role": "user"})
    raw, sig = tok.rsplit(".", 1)
    assert A.read_session(f"{raw}.{'A'*len(sig)}") is None
    assert A.read_session("saçma") is None
    assert A.read_session(None) is None


def test_suresi_dolmus_jeton_reddedilir(monkeypatch):
    monkeypatch.setattr(A, "SESSION_TTL", -1)
    tok = A.issue_session({"id": 1, "email": "a@b.com", "role": "user"})
    assert A.read_session(tok) is None


def test_ana_anahtar_degisince_oturumlar_gecersiz(monkeypatch):
    tok = A.issue_session({"id": 1, "email": "a@b.com", "role": "user"})
    monkeypatch.setenv(V.MASTER_ENV, "TAMAMEN-BASKA-anahtar-" + "q" * 40)
    assert A.read_session(tok) is None


# ─────────────────────────── CSRF + hız ─────────────────────────────
def test_csrf_cift_gonderim():
    t = A.issue_csrf()
    assert A.check_csrf(t, t) is True
    assert A.check_csrf(t, "baska") is False
    assert A.check_csrf(None, t) is False
    assert A.check_csrf(t, None) is False


def test_giris_hiz_siniri():
    for _ in range(A.LOGIN_MAX_PER_15MIN):
        assert A.allow_login("1.2.3.4") is True
    assert A.allow_login("1.2.3.4") is False
    assert A.allow_login("5.6.7.8") is True       # farklı IP etkilenmez


def test_api_hiz_siniri():
    for _ in range(A.API_MAX_PER_MIN):
        assert A.allow_api("u1") is True
    assert A.allow_api("u1") is False
    assert A.allow_api("u2") is True


def test_cerez_guvenlik_bayraklari():
    ck = A.cookie_kwargs(secure=True)
    assert ck["httponly"] is True and ck["secure"] is True
    assert ck["samesite"] == "strict" and ck["path"] == "/cryptomind"


# ─────────────────────── hesap API (uçtan uca) ──────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from agi_trader.server.account_api import create_account_app
    # test istemcisi rotaları kök altında çağırır → çerez yolu "/" olmalı
    app = create_account_app(output_dir=str(tmp_path), secure_cookies=False,
                             cookie_path="/")
    return TestClient(app), str(tmp_path)


def test_api_ilk_kullanici_admin_olur(client):
    c, _ = client
    r = c.post("/account/register", json={"email": "ilk@y.com",
                                          "password": "cok-guclu-parola-1"})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    # ikinci kayıt oturumsuz reddedilmeli
    r2 = c.post("/account/register", json={"email": "iki@y.com",
                                           "password": "cok-guclu-parola-2"})
    assert r2.status_code == 403


def test_api_oturumsuz_anahtar_yazilamaz(client):
    c, _ = client
    r = c.post("/account/keys", json={"provider": "binance", "fields": {"apiKey": "X"}})
    assert r.status_code == 401


def test_api_csrfsiz_yazma_reddedilir(client):
    c, _ = client
    c.post("/account/register", json={"email": "a@y.com", "password": "cok-guclu-parola-1"})
    c.post("/account/login", json={"email": "a@y.com", "password": "cok-guclu-parola-1"})
    # CSRF başlığı GÖNDERİLMİYOR
    r = c.post("/account/keys", json={"provider": "coingecko",
                                      "fields": {"COINGECKO_API_KEY": "ABC12345"}})
    assert r.status_code == 403


def test_api_anahtar_kaydet_ve_maskeli_listele(client):
    c, _ = client
    c.post("/account/register", json={"email": "b@y.com", "password": "cok-guclu-parola-1"})
    lr = c.post("/account/login", json={"email": "b@y.com", "password": "cok-guclu-parola-1"})
    csrf = lr.json()["csrf"]
    r = c.post("/account/keys",
               json={"provider": "coingecko", "fields": {"COINGECKO_API_KEY": "GIZLI-98761234"}},
               headers={A.CSRF_HEADER: csrf})
    assert r.status_code == 200, r.text
    body = r.text
    assert "GIZLI-9876" not in body                       # düz metin sızmadı
    lst = c.get("/account/keys").json()["keys"]
    assert lst[0]["masked"] == "••••1234"
    assert "GIZLI" not in str(lst)


def test_api_status_anahtar_icermez(client):
    c, _ = client
    c.post("/account/register", json={"email": "c@y.com", "password": "cok-guclu-parola-1"})
    lr = c.post("/account/login", json={"email": "c@y.com", "password": "cok-guclu-parola-1"})
    csrf = lr.json()["csrf"]
    c.post("/account/keys", json={"provider": "fred", "fields": {"FRED_API_KEY": "ANAHTAR777"}},
           headers={A.CSRF_HEADER: csrf})
    s = c.get("/account/status").text
    assert "ANAHTAR777" not in s and "fred" not in s.lower()


def test_api_borsa_anahtari_apikey_secret_ister(client):
    c, _ = client
    c.post("/account/register", json={"email": "d@y.com", "password": "cok-guclu-parola-1"})
    lr = c.post("/account/login", json={"email": "d@y.com", "password": "cok-guclu-parola-1"})
    csrf = lr.json()["csrf"]
    r = c.post("/account/keys",
               json={"provider": "binance", "exchange_id": "binance",
                     "fields": {"apiKey": "sadece-key"}},
               headers={A.CSRF_HEADER: csrf})
    assert r.status_code == 400 and "secret" in r.json()["error"]


def test_api_cikis_cerezleri_siler(client):
    c, _ = client
    c.post("/account/register", json={"email": "e@y.com", "password": "cok-guclu-parola-1"})
    c.post("/account/login", json={"email": "e@y.com", "password": "cok-guclu-parola-1"})
    c.post("/account/logout")
    assert c.get("/account/keys").status_code == 401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

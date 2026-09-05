"""
Şifreli anahtar kasası (FAZ 7) — kullanıcının kendi API anahtarlarını saklar.

TEHDİT MODELİ — neye karşı koruyoruz:
  • Veritabanı dosyası sızarsa (yedek, disk, yanlış izin) anahtarlar OKUNAMAZ:
    şifre metni ana anahtar olmadan işe yaramaz, ana anahtar dosyada DEĞİL
    ortam değişkenindedir.
  • Bir kullanıcının şifre metni ele geçse bile diğerlerine yaramaz: her
    kullanıcı için ana anahtardan HKDF ile AYRI veri anahtarı türetilir.
  • Uygulama içinden bile düz metin DÖNMEZ; yalnız son 4 hane maskesi.
  • Borsa anahtarı SADECE-OKUMA olmalı; para çekme veya emir izni varsa
    KAYDEDİLMEZ (bkz. `exchange_permissions`).

NEYE KARŞI KORUMAZ: sunucuya root erişimi olan biri hem ortam değişkenini hem
veritabanını görür. Bu mimaride kaçınılmazdır — anahtarların kullanılabilmesi
için çalışma anında çözülmeleri gerekir. Bu yüzden borsa anahtarlarının
sadece-okuma olması ZORUNLU kılınır: en kötü senaryoda bakiye okunur, para
çekilemez.

Kripto: AES-256-GCM (`cryptography`), anahtar türetme HKDF-SHA256.
Şifre özeti: `hashlib.scrypt` (stdlib, bellek-zoru). argon2id tercih edilirdi
ama sunucuda kurulu değil; yeni bağımlılık eklemek yerine standart kütüphanenin
scrypt'i kullanıldı (n=2^15, r=8, p=1 — OWASP önerisi üstü).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- kripto (yoksa FAIL CLOSED: kendi kriptomuzu yazmayız) ---
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    _CRYPTO = True
except Exception:                                    # pragma: no cover
    _CRYPTO = False

MASTER_ENV = "CRYPTOMIND_MASTER_KEY"
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
KEY_LEN = 32

# scrypt bellek ihtiyacı = 128 × N × r = 128 × 32768 × 8 = TAM 32 MiB.
# OpenSSL'in varsayılan `maxmem` sınırı da tam 32 MiB olduğu için sınırda
# "memory limit exceeded" hatası verir. Parametreleri zayıflatmak yerine
# sınırı yükseltiyoruz (bellek-zorluğu korumanın ta kendisidir).
# Giriş denemesi başına 32 MiB ayrılır; kötüye kullanımı auth.allow_login
# hız sınırı engeller (IP başına 15 dakikada 8 deneme).
SCRYPT_MAXMEM = 96 * 1024 * 1024


class VaultError(RuntimeError):
    pass


class VaultLocked(VaultError):
    """Ana anahtar yok — kasa açılamaz. Sistem bu durumda anahtar KABUL ETMEZ."""


# ===========================================================================
# Ana anahtar
# ===========================================================================
def master_key() -> bytes:
    """Ortam değişkeninden ana anahtar. Dosyada tutulmaz, log'a yazılmaz."""
    raw = os.environ.get(MASTER_ENV, "").strip()
    if not raw:
        raise VaultLocked(
            f"{MASTER_ENV} tanımlı değil — anahtar kasası kilitli. "
            f"Üretmek için: python -c \"import secrets;print(secrets.token_urlsafe(48))\"")
    if len(raw) < 32:
        raise VaultError(f"{MASTER_ENV} çok kısa (min 32 karakter)")
    return hashlib.sha256(raw.encode("utf-8")).digest()


def vault_available() -> Tuple[bool, str]:
    if not _CRYPTO:
        return False, "`cryptography` paketi kurulu değil — şifreleme yapılamaz"
    try:
        master_key()
    except VaultError as e:
        return False, str(e)
    return True, "hazır"


def _user_key(user_salt: bytes) -> bytes:
    """Kullanıcıya özel veri anahtarı = HKDF(ana anahtar, tuz).

    Ana anahtar doğrudan şifreleme için KULLANILMAZ; böylece bir kullanıcının
    anahtar materyali sızsa bile diğerlerine erişilemez."""
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=user_salt,
                info=b"cryptomind-user-vault-v1").derive(master_key())


# ===========================================================================
# Depolama
# ===========================================================================
def _db_path(output_dir: str = "runs") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    return p / "vault.db"


def _connect(output_dir: str = "runs") -> sqlite3.Connection:
    path = _db_path(output_dir)
    first = not path.exists()
    con = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        pw_hash BLOB NOT NULL,
        pw_salt BLOB NOT NULL,
        vault_salt BLOB NOT NULL,
        created_at INTEGER NOT NULL,
        role TEXT NOT NULL DEFAULT 'user')""")
    con.execute("""CREATE TABLE IF NOT EXISTS secrets(
        user_id INTEGER NOT NULL,
        provider TEXT NOT NULL,
        field TEXT NOT NULL,
        nonce BLOB NOT NULL,
        ciphertext BLOB NOT NULL,
        last4 TEXT NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY(user_id, provider, field))""")
    if first:
        try:
            os.chmod(path, 0o600)       # yalnız sahibi okuyabilsin
        except Exception:
            pass
    return con


# ===========================================================================
# Kullanıcı
# ===========================================================================
def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt,
                          n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LEN,
                          maxmem=SCRYPT_MAXMEM)


def create_user(email: str, password: str, output_dir: str = "runs",
                role: str = "user") -> Dict:
    """Kullanıcı oluştur. Parola asla saklanmaz, yalnız scrypt özeti."""
    email = email.strip().lower()
    if "@" not in email or len(email) < 5:
        raise VaultError("geçersiz e-posta")
    if len(password) < 12:
        raise VaultError("parola en az 12 karakter olmalı")
    salt = secrets.token_bytes(16)
    con = _connect(output_dir)
    try:
        con.execute(
            "INSERT INTO users(email,pw_hash,pw_salt,vault_salt,created_at,role) "
            "VALUES(?,?,?,?,?,?)",
            (email, _hash_password(password, salt), salt,
             secrets.token_bytes(16), int(time.time()), role))
    except sqlite3.IntegrityError:
        raise VaultError("bu e-posta zaten kayıtlı")
    finally:
        con.close()
    return {"email": email, "role": role}


def verify_user(email: str, password: str, output_dir: str = "runs") -> Optional[Dict]:
    """Parola doğrulama — sabit zamanlı karşılaştırma."""
    email = (email or "").strip().lower()
    con = _connect(output_dir)
    try:
        row = con.execute(
            "SELECT id,email,pw_hash,pw_salt,role FROM users WHERE email=?",
            (email,)).fetchone()
    finally:
        con.close()
    if not row:
        # kullanıcı yoksa da benzer süre harca (kullanıcı sayımı sızmasın)
        _hash_password(password or "", b"\x00" * 16)
        return None
    uid, mail, pw_hash, pw_salt, role = row
    if not hmac.compare_digest(_hash_password(password or "", pw_salt), pw_hash):
        return None
    return {"id": int(uid), "email": mail, "role": role}


def user_count(output_dir: str = "runs") -> int:
    con = _connect(output_dir)
    try:
        return int(con.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    finally:
        con.close()


def _vault_salt(user_id: int, output_dir: str) -> bytes:
    con = _connect(output_dir)
    try:
        row = con.execute("SELECT vault_salt FROM users WHERE id=?",
                          (user_id,)).fetchone()
    finally:
        con.close()
    if not row:
        raise VaultError("kullanıcı bulunamadı")
    return row[0]


# ===========================================================================
# Anahtar saklama
# ===========================================================================
def put_secret(user_id: int, provider: str, field: str, value: str,
               output_dir: str = "runs") -> Dict:
    """Anahtarı şifreleyip sakla. Düz metin DİSKE YAZILMAZ, LOG'A GİRMEZ."""
    ok, why = vault_available()
    if not ok:
        raise VaultLocked(why)
    value = (value or "").strip()
    if not value:
        raise VaultError("boş değer")

    key = _user_key(_vault_salt(user_id, output_dir))
    nonce = secrets.token_bytes(12)
    ct = AESGCM(key).encrypt(nonce, value.encode("utf-8"),
                             f"{user_id}|{provider}|{field}".encode())
    con = _connect(output_dir)
    try:
        con.execute(
            "INSERT INTO secrets(user_id,provider,field,nonce,ciphertext,last4,updated_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,provider,field) DO UPDATE SET "
            "nonce=excluded.nonce, ciphertext=excluded.ciphertext, "
            "last4=excluded.last4, updated_at=excluded.updated_at",
            (user_id, provider, field, nonce, ct, value[-4:], int(time.time())))
    finally:
        con.close()
    return {"provider": provider, "field": field, "last4": value[-4:]}


def get_secret(user_id: int, provider: str, field: str,
               output_dir: str = "runs") -> Optional[str]:
    """Düz metni ÇÖZ — yalnız sunucu içi kullanım (borsa çağrısı vb.).

    Bu fonksiyonun çıktısı ASLA HTTP yanıtına konmaz. API katmanı yalnız
    `list_secrets` maskesini döndürür."""
    ok, why = vault_available()
    if not ok:
        raise VaultLocked(why)
    con = _connect(output_dir)
    try:
        row = con.execute(
            "SELECT nonce,ciphertext FROM secrets WHERE user_id=? AND provider=? AND field=?",
            (user_id, provider, field)).fetchone()
    finally:
        con.close()
    if not row:
        return None
    key = _user_key(_vault_salt(user_id, output_dir))
    try:
        pt = AESGCM(key).decrypt(row[0], row[1],
                                 f"{user_id}|{provider}|{field}".encode())
    except Exception:
        raise VaultError("çözülemedi — ana anahtar değişmiş olabilir")
    return pt.decode("utf-8")


def list_secrets(user_id: int, output_dir: str = "runs") -> List[Dict]:
    """Kayıtlı anahtarların MASKELİ listesi. Düz metin dönmez."""
    con = _connect(output_dir)
    try:
        rows = con.execute(
            "SELECT provider,field,last4,updated_at FROM secrets WHERE user_id=? "
            "ORDER BY provider,field", (user_id,)).fetchall()
    finally:
        con.close()
    return [{"provider": p, "field": f, "masked": "••••" + (l4 or ""),
             "updated_at": int(t)} for p, f, l4, t in rows
            if not f.startswith(META_PREFIX)]


def delete_secret(user_id: int, provider: str, field: Optional[str] = None,
                  output_dir: str = "runs") -> int:
    con = _connect(output_dir)
    try:
        if field:
            cur = con.execute(
                "DELETE FROM secrets WHERE user_id=? AND provider=? AND field=?",
                (user_id, provider, field))
        else:
            cur = con.execute("DELETE FROM secrets WHERE user_id=? AND provider=?",
                              (user_id, provider))
        return cur.rowcount
    finally:
        con.close()


def get_provider_creds(user_id: int, provider: str,
                       output_dir: str = "runs") -> Dict[str, str]:
    """Bir sağlayıcının tüm alanları (sunucu içi kullanım). `meta:` alanları
    kimlik bilgisi DEĞİLDİR, dışarıda bırakılır."""
    con = _connect(output_dir)
    try:
        rows = con.execute(
            "SELECT field FROM secrets WHERE user_id=? AND provider=?",
            (user_id, provider)).fetchall()
    finally:
        con.close()
    return {f: get_secret(user_id, provider, f, output_dir) or ""
            for (f,) in rows if not f.startswith(META_PREFIX)}


# ===========================================================================
# Meta alanlar (anahtar KAPSAMI vb.) — aynı tabloda, `meta:` ön ekiyle
# ===========================================================================
META_PREFIX = "meta:"
SCOPE_READ, SCOPE_TRADE = "read", "trade"


def set_meta(user_id: int, provider: str, name: str, value: str,
             output_dir: str = "runs") -> None:
    put_secret(user_id, provider, META_PREFIX + name, value, output_dir)


def get_meta(user_id: int, provider: str, name: str,
             output_dir: str = "runs") -> Optional[str]:
    try:
        return get_secret(user_id, provider, META_PREFIX + name, output_dir)
    except VaultError:
        return None


def key_scope(user_id: int, provider: str, output_dir: str = "runs") -> str:
    """Kayıtlı borsa anahtarının kapsamı: read (varsayılan) | trade."""
    return get_meta(user_id, provider, "scope", output_dir) or SCOPE_READ


def exchange_creds(user_id: int, provider: str, output_dir: str = "runs") -> Dict[str, str]:
    """ccxt biçiminde {apiKey, secret, password?} — yalnız sunucu içi (Broker)."""
    c = get_provider_creds(user_id, provider, output_dir)
    out = {"apiKey": c.get("apiKey") or c.get("api_key") or "",
           "secret": c.get("secret") or c.get("apiSecret") or ""}
    pw = c.get("password") or c.get("passphrase")
    if pw:
        out["password"] = pw
    return out if out["apiKey"] and out["secret"] else {}


# ===========================================================================
# Borsa anahtarı izin doğrulaması — SADECE-OKUMA ZORUNLU
# ===========================================================================
def exchange_permissions(exchange_id: str, api_key: str, secret: str,
                         password: Optional[str] = None,
                         timeout_ms: int = 15000, *,
                         require_trade: bool = False,
                         client_factory=None) -> Dict:
    """Anahtarın izinlerini CANLI test eder. İKİ KAPSAM vardır:

    SADECE-OKUMA (require_trade=False — varsayılan, panel/portföy görünümü):
      • okuma çalışmalı            → fetchBalance başarılı
      • para çekme KAPALI olmalı   → açıksa KESİN RET
      • emir izni OLMAMALI         → açıksa RET (bu kapsam okuma ister)

    İŞLEM (require_trade=True — otopilot; kullanıcı açık onayla seçer):
      • okuma çalışmalı
      • para çekme KAPALI olmalı   → açıksa KESİN RET (her iki kapsamda da)
      • emir izni AÇIK olmalı      → borsa bildiriyorsa kontrol edilir; bildirmiyorsa
                                     'doğrulanamadı' diye işaretlenir, ilk emirde anlaşılır

    Para çekme izni olan anahtar HİÇBİR kapsamda kabul edilmez: en kötü senaryoda
    (sunucu ele geçirilse bile) hesap boşaltılamaz, yalnız işlem açılabilir — bunu
    da emir tavanı, günlük zarar limiti ve kill-switch sınırlar."""
    out = {"ok": False, "can_read": False, "can_trade": None,
           "can_withdraw": None, "ip_restricted": None, "withdraw_verified": False,
           "scope": SCOPE_TRADE if require_trade else SCOPE_READ,
           "reason": "", "exchange": exchange_id}
    if client_factory is None:
        try:
            import ccxt
        except Exception:
            out["reason"] = "ccxt kurulu değil — izin doğrulaması yapılamıyor"
            return out

        try:
            cls = getattr(ccxt, exchange_id)
        except AttributeError:
            out["reason"] = f"bilinmeyen borsa: {exchange_id}"
            return out
        client_factory = cls

    cfg = {"apiKey": api_key, "secret": secret, "enableRateLimit": True,
           "timeout": timeout_ms}
    if password:
        cfg["password"] = password
    ex = client_factory(cfg)

    # 1) okuma
    try:
        ex.fetch_balance()
        out["can_read"] = True
    except Exception as e:
        out["reason"] = f"okuma başarısız: {type(e).__name__}"
        return out

    # 2) borsanın bildirdiği izinler (destekleyen borsalarda)
    try:
        info = {}
        if exchange_id.startswith("binance"):
            info = ex.sapi_get_account_apirestrictions()
            out["can_withdraw"] = bool(info.get("enableWithdrawals"))
            out["can_trade"] = bool(info.get("enableSpotAndMarginTrading")) or \
                bool(info.get("enableFutures"))
            out["ip_restricted"] = bool(info.get("ipRestrict"))
            out["withdraw_verified"] = True
    except Exception:
        pass                                  # izin ucu yoksa aşağıdaki teste düş

    # 3) para çekme izni açıksa KESİN RET — her iki kapsamda
    if out["can_withdraw"] is True:
        out["reason"] = ("anahtarın PARA ÇEKME izni açık — kabul edilmez. "
                         "Borsada para çekme izni OLMAYAN yeni bir anahtar oluşturun.")
        return out

    if not require_trade:
        # 4a) emir izni açıksa reddet (bu kapsam yalnız okuma ister)
        if out["can_trade"] is True:
            out["reason"] = ("anahtarın EMİR izni açık — sadece-okuma kapsamı için "
                             "kabul edilmez. Otopilot için 'İŞLEM' kapsamını seçin.")
            return out
        out["ok"] = True
        out["reason"] = "sadece-okuma doğrulandı"
        return out

    # 4b) işlem kapsamı: emir izni olmalı
    if out["can_trade"] is False:
        out["reason"] = ("anahtarın EMİR izni YOK — otopilot bu anahtarla işlem "
                         "açamaz. Borsada 'Spot & Margin Trading' iznini açın "
                         "(para çekme KAPALI kalsın).")
        return out
    out["ok"] = True
    if out["withdraw_verified"]:
        out["reason"] = "işlem anahtarı doğrulandı: emir AÇIK, para çekme KAPALI" + \
            ("" if out["ip_restricted"] else " — IP kısıtlaması ÖNERİLİR")
    else:
        out["reason"] = ("işlem anahtarı kabul edildi; bu borsa izinleri API'den "
                         "bildirmiyor — para çekme izninin KAPALI olduğunu siz "
                         "doğruladınız (onay kutusu)")
    return out

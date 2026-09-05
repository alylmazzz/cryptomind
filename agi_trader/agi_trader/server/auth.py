"""
Oturum, CSRF ve hız sınırı (FAZ 7).

TASARIM KARARLARI ve NEDENLERİ

  • Oturum jetonu HMAC-imzalıdır, sunucuda oturum tablosu tutulmaz.
    İmza `CRYPTOMIND_MASTER_KEY`'den türetilir; ana anahtar değişirse tüm
    oturumlar geçersizleşir (istenen davranış).
  • Çerez `HttpOnly` (JavaScript okuyamaz → XSS ile çalınamaz),
    `Secure` (yalnız HTTPS), `SameSite=Strict` (CSRF'in birincil savunması).
  • SameSite'a rağmen ayrıca ÇİFT GÖNDERİM CSRF jetonu kullanılır: eski
    tarayıcılar ve olası alt alan adı senaryoları için ikinci katman.
  • Hız sınırı hem IP hem kullanıcı bazlı: parola deneme saldırısını yavaşlatır.
  • Başarısız girişte hata mesajı AYRIM YAPMAZ ("e-posta yok" vs "parola
    yanlış") — kullanıcı sayımını engeller.

Not: jeton süresi kısa (12 saat) tutulur; panelin kendisi salt-okunur olduğu
için uzun oturuma gerek yoktur.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from .secure_keys import master_key, verify_user, VaultError

SESSION_COOKIE = "cm_session"
CSRF_COOKIE = "cm_csrf"
CSRF_HEADER = "X-CSRF-Token"
SESSION_TTL = 12 * 3600

# hız sınırları
LOGIN_MAX_PER_15MIN = 8
API_MAX_PER_MIN = 10

_login_hits: Dict[str, Deque[float]] = defaultdict(deque)
_api_hits: Dict[str, Deque[float]] = defaultdict(deque)


# ===========================================================================
# Jeton
# ===========================================================================
def _sign(payload: bytes) -> str:
    return base64.urlsafe_b64encode(
        hmac.new(master_key(), payload, hashlib.sha256).digest()).decode().rstrip("=")


def issue_session(user: Dict) -> str:
    """user = {id, email, role} → imzalı jeton."""
    body = {"u": int(user["id"]), "e": user["email"], "r": user.get("role", "user"),
            "exp": int(time.time()) + SESSION_TTL,
            "n": secrets.token_urlsafe(8)}
    raw = base64.urlsafe_b64encode(
        json.dumps(body, separators=(",", ":")).encode()).decode().rstrip("=")
    return f"{raw}.{_sign(raw.encode())}"


def read_session(token: Optional[str]) -> Optional[Dict]:
    """Jetonu doğrula. İmza veya süre hatalıysa None."""
    if not token or "." not in token:
        return None
    raw, sig = token.rsplit(".", 1)
    try:
        expected = _sign(raw.encode())
    except VaultError:
        return None                       # ana anahtar yok → oturum yok
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        pad = "=" * (-len(raw) % 4)
        body = json.loads(base64.urlsafe_b64decode(raw + pad))
    except Exception:
        return None
    if int(body.get("exp", 0)) < time.time():
        return None
    return {"id": int(body["u"]), "email": body["e"], "role": body.get("r", "user")}


def issue_csrf() -> str:
    return secrets.token_urlsafe(24)


def check_csrf(cookie_token: Optional[str], header_token: Optional[str]) -> bool:
    """Çift gönderim: çerezdeki jeton ile başlıktaki eşleşmeli.

    Saldırgan başka bir siteden istek atabilir ama çerezi OKUYAMAZ, dolayısıyla
    doğru başlığı üretemez."""
    if not cookie_token or not header_token:
        return False
    return hmac.compare_digest(str(cookie_token), str(header_token))


# ===========================================================================
# Hız sınırı
# ===========================================================================
def _hit(store: Dict[str, Deque[float]], key: str, window: float, limit: int) -> bool:
    now = time.time()
    dq = store[key]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


def allow_login(ip: str) -> bool:
    return _hit(_login_hits, ip or "?", 900.0, LOGIN_MAX_PER_15MIN)


def allow_api(user_key: str) -> bool:
    return _hit(_api_hits, user_key or "?", 60.0, API_MAX_PER_MIN)


def reset_rate_limits() -> None:
    """Testler için."""
    _login_hits.clear()
    _api_hits.clear()


# ===========================================================================
# Giriş akışı
# ===========================================================================
def login(email: str, password: str, ip: str = "",
          output_dir: str = "runs") -> Tuple[Optional[Dict], str]:
    """(oturum_bilgisi, hata_mesajı). Hata mesajı AYRIM YAPMAZ."""
    if not allow_login(ip):
        return None, "çok fazla deneme — 15 dakika sonra tekrar deneyin"
    u = verify_user(email, password, output_dir)
    if not u:
        return None, "e-posta veya parola hatalı"
    return u, ""


COOKIE_PATH = "/cryptomind"          # üretimde nginx bu ön ek altında yayımlar


def cookie_kwargs(secure: bool = True, path: str = COOKIE_PATH) -> Dict:
    """FastAPI `set_cookie` için ortak güvenlik ayarları.

    `path` ÖNEMLİ: çerez yalnız bu ön ek altındaki isteklerde gönderilir.
    Üretimde uygulama `/cryptomind/account/...` altında yayımlandığı için
    varsayılan doğrudur; uygulamayı kök altında koşturan test/geliştirme
    ortamı `path="/"` geçmelidir (aksi hâlde çerez hiç geri gelmez ve her
    istek 401 döner)."""
    return {"httponly": True, "secure": secure, "samesite": "strict",
            "path": path, "max_age": SESSION_TTL}

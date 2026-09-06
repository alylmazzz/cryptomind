# -*- coding: utf-8 -*-
"""13. tur (2026-09-04) — YouTube video kurulumlarının mekanik çekirdeği.

21 videonun transkripti çıkarıldı; iddialar DEĞİL, kurulumlar alındı. Bu testler:
  · her özelliğin tanımını kilitler (FVG, ters FVG, manipülasyon mumu, emir bloğu, aralık geri dönüşü)
  · canlı veride bulunan İKİ mantık hatasının regresyonunu tutar:
      (1) ters FVG filtreli bölgeyle aranınca mantıksal olarak ASLA ateşleyemiyordu (1815 pencerede 0)
      (2) emir bloğu yalnız MSS BARINDA aranıyordu; kurulum "MSS oldu, sonra bloğa dönüldü" der
  · seans öncülünün (ölçülmüş) boyutu küçülttüğünü, KAPATMADIĞINI doğrular
  · kayıt defterlerinin (SLEEVE_TR/lifecycle/kütüphane/aciliyet/çıkış modu) eksiksiz olduğunu doğrular
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agi_trader.research import library as LIB  # noqa: E402
from agi_trader.strategies import committee as CM  # noqa: E402
from agi_trader.strategies import sleeves_fast as SF  # noqa: E402
from agi_trader.strategies import sleeves_video as SV  # noqa: E402
from agi_trader.strategies.lifecycle import DEFAULT_SLEEVES  # noqa: E402

NY = 1_788_530_400.0      # 2026-09-04 14:00 UTC → NY_AM penceresi (13-16)
ASIA = 1_788_483_600.0    # 2026-09-04 01:00 UTC → ASYA penceresi (00-03)


def _df(rows, start="2026-09-04 13:00", freq="min"):
    """rows: (open, high, low, close, volume) listesi."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def _flat(n=300, px=100.0, vol=1.0):
    return [(px, px * 1.001, px * 0.999, px, vol) for _ in range(n)]


def _wobble(n=280, px=100.0, seed=11):
    """Hafif dalgalı taban — düz seri RSI'ı kilitler ve ilk atak RSI'ı eşiğin üstüne fırlatır."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = px + float(rng.normal(0, 0.05))
        out.append((c, c * 1.0015, c * 0.9985, c, 1.0))
    return out


def _feat(df):
    p = CM.CommitteeParams()
    f = CM.fast_features(df, p)
    if not f.get("ok"):
        return f
    f["price"] = float(df["close"].iloc[-1])
    return SV.video_features(df, f)


# ═══════════════════════════ özellik tanımları ═══════════════════════════
def test_fvg_bogaz_bosluk_tespiti_ve_bolge():
    rows = _flat(280)
    # 3 mumluk boğa boşluğu: mum1 high=100.1 · mum3 low=101.0 → bölge [100.1, 101.0]
    rows += [(100.0, 100.1, 99.9, 100.0, 1.0), (100.0, 101.5, 100.0, 101.4, 5.0), (101.4, 101.6, 101.0, 101.3, 2.0)]
    rows += [(101.3, 101.3, 100.6, 100.7, 1.0)]        # fiyat boşluğa geri döndü (içeride)
    f = _feat(_df(rows))
    z = f["fvg_bull"]
    assert z is not None and abs(z["lo"] - 100.1) < 1e-6 and abs(z["hi"] - 101.0) < 1e-6
    assert abs(z["mid"] - 100.55) < 1e-6 and f["in_bull_fvg"] is True


def test_ifvg_filtreli_bolgeyle_asla_atesleyemez_regresyon():
    """Ters FVG 'aşılmamış' filtreli bölgeyle aranırsa 'hiç aşılmamış' + 'şimdi aşıldı' aynı anda
    doğru olamaz → canlı veride 1815 pencerede 0 ateşleme. Ham (filtresiz) bölge kullanılmalı."""
    rows = _flat(280)
    # ayı boşluğu: mum1 low=99.9 · mum3 high=98.5 → bölge [98.5, 99.9]; sonra fiyat üstüne kapanır
    rows += [(100.0, 100.1, 99.9, 100.0, 1.0), (100.0, 100.0, 98.2, 98.4, 5.0), (98.4, 98.5, 98.0, 98.3, 2.0)]
    rows += [(98.3, 99.0, 98.3, 98.9, 1.0), (98.9, 99.6, 98.8, 99.5, 1.0), (99.5, 99.8, 99.4, 99.7, 1.0)]
    rows += [(99.7, 100.4, 99.7, 100.3, 3.0)]          # gövde 99,9'un ÜSTÜNDE kapandı → ters çevrildi
    f = _feat(_df(rows))
    assert f["fvg_bear_raw"] is not None, "ham ayı bölgesi bulunmalı"
    assert f["ifvg_up"] is True, "ters FVG ateşlemeli (regresyon: filtreli bölge kullanılırsa False kalır)"
    # ham bölge aşılmış olduğu için FİLTRELİ bölge burada None'dır — eski kod tam bu yüzden hiç ateşlemedi
    assert f["fvg_bear"] is None


def test_manipulasyon_mumu_tam_tanim():
    """Funded Brothers'ın verdiği tam tanım: low[t] < low[t-1] VE close[t] > high[t-1]."""
    rows = _flat(280)
    rows += [(100.0, 100.5, 99.5, 100.0, 1.0)]                     # önceki mum: low 99,5 · high 100,5
    rows += [(100.0, 100.9, 99.2, 100.7, 4.0)]                     # dibi aldı (99,2) ve tepenin üstünde kapattı
    f = _feat(_df(rows))
    assert f["manip_bull"] is True
    # yalnız dibi alıp tepenin ALTINDA kapatırsa geçersiz
    rows2 = _flat(280) + [(100.0, 100.5, 99.5, 100.0, 1.0), (100.0, 100.4, 99.2, 100.2, 4.0)]
    assert _feat(_df(rows2))["manip_bull"] is False


def test_emir_blogu_mss_sonrasi_geri_donuste_bulunur_regresyon():
    """Kurulum 'MSS oldu, SONRA bloğa geri dönüldü' der. Blok yalnız MSS barında aranırsa fiyat o an
    bloğun içinde olmadığı için neredeyse hiç ateşlemez (canlı veride 1815 pencerede 1)."""
    rows = _flat(260)
    rows += [(100.0, 100.8, 100.0, 100.7, 1.0)] * 3                # swing tepe ~100,8
    rows += [(100.7, 100.7, 100.1, 100.2, 1.0)] * 3                # geri çekilme
    rows += [(100.2, 100.3, 99.8, 99.9, 1.0)]                      # SON KIRMIZI MUM = emir bloğu [99,8 · 100,3]
    rows += [(99.9, 101.6, 99.9, 101.5, 6.0)]                      # atak: swing tepenin üstüne gövde kapanışı = MSS
    rows += [(101.5, 101.6, 101.0, 101.1, 1.0)] * 2                # devam
    rows += [(101.1, 101.2, 100.2, 100.25, 1.0)]                   # bloğa GERİ DÖNÜŞ (100,25 ∈ [99,8 · 100,3])
    f = _feat(_df(rows))
    assert f["mss_recent"] is True and f["ob_lo"] is not None
    assert f["ob_lo"] <= f["price"] <= f["ob_hi"] and f["in_ob"] is True
    assert f["mss_bar_ago"] is not None and f["mss_bar_ago"] >= 1


def test_aralik_geri_donusu_kapanis_bazli_fitil_sayilmaz():
    """Broken Lollipop / Data Trader: dışarı ÇIKIŞ ve içeri DÖNÜŞ mum KAPANIŞIYLA olmalı."""
    base = "2026-09-04 08:00"                                       # önceki 4h blok (08-12) → aralık burada oluşur
    rows = [(100.0, 100.6, 99.4, 100.0, 1.0) for _ in range(240)]   # 08:00-12:00 aralık ≈ [99,4 · 100,6]
    rows += [(100.0, 100.2, 99.6, 99.9, 1.0) for _ in range(3)]     # 12:00 sonrası (yeni blok)
    rows += [(99.9, 99.95, 99.0, 99.1, 2.0)]                        # aralığın ALTINA kapanış
    rows += [(99.1, 99.8, 99.05, 99.7, 2.0)]                        # aralığa GERİ kapanış → geçerli
    f = _feat(_df(rows, start=base))
    assert f["htf_range"] is not None and f["range_reclaim_up"] is True
    # yalnız FİTİL aşağı sarkarsa (kapanış hep içeride) geçersiz olmalı
    rows2 = [(100.0, 100.6, 99.4, 100.0, 1.0) for _ in range(240)]
    rows2 += [(100.0, 100.2, 99.6, 99.9, 1.0) for _ in range(3)]
    rows2 += [(99.9, 99.95, 98.9, 99.6, 2.0)]                       # fitil aralığın altında, KAPANIŞ içeride
    rows2 += [(99.6, 99.8, 99.5, 99.7, 2.0)]
    assert _feat(_df(rows2, start=base))["range_reclaim_up"] is False


def test_hacim_profili_poc_ve_deger_alani():
    """POC hacmin yoğunlaştığı fiyatta olmalı.

    NOT (2026-09-05): bu test TAMAMEN DÜZ seri kuruyordu (kapanış sabit) →
    `fast_features` σ hesaplayamayıp `ok=False` dönüyor, `_feat` erken çıkıyor ve
    sözlükte `vprofile` hiç oluşmuyordu (KeyError). Kusur üretim kodunda değil,
    testin kurduğu girdideydi: oynaklığı sıfır bir seri boru hattının meşru olarak
    reddettiği bir girdidir. Fiyat artık hafifçe dalgalanır (σ > 0), hacim yine
    101 kümesinde yoğundur — testin ASIL iddiası korunur."""
    rng = np.random.default_rng(7)
    rows = [(x, x * 1.001, x * 0.999, x, 1.0)
            for x in 100.0 + rng.normal(0, 0.02, 120)]              # taban: az hacim
    rows += [(x, x * 1.001, x * 0.999, x, 9.0)
             for x in 101.0 + rng.normal(0, 0.02, 120)]             # hacim 101'de yoğun → POC ≈ 101
    f = _feat(_df(rows))
    assert f.get("ok") is not False, f"özellik hattı girdiyi reddetti: {f.get('reason')}"
    vp = f["vprofile"]
    assert vp is not None and 100.7 <= vp["poc"] <= 101.3
    assert vp["val"] <= vp["poc"] <= vp["vah"] and vp["width_pct"] > 0


def test_stokastik_ve_bollinger_alanlari():
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    rows = [(p, p * 1.002, p * 0.998, p, 1.0) for p in px]
    f = _feat(_df(rows))
    assert f["stoch_k"] is not None and 0.0 <= f["stoch_k"] <= 100.0
    assert f["stoch_d"] is not None and f["bb_lower"] is not None and f["bb_mid"] is not None
    assert f["bb_width_pct_v"] is not None and f["bb_width_pct_v"] > 0


# ═══════════════════════════ tetikleyiciler + seans ═══════════════════════════
def _fvg_ready_df():
    rows = _wobble(280)
    rows += [(100.0, 100.1, 99.9, 100.0, 1.0), (100.0, 101.5, 100.0, 101.4, 5.0), (101.4, 101.6, 101.0, 101.3, 2.0)]
    rows += [(101.3, 101.3, 101.0, 101.05, 1.0), (101.05, 101.1, 100.8, 100.85, 1.0),
             (100.85, 100.9, 100.55, 100.6, 1.0)]                   # kademeli geri çekilme (RSI nötrlenir)
    rows += [(100.6, 101.0, 100.58, 100.9, 1.0)]                    # boşluk içinde + yeşil bar
    return _df(rows)


def test_fvg_sleeve_atesler_ve_kaynak_kunyesi_var():
    f = _feat(_fvg_ready_df())
    f["trend_up"] = True
    got = SV.fire_video_sleeves(f, SV.ALL_VIDEO_SLEEVES, CM.CommitteeParams(), False, NY)
    kinds = {g["kind"] for g in got}
    assert "fvg_fill" in kinds
    g = next(g for g in got if g["kind"] == "fvg_fill")
    assert g["direction"] == "LONG" and g["stop_hint"] < f["price"] and "FVG" in g["note"]
    assert SV.SOURCES["fvg_fill"]["evidence"] in SV.describe()["evidence_legend"]


def test_seans_onculu_boyutu_kucultur_kapatmaz():
    """Ölçüm: seans dışı ham kenar t −6,3; NY_AM t +3,4. Öncül boyutu küçültür ama sinyali SUSTURMAZ
    (susturursa ölçüm birikmez ve kapı kendini güncelleyemez)."""
    f = _feat(_fvg_ready_df()); f["trend_up"] = True
    ny = SV.fire_video_sleeves(f, SV.ALL_VIDEO_SLEEVES, CM.CommitteeParams(), False, NY)
    f2 = _feat(_fvg_ready_df()); f2["trend_up"] = True
    asia = SV.fire_video_sleeves(f2, SV.ALL_VIDEO_SLEEVES, CM.CommitteeParams(), False, ASIA)
    g_ny = next(g for g in ny if g["kind"] == "fvg_fill")
    g_as = next(g for g in asia if g["kind"] == "fvg_fill")
    assert g_as["size"] < g_ny["size"] and g_as["size"] > 0.0
    assert SV.session_size_mult(NY) == 1.0 and SV.session_size_mult(ASIA) < 1.0
    assert SV.killzone_of(NY) == "NY_AM" and SV.killzone_of(ASIA) == "ASYA"
    assert "seans çarpanı" in g_as["note"]


def test_video_sleeveleri_yalniz_izinli_listede_atesler():
    f = _feat(_fvg_ready_df()); f["trend_up"] = True
    got = SV.fire_video_sleeves(f, [k for k in SV.ALL_VIDEO_SLEEVES if k != "fvg_fill"],
                                CM.CommitteeParams(), False, NY)
    assert "fvg_fill" not in {g["kind"] for g in got}
    assert SV.fire_video_sleeves({"ok": False}, SV.ALL_VIDEO_SLEEVES, CM.CommitteeParams(), False, NY) == []


def test_komite_video_sleevei_degerlendirir():
    """Uçtan uca: komite video sleeve'ini aday olarak görür ve fiş üretir (kanıt kapıları ayrıca uygular)."""
    from test_committee import _slow
    df = _fvg_ready_df()
    price = float(df["close"].iloc[-1])
    ctx = dict(symbol="BTC/USDT", price=price, df=df, slow=_slow(price, regime="TREND YUKARI"),
               qual_cell=None, book={"spread_bps": 2.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6},
               fees={"maker_bps": 0.0, "taker_bps": 5.0}, open_positions={}, max_open=3,
               exposure_room=700.0, capital=1000.0, max_order=200.0,
               notional_fn=lambda stop_pct: min(200.0, 10.0 / (stop_pct / 100.0)),
               p_win=0.5, halted=False, paused_reason=None, daily_loss_left_pct=5.0,
               market_type="spot", now=NY)
    v = CM.evaluate(ctx, CM.CommitteeParams(), {})
    d = v.to_dict()
    video_kinds = set(SV.ALL_VIDEO_SLEEVES)
    seen = {c["kind"] for c in (d.get("competition") or [])} | ({d.get("trigger")} if d.get("trigger") else set())
    seen |= {t["kind"] for t in (d.get("silenced") or [])}
    assert seen & video_kinds, f"komite hiçbir video sleeve'i görmedi: {seen}"


# ═══════════════════════════ kayıt defterleri ═══════════════════════════
def test_kayit_defterleri_eksiksiz():
    for k in SV.ALL_VIDEO_SLEEVES:
        assert k in SF.SLEEVE_TR, k
        assert k in SF.SLEEVE_EXIT_MODE and SF.SLEEVE_EXIT_MODE[k] in ("FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK")
        assert k in SF.SLEEVE_TIME_STOP_MIN and 30 <= SF.SLEEVE_TIME_STOP_MIN[k] <= 480
        assert k in SF.SLEEVE_URGENCY and 0 <= SF.SLEEVE_URGENCY[k] <= 3
        assert k in DEFAULT_SLEEVES, f"{k} lifecycle kaydında yok → canlı kapısı belirsiz kalır"
        assert k in SF.ALL_SLEEVES
        assert any(k in v for v in SF.REGIME_SLEEVES.values()), f"{k} hiçbir rejimde açık değil"
        assert k in SV.SOURCES and SV.SOURCES[k]["channels"] and SV.SOURCES[k]["evidence"]


def test_kutuphane_video_kayitlari():
    vids = [r for r in LIB.LIBRARY if r.get("video_sourced")]
    assert len(vids) == 10 and len(LIB.LIBRARY) == 260
    assert {r["impl_key"] for r in vids} == set(SV.ALL_VIDEO_SLEEVES)
    for r in vids:
        assert r["impl_kind"] == "sleeve" and r["status"] == "IMPLEMENTED" and r["pipeline_stage"] == "PAPER"
    assert set(SV.ALL_VIDEO_SLEEVES) <= set(LIB.sleeves_implemented())


def test_kunye_durustluk_alanlari():
    d = SV.describe()
    assert len(d["sources"]) == 10 and len(d["not_implemented"]) >= 4
    assert any("kapalı kaynak" in x["item"].lower() or "Kapalı kaynak" in x["item"] for x in d["not_implemented"])
    assert any("%10" in x["item"] for x in d["not_implemented"]), "günlük %10 hedefi dürüstlük notunda olmalı"
    m = d["measured"]
    # 7 günlük ASIL ölçüm: hiçbir pencere pozitif değil. Çürütülen 1 günlük ölçüm de KAYITLI durmalı
    # (bulgunun nasıl değiştiği silinmez) — bu testin amacı "iyi görünen sayıyı saklama" refleksini engellemek.
    assert m["all"]["t"] < 0 and m["all"]["n"] >= 5000
    assert all(v["t"] < 2.0 for v in m["by_session"].values()), "hiçbir seans anlamlı pozitif değil"
    assert d["verdict"] == "SHADOW" and m["verdict"] == "SHADOW"
    sup = d["measured_superseded"]
    assert sup["by_session"]["NY_AM"]["t"] > 3.0 and "çürüttü" in sup["note"]
    for row in d["sources"]:
        assert row["evidence"] in d["evidence_legend"]
    # hiçbir kaynak "doğrulanmış canlı kanıt" iddia etmiyor olmalı
    assert not any(row["evidence"] in ("AUDITED", "VERIFIED") for row in d["sources"])


def test_video_sleeveleri_golgede_dogar_canliya_cikamaz():
    """Ölçüm negatif çıktı → kurulumlar SHADOW. Sinyal üretirler ama simülatör (paper) dahil emir VERMEZLER.
    Bu testin kilidi: birisi ölçüm yapmadan bunları PAPER'a alırsa test kırılır."""
    from agi_trader.strategies.lifecycle import Lifecycle, SHADOW_SLEEVES
    lc = Lifecycle()
    assert set(SHADOW_SLEEVES) == set(SV.ALL_VIDEO_SLEEVES)
    for k in SV.ALL_VIDEO_SLEEVES:
        assert lc.stage(k) == "SHADOW", k
        assert lc.can_trade(k, "paper") is False, f"{k} kanıtsız hâlde paper'da emir veremez"
        assert lc.can_trade(k, "live") is False
    assert lc.stage("dip") == "PAPER" and lc.can_trade("dip", "paper") is True     # mevcutlar etkilenmedi


def test_komite_golgedeki_sleevei_susturur_ve_kaydeder():
    """Lifecycle SHADOW → komite `allowed` listesinden çıkarır; aday `silenced` olarak kaydedilir
    (kaçırılan-fırsat motoru gölgeler). Böylece ölçüm emirsiz sürer."""
    from agi_trader.strategies.lifecycle import Lifecycle
    from test_committee import _slow
    df = _fvg_ready_df()
    price = float(df["close"].iloc[-1])
    ctx = dict(symbol="BTC/USDT", price=price, df=df, slow=_slow(price, regime="TREND YUKARI"),
               qual_cell=None, book={"spread_bps": 2.0, "bid_depth_usd": 1e6, "ask_depth_usd": 1e6},
               fees={"maker_bps": 0.0, "taker_bps": 5.0}, open_positions={}, max_open=3,
               exposure_room=700.0, capital=1000.0, max_order=200.0,
               notional_fn=lambda stop_pct: min(200.0, 10.0 / (stop_pct / 100.0)),
               p_win=0.5, halted=False, paused_reason=None, daily_loss_left_pct=5.0,
               market_type="spot", now=NY, lifecycle=Lifecycle(), mode="paper")
    v = CM.evaluate(ctx, CM.CommitteeParams(), {})
    d = v.to_dict()
    assert d.get("trigger") not in SV.ALL_VIDEO_SLEEVES, "gölgedeki sleeve işlem tetikleyicisi olamaz"
    assert not (set(SV.ALL_VIDEO_SLEEVES) & {c["kind"] for c in (d.get("competition") or [])})
    # susturma SEBEBİ doğru olmalı: "gölgede (kanıt yok)" ile "bu rejimde kapalı" farklı şeyler
    sil = {t["kind"]: t.get("gate") for t in (d.get("silenced") or [])}
    vid_sil = {k: g for k, g in sil.items() if k in SV.ALL_VIDEO_SLEEVES}
    assert vid_sil, "gölgedeki aday kaydedilmeli (ölçüm emirsiz sürer)"
    assert all(g == "YAŞAM_DÖNGÜSÜ" for g in vid_sil.values()), vid_sil
    from agi_trader.learn.missed import GATE_TR
    assert "YAŞAM_DÖNGÜSÜ" in GATE_TR and "gölge" in GATE_TR["YAŞAM_DÖNGÜSÜ"]


def test_video_ozellikleri_bozuk_veride_patlamaz():
    for bad in (_df(_flat(5)), _df([(1.0, 1.0, 1.0, 1.0, 0.0)] * 40)):
        p = CM.CommitteeParams()
        f = CM.fast_features(bad, p)
        f = SV.video_features(bad, f if isinstance(f, dict) else {})
        assert isinstance(f, dict)
    df = _df(_flat(300))
    df.loc[df.index[-1], "volume"] = float("nan")
    f = _feat(df)
    assert isinstance(f, dict)

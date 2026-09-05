"""
AGI Trader — Çok katmanlı, çok ajanlı, açıklanabilir kripto karar-destek motoru.

Spesifikasyon (prompt.txt) doğrultusunda tasarlanmıştır:
  Veri toplama -> Teknik analiz (120+ indikatör) -> Formasyon/SMC tespiti ->
  Sentiment/Twitter -> AI ensemble -> Risk yönetimi -> Ağırlıklı karar motoru
  (>= %90 güven) -> Execution (paper-trade + kill-switch) -> Self-improvement journal.

Güvenlik ilkesi: Sistem VARSAYILAN olarak paper-trading modunda çalışır.
Gerçek emir göndermek için config'de execution.mode = "live" ve
ayrı bir onay (allow_live: true) GEREKİR. Hiçbir API anahtarı koda gömülmez.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]

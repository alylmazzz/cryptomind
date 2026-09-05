/**
 * CANLI İŞLEM GÜNLÜĞÜ — sayfanın üst-sağ köşesinde, yapışkan.
 *
 * Kaynak: /api/simulator/feed (10 sn). İlk işlem burada görünür: durum bandı
 * ("İLK İŞLEM BEKLENİYOR" → "MAKER EMRİ BEKLİYOR" → "AÇIK POZİSYON"), tetikleyiciye
 * en yakın pariteler ve eksik şartları, kısa vadeli piyasa olasılıkları (ölçülmüş
 * modellerden), olay akışı ve kapanan işlemler. Panel sayı ÜRETMEZ.
 */
import { useState, useEffect, useCallback } from 'react'
import { ScrollText, Hourglass, TrendingUp, Radar } from 'lucide-react'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E',
            surface:'#1A2240', bg:'#0A0E1A', cyan:'#22D3EE', violet:'#A78BFA' }
const usd = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) < 0 ? '−' : ''}$${Math.abs(Number(x)).toLocaleString('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const hhmm = (ts: any) => ts ? new Date(ts * 1000).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'
const ago = (ts: any) => ts ? `${Math.max(0, Math.round(Date.now() / 1000 - ts))} sn önce` : '—'
const REG_TR: Record<string, string> = { 'TREND YUKARI': 'trend ↑', 'TREND AŞAĞI': 'trend ↓', 'RANGE / YATAY': 'yatay', 'VOLATİL': 'volatil' }

export default function TradeLog() {
  const [f, setF] = useState<any>(null)
  const [down, setDown] = useState(false)
  const pull = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/simulator/feed`)
      const d = await r.json().catch(() => null)
      if (r.ok && d) { setF(d); setDown(false) } else setDown(true)
    } catch { setDown(true) }
  }, [])
  useEffect(() => { pull(); const iv = setInterval(pull, 10000); return () => clearInterval(iv) }, [pull])

  const st = f?.status
  const state = st?.state || 'NOT_READY'
  const col = state === 'OPEN' ? C.neon : state === 'PENDING' ? C.warn : state === 'HALT' ? C.danger : C.cyan
  const o = f?.outlook
  const cons = o?.consensus_4h || {}
  const nTot = (cons.LONG || 0) + (cons.SHORT || 0) + (cons.FLAT || 0)

  return (
    <div className="cm-tradelog" style={{ padding: 8 }}>
      {/* ── DURUM BANDI ── */}
      <div style={{ border: `1px solid ${col}66`, background: `${col}12`, borderRadius: 8, padding: '8px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <ScrollText size={13} color={col} />
          <span className="cm-title" style={{ color: C.text }}>CANLI İŞLEM GÜNLÜĞÜ</span>
          {down && <span style={{ marginLeft: 'auto', fontSize: 8, color: C.danger }}>API yanıt vermiyor</span>}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
          {state === 'WAITING_FIRST' && <Hourglass size={14} color={col} className="anim-pulse" />}
          <span style={{ fontSize: 13, fontWeight: 900, letterSpacing: 0.8, color: col }}>
            {st?.label || 'SİMÜLATÖR KURULUYOR…'}
          </span>
        </div>
        {st && (
          <div className="mono" style={{ fontSize: 9, color: C.muted, marginTop: 4, lineHeight: 1.5 }}>
            döngü #{st.cycle} · {st.symbols} parite · her {st.loop_sec} sn · son karar {ago(st.last_cycle_ts)}<br />
            {f?.venue?.exchange_id?.toUpperCase()} · maker %{((f?.venue?.maker_bps || 0) / 100).toFixed(2)} / taker %{((f?.venue?.taker_bps || 0) / 100).toFixed(2)}
            · özsermaye {usd(st.equity)} · net {usd(st.net_pnl)} · komisyon {usd(st.fees_paid)}
          </div>
        )}
        {f?.risk_mode && (
          <div style={{ marginTop: 6, fontSize: 9.5, padding: '4px 7px', borderRadius: 5,
                        border: `1px solid ${(f.risk_mode.level === 0 ? C.neon : f.risk_mode.level === 1 ? C.warn : C.danger)}66`,
                        color: f.risk_mode.level === 0 ? C.neon : f.risk_mode.level === 1 ? C.warn : C.danger }}>
            <b>Piyasa modu: {f.risk_mode.label}</b>
            {f.risk_mode.reasons?.length > 0 && <span style={{ color: C.muted }}> · {f.risk_mode.reasons.join(' · ')}</span>}
            {f.risk_mode.breadth != null && <span style={{ color: C.muted }}> · genişlik {f.risk_mode.breadth >= 0 ? '+' : ''}{f.risk_mode.breadth}</span>}
            {f.tiers && <span style={{ color: C.muted }}> · {f.tiers.heavy} ağır + {f.tiers.light} hafif parite</span>}
          </div>
        )}
        {state === 'WAITING_FIRST' && (
          <div style={{ fontSize: 9, color: '#8892B0', marginTop: 4, lineHeight: 1.5 }}>
            İlk aday tetikleyiciyi geçince 12 rol oylar, maker limit emri atılır ve burada görünür.
            Sinyal uydurulmaz; koşul oluşmadan işlem yoktur.
          </div>
        )}
      </div>

      {/* ── ÖZET ŞERİDİ: PORTFÖY MODU · EN İYİ EYLEM · SAĞLIK ── */}
      {f?.portfolio_mode && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 6 }}>
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: '5px 8px' }}>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: 0.6 }}>PORTFÖY MODU</div>
            <div style={{ fontSize: 11, fontWeight: 800, color: f.portfolio_mode.mode === 'RISK_ON' ? C.neon : f.portfolio_mode.mode === 'SELECTIVE' ? C.cyan : f.portfolio_mode.mode === 'DEFENSIVE' ? C.warn : C.danger }}>
              {f.portfolio_mode.mode}</div>
            <div style={{ fontSize: 8, color: C.muted }}>{(f.portfolio_mode.reasons || []).slice(0, 2).join(' · ') || 'genişlik/korelasyon/haber normal'}</div>
          </div>
          <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: '5px 8px' }}>
            <div style={{ fontSize: 8, color: C.muted, letterSpacing: 0.6 }}>ŞİMDİ EN İYİ EYLEM</div>
            <div style={{ fontSize: 11, fontWeight: 800, color: f.best_action?.action === 'BUY' ? C.neon : f.best_action?.action === 'CASH' || f.best_action?.action === 'HALT' ? C.danger : C.text }}>
              {f.best_action?.action}{f.best_action?.symbol ? ` · ${f.best_action.symbol.replace('/USDT', '')}` : ''}</div>
            <div style={{ fontSize: 8, color: C.muted }}>{String(f.best_action?.why || '').slice(0, 70)}</div>
          </div>
          {f.missed && <div style={{ gridColumn: '1 / -1', fontSize: 8, color: C.muted }}>
            👁️ kaçırılan kazanç <b style={{ color: C.warn }}>{f.missed.n_missed}</b> · doğru kaçınma <b style={{ color: C.neon }}>{f.missed.n_avoided}</b> · gölgede {f.missed.n_open}{f.missed.top_gate ? ` · en çok: ${f.missed.top_gate}` : ''}
            {f.missed.last?.attribution?.verdict ? ` · son: ${f.missed.last.symbol} ${f.missed.last.attribution.verdict}` : ''}
          </div>}
          {f.resource && <div style={{ gridColumn: '1 / -1', fontSize: 8, color: C.muted }}>
            kaynak {f.resource.state} · RSS {f.resource.rss_mb ?? '—'} MB / tavan {f.resource.cap_mb} · döngü {f.resource.cycle_sec ?? '—'} sn{f.resource.errors_10m ? ` · ⚠ ${f.resource.errors_10m} hata/10dk` : ''}
            {f.resource.store ? ` · depo isabet ${f.resource.store.hits}/${f.resource.store.hits + f.resource.store.misses}` : ''}
          </div>}
        </div>
      )}

      {/* ── TOP 3 FIRSAT KARTI ── */}
      {f?.top_opportunities?.length > 0 && (
        <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
          <div className="cm-title" style={{ marginBottom: 4 }}>TOP FIRSATLAR (net EV sırası)</div>
          {f.top_opportunities.map((o: any, i: number) => (
            <div key={i} style={{ fontSize: 9, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                <b style={{ color: o.allowed ? C.neon : C.muted }}>{o.symbol}</b>
                <span style={{ color: C.muted }}>{o.sleeve}{o.tier === 'light' ? ' · hafif' : ''} · {o.exit_mode}</span>
                <span className="mono" style={{ color: (o.ev_pct || 0) > 0 ? C.neon : C.danger }}>EV %{o.ev_pct}</span>
                <span className="mono" style={{ color: C.muted }}>p {Math.round((o.p_win || 0) * 100)}%</span>
              </div>
              <div className="mono" style={{ fontSize: 8, color: C.muted }}>
                giriş bölgesi {o.entry?.entry_low?.toPrecision?.(6) ?? '—'}–{o.entry?.entry_high?.toPrecision?.(6) ?? '—'} · optimal {o.entry?.optimal?.price?.toPrecision?.(6) ?? '—'} ({o.entry?.optimal?.order_type}) · max chase {o.entry?.max_chase?.toPrecision?.(6) ?? '—'}
                <br />stop {o.plan?.stop?.toPrecision?.(6)} (%{o.plan?.stop_pct}) · hedef {o.plan?.target?.toPrecision?.(6)} (%{o.plan?.target_pct}) · R/R {o.plan?.rr} · kâr {usd(o.expected_profit_usdt)} · komisyon {usd(o.fee_usdt, 3)} · risk {usd(o.max_loss_usdt)}
              </div>
              <div style={{ fontSize: 8, color: o.allowed ? C.neon : C.warn }}>{o.result}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── BEKLEYEN / AÇIK ── */}
      {f?.pending?.map((p: any) => (
        <div key={p.symbol} style={{ fontSize: 10, marginTop: 6, padding: '5px 8px', borderRadius: 6, border: `1px solid ${C.warn}66`, color: C.warn }}>
          ⏳ {p.symbol} maker {p.side} {usd(p.notional)} @ {Number(p.price).toPrecision(6)} · {p.bars} bar bekledi
        </div>
      ))}
      {f?.positions?.map((p: any) => (
        <div key={p.symbol} style={{ fontSize: 10, marginTop: 6, padding: '5px 8px', borderRadius: 6, border: `1px solid ${C.neon}55` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <b style={{ color: p.direction === 'LONG' ? C.neon : C.danger }}>{p.symbol} {p.direction} · {p.order_type} · {p.trigger}</b>
            <span className="mono" style={{ color: p.unrealized >= 0 ? C.neon : C.danger }}>{usd(p.unrealized)} ({p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct}%)</span>
          </div>
          <div className="mono" style={{ fontSize: 8.5, color: C.muted }}>
            giriş {p.entry} · stop {Number(p.hard_stop || p.stop).toPrecision(6)}{p.partial_done ? ' (BE)' : ''} · hedef {Number(p.target).toPrecision(6)} · {p.age_min} dk · {p.exit_mode}
          </div>
          <div className="mono" style={{ fontSize: 8.5, color: C.muted }}>
            tepe net %{p.peak_net_pct} · net şimdi %{p.net_pct} · PCR {p.peak_capture_now ?? '—'}
            {p.armed ? <span style={{ color: C.neon }}> · TEPE KORUMASI SİLAHLI → çık ≤ {Number(p.giveback_level).toPrecision(6)}</span> : ' · tepe koruması silahlanmadı'}
            {p.chandelier ? ` · ATR trail ${Number(p.chandelier).toPrecision(6)}` : ''}
            {p.cont_prob != null ? ` · devam p ${Math.round(p.cont_prob * 100)}%` : ''}
          </div>
        </div>
      ))}

      {/* ── KISA PİYASA OLASILIKLARI ── */}
      {o && (
        <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
            <Radar size={11} color={C.info} /><span className="cm-title">KISA VADELİ PİYASA OLASILIKLARI</span>
          </div>
          <div style={{ fontSize: 9.5, lineHeight: 1.6 }}>
            <div>
              <span style={{ color: C.muted }}>4h konsensüs ({nTot} parite): </span>
              <b style={{ color: C.neon }}>LONG {cons.LONG || 0}</b> · <b style={{ color: C.danger }}>SHORT {cons.SHORT || 0}</b> · <span style={{ color: C.muted }}>NÖTR {cons.FLAT || 0}</span>
              {o.avg_prob_up != null && <span style={{ color: C.muted }}> · ort. yukarı olasılığı <b style={{ color: C.text }}>%{Math.round(o.avg_prob_up * 100)}</b></span>}
            </div>
            {o.regimes && Object.keys(o.regimes).length > 0 && (
              <div><span style={{ color: C.muted }}>rejim (HMM): </span>
                {Object.entries(o.regimes).map(([k, v]: any) => `${REG_TR[k] || k} ${v}`).join(' · ')}</div>
            )}
            {o.movers?.length > 0 && (
              <div style={{ marginTop: 3 }}>
                <span style={{ color: C.muted }}>bugün ≥%1 oynama olasılığı (mover): </span>
                {o.movers.map((m: any) => (
                  <span key={m.symbol} className="mono" style={{ marginRight: 6, color: m.trusted === false ? C.muted : C.text }}
                        title={`taban %${Math.round((m.base_rate || 0) * 100)} · lift ${m.lift} · beklenen hareket %${m.expected_move_pct}${m.trusted === false ? ' · model doğrulamayı geçemedi' : ''}`}>
                    {m.symbol.replace('/USDT', '')} <b style={{ color: (m.probability || 0) >= 0.8 ? C.neon : C.text }}>%{Math.round((m.probability || 0) * 100)}</b>
                  </span>
                ))}
              </div>
            )}
            <div style={{ color: C.muted }}>komite kazanma olasılığı öncülü: <b style={{ color: C.text }}>%{Math.round((o.p_win_committee || 0.5) * 100)}</b> (kendi kaydıyla güncellenir)</div>
            <div style={{ fontSize: 8, color: C.muted, marginTop: 2 }}>{o.note}</div>
          </div>
        </div>
      )}

      {/* ── HABER & SOSYAL TARAYICI ── */}
      {f?.news_market && (
        <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
            <span className="cm-title">HABER & SOSYAL TARAYICI</span>
            <span style={{ fontSize: 8, color: f.news_market.level >= 2 ? C.danger : f.news_market.level === 1 ? C.warn : C.neon }}>
              risk-off {f.news_market.risk_off_score ?? '—'} · seviye {f.news_market.level}
            </span>
            {f.news_market.age_sec != null && <span className="mono" style={{ marginLeft: 'auto', fontSize: 8, color: C.muted }}>{Math.round(f.news_market.age_sec / 60)} dk önce</span>}
          </div>
          {f.news_market.last_error && <div style={{ fontSize: 8.5, color: C.danger }}>tarayıcı hatası: {f.news_market.last_error}</div>}
          {!f.news_market.items?.length && !f.news_market.last_error && <div style={{ fontSize: 8.5, color: C.muted }}>sistemik risk başlığı yok (RSS·Reddit·StockTwits·Binance duyuruları, 10 dk)</div>}
          {f.news_market.items?.slice(0, 3).map((it: any, i: number) => (
            <div key={i} style={{ fontSize: 8.5, color: C.warn, padding: '1px 0' }}>⚠ [{it.source}] {String(it.title).slice(0, 90)}</div>
          ))}
          {f.nearest?.filter((n: any) => n.news && (n.news.n || 0) > 0).slice(0, 4).map((n: any) => (
            <div key={n.symbol} className="mono" style={{ fontSize: 8.5, color: C.muted, padding: '1px 0' }}>
              {n.symbol.replace('/USDT', '')} haber {n.news.score >= 0 ? '+' : ''}{n.news.score} ({n.news.n}){n.news.confirmed ? ' · hareket DOĞRULANDI' : ''}{n.news.catalysts?.length ? ' · ' + n.news.catalysts.join(',') : ''}{n.news.risks?.length ? ' · risk: ' + n.news.risks.join(',') : ''}
            </div>
          ))}
        </div>
      )}

      {/* ── TETİKLEYİCİYE EN YAKIN ── */}
      {f?.nearest?.length > 0 && (
        <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
            <TrendingUp size={11} color={C.warn} /><span className="cm-title">TETİKLEYİCİYE EN YAKIN</span>
          </div>
          {f.nearest.map((n: any) => (
            <div key={n.symbol} style={{ fontSize: 9, padding: '3px 0', borderTop: `1px solid ${C.border}` }}>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                <b style={{ color: n.ready ? C.neon : C.text }}>{n.symbol.replace('/USDT', '')}</b>
                {n.tier === 'light' && <span style={{ fontSize: 7.5, color: C.muted, border: `1px solid ${C.border}`, borderRadius: 3, padding: '0 3px' }}>hafif</span>}
                <span className="mono" style={{ color: C.muted }}>
                  {n.template === 'pullback' ? `geri çekilme · EMA %${n.dist_ema_pct ?? '—'} · trend ${n.trend_up ? '↑' : '↓'}` : `dip · z ${n.z ?? '—'} · RSI ${n.rsi ?? '—'}`}
                </span>
                {n.ready && <span style={{ color: C.neon, fontWeight: 800 }}>TETİK HAZIR → oylama</span>}
              </div>
              {n.missing?.length > 0 && <div style={{ color: C.muted, fontSize: 8.5 }}>eksik: {n.missing.join(' · ')}</div>}
            </div>
          ))}
        </div>
      )}

      {/* ── OLAY AKIŞI ── */}
      <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
        <div className="cm-title" style={{ marginBottom: 4 }}>OLAY AKIŞI</div>
        {!f?.events?.length && <div style={{ fontSize: 9, color: C.muted }}>henüz olay yok</div>}
        <div style={{ maxHeight: 220, overflowY: 'auto' }}>
          {f?.events?.map((e: any, i: number) => (
            <div key={i} style={{ fontSize: 9, padding: '2px 0', color: e.type === 'error' ? C.danger : e.type === 'entry' ? C.neon : e.type === 'exit' ? C.warn : e.type === 'learn' ? C.violet : '#8892B0', lineHeight: 1.4 }}>
              <span className="mono" style={{ color: C.muted }}>{hhmm(e.ts)}</span> {e.msg}
            </div>
          ))}
        </div>
      </div>

      {/* ── KAPANAN İŞLEMLER ── */}
      <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
        <div className="cm-title" style={{ marginBottom: 4 }}>KAPANAN İŞLEMLER</div>
        {!f?.trades?.length && <div style={{ fontSize: 9, color: C.muted }}>henüz kapanan işlem yok — ilk işlem açılıp kapanınca burada ve GÜNLÜK'te görünür</div>}
        {f?.trades?.map((t: any, i: number) => (
          <div key={i} className="mono" style={{ fontSize: 9, display: 'flex', justifyContent: 'space-between', gap: 6, padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
            <span>{t.symbol} {t.direction} · {t.reason} · {t.hold_bucket}</span>
            <span style={{ color: t.net_pnl >= 0 ? C.neon : C.danger }}>{usd(t.net_pnl)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

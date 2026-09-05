/**
 * RİSK RAYLARI — aynı trend sinyali, üç farklı kaldıraç. Hepsi PAPER.
 *
 * Kaynak: /api/trend/tracks (60 sn, sekme açıkken). Uç yoksa (404) kart HİÇ çizilmez.
 *
 * NEDEN VAR: "daha yüksek kazanç" isteği bu sistemde tek bir ölçülmüş yere işaret ediyor —
 * trend katmanı. Scalp katmanı ölçüldü ve negatif; oraya kaldıraç koymak zararı çarpar.
 * Kaldıraç Sharpe'ı ARTIRMAZ: ortalamayı da sapmayı da aynı katsayıyla büyütür. Bu yüzden
 * kart getiriyi hep DÜŞÜŞLE yan yana gösterir — biri diğeri olmadan okunamaz.
 *
 * Panel sayı ÜRETMEZ: API'den gelmeyen değer "—" yazılır. Ölçülmemiş Sharpe (n < 5 gün)
 * SIFIR değil BOŞ gösterilir; "ölçülmedi" ile "kötü" farklı şeylerdir.
 */
import { useState, useEffect, useCallback } from 'react'
import { TrendingUp, ShieldAlert } from 'lucide-react'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', muted:'#6B7394', text:'#E2E8F0',
            border:'#1E2A45', panel:'#131A2E', surface:'#1A2240', violet:'#A78BFA' }
const REFRESH_MS = 60000

type Track = {
  track?: string; available?: boolean; reason?: string
  initial?: number; equity?: number; return_pct?: number; days?: number
  mean_daily_pct?: number | null; vol_daily_pct?: number | null
  sharpe?: number | null; sharpe_note?: string | null
  max_drawdown_pct?: number; peak_equity?: number; dd_locked?: boolean
  invested_pct?: number; target_vol_pct?: number | null; max_lev?: number | null
  max_exposure?: number | null
  dd_gates_pct?: { soft?: number; hard?: number; kill?: number } | null
  ruined?: boolean
  last_rebalance?: string
}
type Payload = { tracks?: Track[]; measured_tracks?: number; note?: string; warning?: string }

const TR: Record<string, string> = {
  base: 'TEMEL', aggressive: 'AGRESİF', extreme: 'UÇ', max: 'MAKSİMUM',
}
const DESC: Record<string, string> = {
  base: 'ölçülmüş taban — 48 gerçek gün',
  aggressive: 'kaldıraçlı — düşüş ~3 kat',
  extreme: 'yüksek kaldıraç — düşüş ~5 kat',
  max: '%1/gün hedefi — 17× maruziyet · TASFİYE mümkün',
}

const num = (v: number | null | undefined, d = 2, suf = '') =>
  (typeof v === 'number' && isFinite(v)) ? v.toFixed(d) + suf : '—'

export default function RiskTracks() {
  const [d, setD] = useState<Payload | null>(null)
  const [phase, setPhase] = useState<'load' | 'ok' | 'hide'>('load')
  const [gone, setGone] = useState(false)

  const pull = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/trend/tracks`)
      if (r.status === 404) { setGone(true); setPhase(p => (p === 'ok' ? p : 'hide')); return }
      if (!r.ok) throw new Error(String(r.status))
      setD(await r.json())
      setPhase('ok')
    } catch {
      setPhase(p => (p === 'ok' ? p : 'hide'))
    }
  }, [])

  useEffect(() => {
    if (gone) return
    pull()
    const iv = setInterval(pull, REFRESH_MS)
    return () => clearInterval(iv)
  }, [pull, gone])

  if (phase === 'hide') return null
  if (phase === 'load' || !d) return <div style={{ fontSize: 10, color: C.muted }}>risk rayları yükleniyor…</div>

  const rows = (d.tracks || [])

  return (
    <div style={{ fontSize: 10, color: C.text }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <TrendingUp size={12} color={C.violet} />
        <span style={{ letterSpacing: 0.6, color: C.violet }}>RİSK RAYLARI — AYNI SİNYAL, FARKLI KALDIRAÇ</span>
        <span style={{ color: C.muted }}>· hepsi PAPER · ölçülen ray: {d.measured_tracks ?? '—'}</span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', minWidth: 880, width: '100%' }}>
          <thead>
            <tr style={{ color: C.muted, textAlign: 'right' }}>
              {['RAY', 'ÖZSERMAYE', 'GETİRİ %', 'GÜN', 'GÜNLÜK ORT %', 'GÜNLÜK SAPMA %',
                'SHARPE', 'MAKS DÜŞÜŞ %', 'YATIRILAN %', 'HEDEF VOL %', 'MARUZİYET TAVANI', 'DD KAPILARI %'].map((h, i) => (
                <th key={h} style={{ padding: '3px 6px', borderBottom: `1px solid ${C.border}`,
                                     textAlign: i === 0 ? 'left' : 'right', fontWeight: 400 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(t => {
              const k = t.track || '?'
              if (!t.available) {
                return (
                  <tr key={k}>
                    <td style={{ padding: '3px 6px', color: C.text }}>{TR[k] || k}</td>
                    <td colSpan={11} style={{ padding: '3px 6px', color: C.muted }}>{t.reason || 'veri yok'}</td>
                  </tr>
                )
              }
              const pos = (t.return_pct ?? 0) >= 0
              return (
                <tr key={k} style={{ borderBottom: `1px solid ${C.border}` }}>
                  <td style={{ padding: '3px 6px' }}>
                    <div style={{ color: C.text }}>{TR[k] || k}</div>
                    <div style={{ color: C.muted, fontSize: 9 }}>{DESC[k] || ''}</div>
                    {t.ruined && (
                      <div style={{ color: C.danger, fontSize: 9, display: 'flex', alignItems: 'center', gap: 3 }}>
                        <ShieldAlert size={9} /> TASFİYE — sermaye tükendi
                      </div>
                    )}
                    {!t.ruined && t.dd_locked && (
                      <div style={{ color: C.danger, fontSize: 9, display: 'flex', alignItems: 'center', gap: 3 }}>
                        <ShieldAlert size={9} /> KİLİTLİ — nakit
                      </div>
                    )}
                  </td>
                  <td style={{ padding: '3px 6px', textAlign: 'right' }}>{num(t.equity)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: pos ? C.neon : C.danger }}>
                    {num(t.return_pct)}
                  </td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.muted }}>{t.days ?? '—'}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right' }}>{num(t.mean_daily_pct, 4)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.muted }}>{num(t.vol_daily_pct, 4)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right' }}
                      title={t.sharpe_note || ''}>
                    {t.sharpe === null || t.sharpe === undefined
                      ? <span style={{ color: C.muted }}>ölçülmedi</span>
                      : num(t.sharpe)}
                  </td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.warn }}>{num(t.max_drawdown_pct)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right' }}>{num(t.invested_pct, 1)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.muted }}>{num(t.target_vol_pct, 0)}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.muted }}>
                    {t.max_exposure ? `${t.max_exposure}×` : '—'}
                  </td>
                  <td style={{ padding: '3px 6px', textAlign: 'right', color: C.muted }}>
                    {t.dd_gates_pct
                      ? `${t.dd_gates_pct.soft}/${t.dd_gates_pct.hard}/${t.dd_gates_pct.kill}`
                      : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {d.note && (
        <div style={{ marginTop: 6, color: C.muted, lineHeight: 1.5 }}>{d.note}</div>
      )}
      {d.warning && (
        <div style={{ marginTop: 4, padding: 6, background: C.surface, border: `1px solid ${C.border}`,
                      color: C.warn, lineHeight: 1.5 }}>
          <ShieldAlert size={10} style={{ verticalAlign: -1, marginRight: 4 }} />
          {d.warning}
        </div>
      )}
      <div style={{ marginTop: 4, color: C.muted, lineHeight: 1.5 }}>
        DD KAPILARI: düşüş <b>yumuşak</b> eşiğe kadar kaldıraç tam; <b>sert</b> eşikte yarıya iner;
        <b> kill</b> eşiğinde sıfırlanır (ray nakde geçer ve sert eşiğin altına dönene dek kilitli kalır).
        Kaldıracın 1×'i aşan kısmı borçtur ve finansman gideri günlük olarak düşülür.
        MARUZİYET TAVANI = sum(ağırlıklar) sınırı; gerçek risk budur, çarpan değil.
      </div>
      <div style={{ marginTop: 4, padding: 6, background: C.surface, border: `1px solid ${C.danger}`,
                    color: C.danger, lineHeight: 1.5 }}>
        <ShieldAlert size={10} style={{ verticalAlign: -1, marginRight: 4 }} />
        MAKSİMUM rayının sınırı: portföy <b>günde bir kez</b> yeniden dengelenir, dolayısıyla düşüş
        kapıları <b>gün içinde çalışmaz</b> — tek bir felaket günü tam kaldıraçla geçer. 17× maruziyette
        dayanakta −%5,9'luk bir hareket sermayeyi bitirir. Bu ray bir hedef değil, <b>%1/gün'ün
        ölçülmüş fiyat etiketidir</b>.
      </div>
    </div>
  )
}

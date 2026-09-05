/**
 * VİDEO KAYNAKLI KURULUMLAR — 21 YouTube videosundan çıkarılan MEKANİK kurulumlar.
 *
 * Kaynak: /api/video-sources (60 sn, sekme açıkken). Uç yoksa (404) ya da hata verirse
 * kart HİÇ çizilmez — bozuk kutu göstermek yerine sessizce gizlenir.
 *
 * Kanalların "günlük %10", "%90 kazanma" İDDİALARI ALINMADI; yalnız anlattıkları mekanik
 * kodlandı. Kart üçünü yan yana koyar: (1) kurulum nereden geldi ve iddia neydi,
 * (2) o iddianın kanıtı neydi, (3) BİZ ölçünce ne çıktı.
 *
 * Panel sayı ÜRETMEZ: API'den gelmeyen değer "—" yazılır.
 */
import { useState, useEffect, useCallback } from 'react'
import { Video, Clock, Ban } from 'lucide-react'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', muted:'#6B7394', text:'#E2E8F0',
            border:'#1E2A45', panel:'#131A2E', surface:'#1A2240', violet:'#A78BFA' }
const GRAY_BLUE = '#7A8CA8'    // "geri test var ama küçük/tek dönem" — ne kırmızı ne sarı
const DIM_RED   = '#A06A76'    // kısılmış seans çarpanı (< 0,5)
const REFRESH_MS = 60000
const COLS = '190px 150px minmax(220px, 1fr) 96px 96px 168px'
const MINW = 950

type Live = { n?: number; wins?: number; net?: number }
type EvidenceLive = { n?: number; mean_net_pct?: number; t_stat?: number; pos_share?: number }
type Source = {
  sleeve?: string; sleeve_tr?: string; channels?: string[]; claim?: string; evidence?: string
  note?: string; live?: Live; state?: string; evidence_live?: EvidenceLive
}
type Cell = { n?: number; mean_net_pct?: number; t?: number }
type Payload = {
  configured?: boolean
  sources?: Source[]
  not_implemented?: { item?: string; why?: string }[]
  killzones?: { name?: string; utc?: string; size_mult?: number }[]
  session_size_mult?: Record<string, number>
  measured?: { window?: string; n_candidates?: number; cost_pct_roundtrip?: number
               all?: Cell; by_session?: Record<string, Cell>; note?: string }
  evidence_legend?: Record<string, string>
  note?: string
  n_videos?: number; videos?: number
}

/* ── biçimlendirme: gelmeyen değer "—" ── */
const num = (x: any, d = 0) => (x == null || !isFinite(x)) ? '—'
  : Number(x).toLocaleString('tr-TR', { minimumFractionDigits: d, maximumFractionDigits: d })
const pct = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—' : `${Number(x) >= 0 ? '+' : ''}${Number(x).toFixed(d)}%`
const usd = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) < 0 ? '−' : ''}$${Math.abs(Number(x)).toLocaleString('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const tst = (x: any) => (x == null || !isFinite(x)) ? '—'
  : Number(x).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).replace(/^-/, '−')
const mlt = (x: any) => (x == null || !isFinite(x)) ? '—' : `×${Number(x).toLocaleString('tr-TR', { maximumFractionDigits: 2 })}`
const share = (x: any) => (x == null || !isFinite(x)) ? '—' : `%${Math.round(Number(x) * 100)}`

/* t > 2 yeşil · t < −2 kırmızı · arası gri (yani: kanıt yok) */
const tCol = (x: any) => (x == null || !isFinite(x)) ? C.muted : Number(x) > 2 ? C.neon : Number(x) < -2 ? C.danger : C.muted
const signCol = (x: any) => (x == null || !isFinite(x)) ? C.muted : Number(x) >= 0 ? C.neon : C.danger
/* çarpan 1 (ve üstü) yeşil · 0,5'in altı kırmızımsı gri · arası nötr */
const multCol = (x: any) => (x == null || !isFinite(x)) ? C.muted : Number(x) >= 1 ? C.neon : Number(x) < 0.5 ? DIM_RED : C.muted

const EV_COL: Record<string, string> = {
  NONE: C.danger, ANECDOTE: C.danger, SCREENSHOT: C.danger,
  REPLAY_DEMO: C.warn, BACKTEST_TINY: C.warn, MANUAL_BACKTEST_31: C.warn,
  BACKTEST_SMALL: GRAY_BLUE,
}
const STATE_COL: Record<string, string> = { PROVEN: C.neon, UNPROVEN: C.muted, PROBATION: C.warn, PAUSED: C.danger }

function Badge({ text, col, title }: { text: string; col: string; title?: string }) {
  return <span title={title} style={{ fontSize: 8.5, fontWeight: 800, color: col, border: `1px solid ${col}55`,
    borderRadius: 4, padding: '1px 6px', whiteSpace: 'nowrap', cursor: title ? 'help' : 'default' }}>{text}</span>
}

export default function VideoSources() {
  const [d, setD] = useState<Payload | null>(null)
  const [phase, setPhase] = useState<'load' | 'ok' | 'hide'>('load')
  const [gone, setGone] = useState(false)          // 404 → uç yok, yoklamayı da bırak
  const [openNotes, setOpenNotes] = useState<Set<string>>(new Set())

  const toggle = useCallback((k: string) => setOpenNotes(prev => {
    const n = new Set(prev)
    if (n.has(k)) n.delete(k)
    else n.add(k)
    return n
  }), [])

  const pull = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/video-sources`)
      if (r.status === 404) { setGone(true); setPhase(p => (p === 'ok' ? p : 'hide')); return }
      if (!r.ok) throw new Error(String(r.status))
      setD(await r.json())
      setPhase('ok')
    } catch {
      // hiç veri gelmediyse sessizce gizle; bir kez geldiyse son iyi veri ekranda kalsın
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
  if (phase === 'load' || !d) return <div style={{ fontSize: 10, color: C.muted }}>video kaynakları yükleniyor…</div>

  const configured = d.configured !== false
  const sources: Source[] = Array.isArray(d.sources) ? d.sources : []
  const ni = Array.isArray(d.not_implemented) ? d.not_implemented : []
  const m = d.measured || {}
  const legend = d.evidence_legend || {}
  const nVideos = typeof d.n_videos === 'number' ? d.n_videos : typeof d.videos === 'number' ? d.videos : null

  // ölçüm satırları: önce "TÜMÜ" (ham sinyal), sonra seans kırılımı
  const mrows: [string, Cell][] = [
    ...(m.all ? ([['TÜMÜ', m.all]] as [string, Cell][]) : []),
    ...(Object.entries(m.by_session || {}) as [string, Cell][]),
  ]

  // seans rozetleri: killzones (UTC penceresi var) + yalnız çarpanı bilinen seanslar
  const kz = Array.isArray(d.killzones) ? d.killzones : []
  const smult = d.session_size_mult || {}
  const kzNames = new Set(kz.map(k => String(k?.name ?? '')))
  const sessions = [
    ...kz.map(k => ({ name: String(k?.name ?? '—'), utc: k?.utc, m: k?.size_mult ?? smult[String(k?.name ?? '')] })),
    ...Object.entries(smult).filter(([k]) => !kzNames.has(k)).map(([k, v]) => ({ name: k, utc: undefined, m: v })),
  ]

  return (
    <div style={{ fontSize: 9.5 }}>
      {/* ── 1) üst şerit: künye ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
        <Video size={13} color={C.violet} />
        <span style={{ fontSize: 11, fontWeight: 800, letterSpacing: 0.6, color: C.text }}>
          {sources.length || '—'} KURULUM · {nVideos ?? '—'} VİDEO · İDDİALAR ALINMADI, MEKANİK ALINDI
        </span>
        <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>{m.window || '—'}</span>
        {!configured && <Badge text="ÖLÇÜM YOK (yapılandırılmadı)" col={C.muted} title="canlı sütunlar boş kalır" />}
        <span style={{ marginLeft: 'auto' }}>
          <Badge text="DOĞRULANMIŞ GÜNLÜK %10 KANITI YOK" col={C.warn}
                 title="Kanallar günlük %10 / %90 kazanma iddia etti; hiçbirinin denetlenebilir kaydı yok." />
        </span>
      </div>

      {/* ── 2) ölçüm özeti: BİZ ölçtük ── */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 3 }}>
        <div className="cm-title">
          BİZ ÖLÇTÜK {m.cost_pct_roundtrip != null ? `(ham sinyal, maliyet %${num(m.cost_pct_roundtrip, 2)} düşülmüş)` : '(ham sinyal)'}
        </div>
        <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>{num(m.n_candidates)} aday</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 9 }}>
          <thead><tr style={{ color: C.muted, textAlign: 'left' }}>
            {['pencere', 'n', 'ort net %', 't'].map(h => (
              <th key={h} style={{ padding: '2px 6px', fontWeight: 600, borderBottom: `1px solid ${C.border}` }}>{h}</th>
            ))}
          </tr></thead>
          <tbody>
            {mrows.length === 0 && <tr><td colSpan={4} style={{ padding: '3px 6px', color: C.muted }}>ölçüm gelmedi — —</td></tr>}
            {mrows.map(([k, v]) => (
              <tr key={k} style={{ borderTop: `1px solid ${C.border}` }}>
                <td style={{ padding: '2px 6px', fontWeight: k === 'TÜMÜ' ? 800 : 600, color: k === 'TÜMÜ' ? C.text : C.muted, whiteSpace: 'nowrap' }}>{k}</td>
                <td className="mono" style={{ padding: '2px 6px' }}>{num(v?.n)}</td>
                <td className="mono" style={{ padding: '2px 6px', color: signCol(v?.mean_net_pct) }}>{pct(v?.mean_net_pct, 3)}</td>
                <td className="mono" style={{ padding: '2px 6px', color: tCol(v?.t), fontWeight: 700 }}>{tst(v?.t)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {m.note && <div style={{ fontSize: 8.5, color: C.muted, marginTop: 3, lineHeight: 1.5 }}>{m.note}</div>}

      {/* ── 3) kurulum tablosu: iddia · kanıt · canlı ── */}
      <div className="cm-title" style={{ margin: '8px 0 3px' }}>KURULUMLAR (kanal · iddia · kanıt · canlı)</div>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: MINW }}>
          <div style={{ display: 'grid', gridTemplateColumns: COLS, gap: 8, fontSize: 8, color: C.muted,
                        letterSpacing: 0.5, paddingBottom: 3, borderBottom: `1px solid ${C.border}` }}>
            <div>KURULUM</div><div>KANALLAR</div><div>İDDİA (kanalın sözü)</div><div>KANIT</div><div>DURUM</div><div>CANLI (bizim)</div>
          </div>
          {sources.length === 0 && <div style={{ fontSize: 9.5, color: C.muted, padding: '4px 0' }}>kurulum listesi gelmedi — —</div>}
          {sources.map((s, i) => {
            const key = String(s.sleeve ?? i)
            const isOpen = openNotes.has(key)
            const ch = Array.isArray(s.channels) ? s.channels : []
            const chTxt = ch.length === 0 ? '—' : ch.length > 2 ? `${ch.slice(0, 2).join(', ')} +${ch.length - 2}` : ch.join(', ')
            const lv = configured ? s.live : undefined
            const el = configured ? s.evidence_live : undefined
            return (
              <div key={key} style={{ borderTop: `1px solid ${C.border}` }}>
                <div role="button" tabIndex={0} aria-expanded={isOpen}
                     onClick={() => toggle(key)}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(key) } }}
                     title="ayrıntı için tıkla"
                     style={{ display: 'grid', gridTemplateColumns: COLS, gap: 8, alignItems: 'center',
                              padding: '4px 0', cursor: 'pointer' }}>
                  <div style={{ minWidth: 0 }}>
                    <b style={{ color: C.text, fontSize: 9.5 }}>{s.sleeve_tr || s.sleeve || '—'}</b>
                    <div className="mono" style={{ fontSize: 8, color: C.muted }}>{s.sleeve || '—'}</div>
                  </div>
                  <div title={ch.join(' · ') || undefined} style={{ fontSize: 9, color: C.muted, minWidth: 0,
                       whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{chTxt}</div>
                  <div title={s.claim || undefined} style={{ fontSize: 9, color: C.muted, minWidth: 0,
                       whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.claim || '—'}</div>
                  <div><Badge text={s.evidence || '—'} col={EV_COL[String(s.evidence)] || C.muted}
                              title={legend[String(s.evidence)] || 'kanıt sınıfı açıklanmadı'} /></div>
                  <div><Badge text={s.state || '—'} col={STATE_COL[String(s.state)] || C.muted} /></div>
                  <div className="mono" style={{ fontSize: 8.5, color: C.muted, whiteSpace: 'nowrap' }}>
                    {lv?.n != null ? `${num(lv.n)} işlem` : '—'} · <span style={{ color: signCol(lv?.net) }}>{usd(lv?.net)}</span>
                    {' · t '}<span style={{ color: tCol(el?.t_stat) }}>{tst(el?.t_stat)}</span>
                  </div>
                </div>
                {isOpen && (
                  <div style={{ padding: '0 0 6px 2px', fontSize: 9, color: C.muted, lineHeight: 1.55 }}>
                    <div>{s.note || 'not yok — —'}</div>
                    {ch.length > 0 && <div>kanallar: <span style={{ color: C.text }}>{ch.join(' · ')}</span></div>}
                    <div>kanıt: {legend[String(s.evidence)] || s.evidence || '—'}</div>
                    <div className="mono">canlı ölçüm: n {num(el?.n)} · ort net {pct(el?.mean_net_pct, 3)} · t {tst(el?.t_stat)} · pozitif payı {share(el?.pos_share)} · kazanan {num(lv?.wins)}</div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ── 4) seans çarpanları ── */}
      <div className="cm-title" style={{ margin: '8px 0 3px', display: 'flex', alignItems: 'center', gap: 4 }}>
        <Clock size={10} color={C.muted} /> SEANS ÇARPANLARI
      </div>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        {sessions.length === 0 && <span style={{ fontSize: 9, color: C.muted }}>seans çarpanı gelmedi — —</span>}
        {sessions.map(s => (
          <Badge key={s.name} col={multCol(s.m)} text={`${s.name}${s.utc ? ` ${s.utc}` : ''} ${mlt(s.m)}`}
                 title={s.utc ? `${s.name} · UTC ${s.utc}` : `${s.name} · UTC penceresi verilmedi`} />
        ))}
      </div>
      <div style={{ fontSize: 8.5, color: C.muted, marginTop: 3 }}>
        Seans öncülü ölçümle geldi; sistem kendi işlemlerini biriktirdikçe seans kapısı bunu ezer.
      </div>

      {/* ── 5) alınmayanlar ── */}
      {ni.length > 0 && (
        <div style={{ marginTop: 8, border: `1px solid ${C.danger}55`, borderRadius: 6, padding: '6px 8px', background: C.panel }}>
          <div className="cm-title" style={{ color: C.danger, marginBottom: 3, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Ban size={10} color={C.danger} /> ALINMAYANLAR (ve nedeni)
          </div>
          {ni.map((x, i) => (
            <div key={i} style={{ fontSize: 9, padding: '2px 0', lineHeight: 1.5, borderTop: i ? `1px solid ${C.border}` : undefined }}>
              <b style={{ color: C.text }}>{x?.item || '—'}</b> <span style={{ color: C.muted }}>— {x?.why || '—'}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── 6) künye notu ── */}
      {d.note && <div style={{ fontSize: 8.5, color: C.muted, marginTop: 8, lineHeight: 1.5 }}>{d.note}</div>}
    </div>
  )
}

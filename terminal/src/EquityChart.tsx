/**
 * ÖZSERMAYE EĞRİSİ — paylaşılan, ZAMAN eksenli grafik (Simulator + Trading).
 *
 * x ekseni gerçek zamanı gösterir: görünen aralık ≤ 24 sa ise HH:mm, daha uzunsa dd.MM HH:mm.
 * Alt şeritteki Brush TÜM serinin üzerinde kayar; kullanıcı pencereyi sağa/sola çekerek ilk
 * başlangıç anından bugüne her yere gidebilir. Grafik alanında basılı tutup sürükleme = pan,
 * fare tekerleği = imleç etrafında zoom (en az 20 nokta), dokunmatikte tek parmak = pan.
 *
 * Pencere durumu veri indeksiyle (startIndex/endIndex) tutulur. Yeni veri geldiğinde kullanıcı
 * EN SAĞDAYSA pencere sağa yapışık kalır ("canlı takip"); değilse baktığı ZAMAN aralığı korunur —
 * backend seriyi yeniden örneklese bile kullanıcı bulunduğu yerden fırlatılmaz.
 *
 * Panel sayı ÜRETMEZ: gelen noktalar olduğu gibi çizilir, eksik değer "—" yazılır.
 */
import { useState, useEffect, useRef, useCallback, useMemo, useId } from 'react'
import { ComposedChart, Area, Scatter, XAxis, YAxis, Tooltip, ReferenceLine, Brush, ResponsiveContainer } from 'recharts'

export type EqPoint = { ts: number; equity: number; src?: string }
export type EqMark = { ts: number; equity: number; net_pnl: number; symbol: string; seq: number }
type Quick = '1' | '6' | '24' | 'all' | null

const C = { neon:'#00FF88', danger:'#FF3B5C', muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45',
            panel:'#131A2E', surface:'#1A2240', bg:'#0A0E1A', violet:'#A78BFA' }
const MIN_PTS = 20                       // tekerlek zoom alt sınırı (nokta)
const PLOT_LEFT = 44, PLOT_RIGHT = 8     // YAxis genişliği + sağ marj (px) — pan/zoom oranı için
const DAY = 24 * 3600

const usd = (x: any) => (x == null || !isFinite(x)) ? '—'
  : `$${Number(x).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
const signedUsd = (x: any) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) >= 0 ? '+' : '−'}${Math.abs(Number(x)).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} $`
const fmtTick = (ts: number, spanSec: number) => new Date(ts * 1000).toLocaleString('tr-TR',
  spanSec <= DAY ? { hour: '2-digit', minute: '2-digit' } : { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
const fmtDM = (ts: number) => new Date(ts * 1000).toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })
const fmtFull = (ts: number) => new Date(ts * 1000).toLocaleString('tr-TR',
  { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
/** ts ≥ hedef olan ilk indeks (ikili arama; seri artan ts) */
const lowerBound = (pts: EqPoint[], ts: number) => {
  let lo = 0, hi = pts.length
  while (lo < hi) { const m = (lo + hi) >> 1; if (pts[m].ts < ts) lo = m + 1; else hi = m }
  return lo
}

export default function EquityChart({ points, marks, capital, color = C.violet, height = 170, label }: {
  points: EqPoint[]; marks?: EqMark[]; capital?: number; color?: string; height?: number; label?: string
}) {
  // Geçersiz noktaları (ts/equity yok) çizme — ama uydurma da yapma
  const pts = useMemo(() => (points || []).filter(p => p && isFinite(p.ts) && isFinite(p.equity)), [points])
  const n = pts.length
  const last = n - 1
  const [win, setWin] = useState({ s: 0, e: 0 })
  const [quick, setQuick] = useState<Quick>(null)
  const [dragging, setDragging] = useState(false)
  const ptsRef = useRef(pts)
  const winRef = useRef(win)
  // Görünen zaman aralığı + "en sağda mı" bilgisi: veri yenilenince pencereyi yeniden kurmak için
  const range = useRef<{ sTs: number; eTs: number; sticky: boolean; quick: Quick } | null>(null)
  const drag = useRef<{ x: number; s: number; e: number } | null>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const gid = 'eqg' + useId().replace(/[^a-zA-Z0-9]/g, '')

  useEffect(() => { ptsRef.current = pts }, [pts])
  useEffect(() => { winRef.current = win }, [win])

  /** Pencereyi [s,e] indeksine taşı (sınırda durur, en az 2 nokta). */
  const apply = useCallback((s: number, e: number, q: Quick = null) => {
    const p = ptsRef.current; const lst = p.length - 1
    if (lst < 1) return
    s = clamp(Math.round(s), 0, lst); e = clamp(Math.round(e), 0, lst)
    if (s > e) [s, e] = [e, s]
    if (e === s) { if (e < lst) e = s + 1; else s = e - 1 }
    range.current = { sTs: p[s].ts, eTs: p[e].ts, sticky: e === lst, quick: q }
    winRef.current = { s, e }
    setWin(w => (w.s === s && w.e === e) ? w : { s, e })
    setQuick(q)
  }, [])

  // Veri değişti: ilk açılışta son 6 sa (seri kısaysa TÜMÜ); sonra canlı takip / yerinde kal
  useEffect(() => {
    if (last < 1) { range.current = null; return }
    const prev = range.current
    if (!prev) { const s = lowerBound(pts, pts[last].ts - 6 * 3600); apply(s, last, s === 0 ? 'all' : '6'); return }
    if (prev.sticky) {
      if (prev.quick === 'all') { apply(0, last, 'all'); return }
      const hours = prev.quick === '1' ? 1 : prev.quick === '6' ? 6 : prev.quick === '24' ? 24 : null
      const dur = hours != null ? hours * 3600 : Math.max(0, prev.eTs - prev.sTs)
      apply(lowerBound(pts, pts[last].ts - dur), last, prev.quick); return
    }
    apply(lowerBound(pts, prev.sTs), Math.min(last, lowerBound(pts, prev.eTs)), null)
  }, [pts, last, apply])

  /* ── Brush (alt şerit) ── */
  const onBrush = useCallback(({ startIndex, endIndex }: { startIndex: number; endIndex: number }) => {
    const w = winRef.current
    if (startIndex === w.s && endIndex === w.e) return   // kontrollü prop yankısı — dokunma
    apply(startIndex, endIndex)
  }, [apply])

  /* ── Pan: basılı tut + sürükle (fare / tek parmak) ── */
  const plotWidth = () => { const r = wrap.current?.getBoundingClientRect(); return Math.max(1, (r?.width || 0) - PLOT_LEFT - PLOT_RIGHT) }
  const isBrushTarget = (t: EventTarget | null) => !!(t as Element | null)?.closest?.('.recharts-brush')
  const startDrag = (x: number) => { drag.current = { x, s: winRef.current.s, e: winRef.current.e }; setDragging(true) }
  const endDrag = () => { if (drag.current) { drag.current = null; setDragging(false) } }
  const panTo = (x: number) => {
    const d = drag.current; if (!d) return
    const lst = ptsRef.current.length - 1; const size = d.e - d.s
    const di = Math.round(-(x - d.x) / plotWidth() * size)
    const s = clamp(d.s + di, 0, lst - size)
    if (s !== winRef.current.s) apply(s, s + size)
  }

  /* ── Zoom: fare tekerleği, imleç etrafında (passive değil → sayfa kaymasın) ── */
  useEffect(() => {
    const el = wrap.current; if (!el) return
    const h = (ev: WheelEvent) => {
      const p = ptsRef.current; const lst = p.length - 1
      if (lst < 1) return
      ev.preventDefault()
      const { s, e } = winRef.current; const size = e - s
      const r = el.getBoundingClientRect()
      const ratio = clamp((ev.clientX - r.left - PLOT_LEFT) / Math.max(1, r.width - PLOT_LEFT - PLOT_RIGHT), 0, 1)
      const minSpan = Math.min(MIN_PTS, p.length) - 1
      const newSize = clamp(Math.round(size * (ev.deltaY > 0 ? 1.25 : 0.8)), minSpan, lst)
      if (newSize === size) return
      const anchor = s + ratio * size
      const ns = clamp(Math.round(anchor - ratio * newSize), 0, lst - newSize)
      apply(ns, ns + newSize)
    }
    el.addEventListener('wheel', h, { passive: false })
    return () => el.removeEventListener('wheel', h)
  }, [apply])

  /* ── Hızlı düğmeler ── */
  const goHours = (h: number, q: Quick) => { const p = ptsRef.current; const lst = p.length - 1; if (lst < 1) return; apply(lowerBound(p, p[lst].ts - h * 3600), lst, q) }
  const goAll = () => apply(0, ptsRef.current.length - 1, 'all')
  const goStart = () => {   // pencereyi ilk noktaya götür; görünen süreyi koru (TÜMÜ'deyse 6 sa)
    const p = ptsRef.current; const lst = p.length - 1; if (lst < 1) return
    const { s, e } = winRef.current
    const dur = (s === 0 && e === lst) ? 6 * 3600 : Math.max(0, p[e].ts - p[s].ts)
    apply(0, Math.min(lst, lowerBound(p, p[0].ts + dur)), null)
  }

  /* ── Görünen aralık (state bir kare geride kalabilir → kelepçele) ── */
  const s = n ? clamp(win.s, 0, last) : 0
  const e = n ? Math.max(clamp(win.e, 0, last), Math.min(last, s + 1)) : 0   // ilk karede s===e olmasın (NaN ölçek)
  const vs = n ? pts[s].ts : 0, ve = n ? pts[e].ts : 0
  const span = Math.max(1, ve - vs)
  const fullSpan = n ? Math.max(1, pts[last].ts - pts[0].ts) : 1
  const tol = Math.max(30, span / Math.max(1, e - s) / 2)   // tooltip'te işlem eşleştirme toleransı (sn)
  const winMarks = useMemo(() => (marks || []).filter(m => m && isFinite(m.ts) && isFinite(m.equity) && m.ts >= vs && m.ts <= ve), [marks, vs, ve])
  const winsM = useMemo(() => winMarks.filter(m => (m.net_pnl ?? 0) >= 0), [winMarks])
  const lossM = useMemo(() => winMarks.filter(m => (m.net_pnl ?? 0) < 0), [winMarks])
  const nLedger = useMemo(() => pts.filter(p => p.src === 'ledger').length, [pts])
  const nLive = useMemo(() => pts.filter(p => p.src === 'live').length, [pts])
  const tr = (x: number) => x.toLocaleString('tr-TR')

  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 4 }}>
        {label && <span className="cm-title">{label}</span>}
        {n >= 2 && <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>
          {fmtDM(vs)} → {fmtDM(ve)} · {tr(n)} nokta{(nLedger || nLive) ? ` (defter ${tr(nLedger)} + canlı ${tr(nLive)})` : ''}
        </span>}
        {n >= 2 && <div style={{ marginLeft: 'auto', display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <QBtn color={color} on={quick === '1'} onClick={() => goHours(1, '1')}>1 sa</QBtn>
          <QBtn color={color} on={quick === '6'} onClick={() => goHours(6, '6')}>6 sa</QBtn>
          <QBtn color={color} on={quick === '24'} onClick={() => goHours(24, '24')}>24 sa</QBtn>
          <QBtn color={color} on={quick === 'all'} onClick={goAll}>TÜMÜ</QBtn>
          <QBtn color={color} onClick={goStart} title="pencereyi ilk noktaya götür">⟵ BAŞLANGIÇ</QBtn>
        </div>}
      </div>

      {n < 2 ? (
        <div style={{ height, fontSize: 10, color: C.muted, padding: 8 }}>ilk döngü bekleniyor…</div>
      ) : (
        <div ref={wrap} role="img" aria-label={`özsermaye eğrisi, ${fmtFull(vs)} – ${fmtFull(ve)}`}
             style={{ height, userSelect: 'none', touchAction: 'pan-y', cursor: dragging ? 'grabbing' : 'grab' }}
             onMouseDown={ev => { if (ev.button !== 0 || isBrushTarget(ev.target)) return; ev.preventDefault(); startDrag(ev.clientX) }}
             onMouseMove={ev => { if (drag.current) panTo(ev.clientX) }}
             onMouseUp={endDrag} onMouseLeave={endDrag}
             onTouchStart={ev => { if (ev.touches.length !== 1 || isBrushTarget(ev.target)) return; startDrag(ev.touches[0].clientX) }}
             onTouchMove={ev => { if (drag.current && ev.touches.length === 1) panTo(ev.touches[0].clientX) }}
             onTouchEnd={endDrag} onTouchCancel={endDrag}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={pts} margin={{ top: 6, right: PLOT_RIGHT, bottom: 0, left: 0 }}>
              <defs><linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.45} /><stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient></defs>
              <XAxis dataKey="ts" type="number" domain={[vs, ve]} allowDataOverflow height={16} minTickGap={40}
                     tickFormatter={(v: any) => fmtTick(Number(v), span)} tick={{ fontSize: 9, fill: C.muted }}
                     axisLine={{ stroke: C.border }} tickLine={false} />
              <YAxis domain={['auto', 'auto']} width={PLOT_LEFT} tick={{ fontSize: 9, fill: C.muted }}
                     tickFormatter={(v: any) => '$' + Number(v).toFixed(0)} axisLine={false} tickLine={false} />
              <Tooltip content={<EqTip marks={winMarks} tol={tol} color={color} />} cursor={{ stroke: C.border }} isAnimationActive={false} />
              {capital != null && isFinite(capital) && (
                <ReferenceLine y={capital} stroke={C.muted} strokeDasharray="3 3"
                               label={{ value: `sermaye ${usd(capital)}`, position: 'insideTopRight', fontSize: 8, fill: C.muted }} />
              )}
              <Area type="monotone" dataKey="equity" stroke={color} fill={`url(#${gid})`} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              {/* boş data verilirse recharts Scatter grafik verisine düşer (her noktaya nokta basar) → yalnız doluyken çiz */}
              {winsM.length > 0 && <Scatter data={winsM} dataKey="equity" name="işlem" fill={C.neon} isAnimationActive={false}
                       shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={2.5} fill={C.neon} />} />}
              {lossM.length > 0 && <Scatter data={lossM} dataKey="equity" name="işlem" fill={C.danger} isAnimationActive={false}
                       shape={(p: any) => <circle cx={p.cx} cy={p.cy} r={2.5} fill={C.danger} />} />}
              <Brush dataKey="ts" height={22} travellerWidth={8} startIndex={s} endIndex={e} onChange={onBrush}
                     tickFormatter={(v: any) => fmtTick(Number(v), fullSpan)} fill={C.panel} stroke={C.border}
                     ariaLabel="zaman penceresi" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

/** Hızlı aralık düğmesi (aktif olan grafik rengiyle dolu) */
function QBtn({ on, onClick, children, title, color }: { on?: boolean; onClick: () => void; children: any; title?: string; color: string }) {
  return (
    <button type="button" className="cm-btn" onClick={onClick} title={title} aria-pressed={!!on}
      style={{ fontSize: 8, padding: '1px 6px', letterSpacing: 0.3, background: on ? color : C.surface, color: on ? C.bg : C.muted }}>{children}</button>
  )
}

/** Tooltip: tam tarih-saat + $ değer (+ "(defterden)") + o ana denk gelen kapanan işlemler */
function EqTip({ active, payload, label, marks, tol, color }: {
  active?: boolean; payload?: any[]; label?: any; marks: EqMark[]; tol: number; color: string
}) {
  if (!active || label == null) return null
  const ts = Number(label)
  const pt = payload?.find(p => p?.payload && p.payload.seq == null)?.payload as EqPoint | undefined
  const near = marks.filter(m => Math.abs(m.ts - ts) <= tol)
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 4, padding: '4px 7px', fontSize: 10 }}>
      <div className="mono" style={{ color: C.muted, fontSize: 9 }}>{fmtFull(ts)}</div>
      <div className="mono" style={{ color }}>{usd(pt?.equity)}{pt?.src === 'ledger' && <span style={{ color: C.muted }}> (defterden)</span>}</div>
      {near.map(m => (
        <div key={m.seq} className="mono" style={{ color: (m.net_pnl ?? 0) >= 0 ? C.neon : C.danger }}>
          #{m.seq} {m.symbol} net {signedUsd(m.net_pnl)}
        </div>
      ))}
    </div>
  )
}

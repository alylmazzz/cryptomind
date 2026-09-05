/**
 * SAYFALI İŞLEM TABLOSU — sunucu sayfalaması (`/api/simulator/trades?page=&per_page=`).
 *
 * Sayfa 1 = EN YENİ işlemler; `seq` = kronolojik 1-tabanlı sıra numarası (#1 ilk işlem),
 * sayfa içinde azalarak gelir. Tıklanan sayfada hangi sıradaki işlemler varsa o gösterilir
 * (ör. sayfa 3 → #35–#11). Sayfa 1 açıkken 60 sn'de bir yenilenir; başka sayfadaysa
 * YENİLENMEZ (kullanıcının baktığı sayfa kaymasın). Uç yoksa / hata verirse `fallback`
 * (mevcut st.trades — son 50) tek sayfa olarak gösterilir.
 *
 * Panel sayı ÜRETMEZ: API'den gelmeyen değer "—" yazılır.
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react'

export type Trade = {
  seq?: number; symbol?: string; direction?: string; entry?: number; exit?: number; notional?: number
  gross_pnl?: number; fees?: number; net_pnl?: number; pnl_pct?: number; net_pct_realized?: number
  reason?: string; opened_ts?: number; closed_ts?: number; hold_sec?: number; hold_bucket?: string
  peak_net_pct?: number; peak_capture?: number; win?: boolean; order_type?: string; sleeve?: string
  exit_mode?: string; partial_done?: boolean; [k: string]: any
}
export type TradesPage = {
  configured?: boolean; total?: number; page?: number; per_page?: number; pages?: number; order?: string
  from_seq?: number; to_seq?: number; trades?: Trade[]; note?: string
}

const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', muted:'#6B7394', text:'#E2E8F0',
            border:'#1E2A45', surface:'#1A2240', bg:'#0A0E1A', violet:'#A78BFA' }
const PER_PAGE = [25, 50, 100]
const REFRESH_MS = 60000

const usd = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) < 0 ? '−' : ''}$${Math.abs(Number(x)).toLocaleString('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const pct = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—' : `${Number(x) >= 0 ? '+' : ''}${Number(x).toFixed(d)}%`
const prec = (x: any) => (x == null || !isFinite(x)) ? '—' : Number(x).toPrecision(6)
const tstr = (ts: any) => ts ? new Date(ts * 1000).toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'
const num = (x: any) => (x == null || !isFinite(x)) ? '—' : String(x)

/** « ‹ [1] 2 3 … n › » — en çok 7 sayfa düğmesi */
function pageList(page: number, pages: number): (number | '…')[] {
  if (pages <= 7) return Array.from({ length: pages }, (_, i) => i + 1)
  if (page <= 4) return [1, 2, 3, 4, 5, '…', pages]
  if (page >= pages - 3) return [1, '…', pages - 4, pages - 3, pages - 2, pages - 1, pages]
  return [1, '…', page - 1, page, page + 1, '…', pages]
}

export default function TradeTable({ fetchPage, fallback, title }: {
  fetchPage: (page: number, perPage: number) => Promise<TradesPage>
  fallback?: any[]
  title?: string
}) {
  const [page, setPage] = useState(1)
  const [perPage, setPerPage] = useState(PER_PAGE[0])
  const [data, setData] = useState<TradesPage | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const reqId = useRef(0)   // geç gelen eski yanıt yeni sayfayı ezmesin

  const load = useCallback(async (p: number, pp: number) => {
    const id = ++reqId.current
    setLoading(true)
    try {
      const d = await fetchPage(p, pp)
      if (id !== reqId.current) return
      setData(d); setErr(null)
    } catch (e: any) {
      if (id !== reqId.current) return
      const m = String(e?.message || e)
      setErr(/^\d{3}$/.test(m) ? `işlem listesi alınamadı (HTTP ${m})` : `işlem listesi alınamadı (${m})`)
    } finally {
      if (id === reqId.current) setLoading(false)
    }
  }, [fetchPage])

  useEffect(() => { load(page, perPage) }, [load, page, perPage])
  // yalnız sayfa 1 kendini yeniler
  useEffect(() => {
    if (page !== 1) return
    const iv = setInterval(() => load(1, perPage), REFRESH_MS)
    return () => clearInterval(iv)
  }, [load, page, perPage])
  // sayfa sayısı küçüldüyse (ör. sayfa başına değişti) son sayfaya kelepçele
  useEffect(() => { if (data?.pages && page > data.pages) setPage(data.pages) }, [data?.pages, page])

  // Uç hiç yanıt vermediyse (404 / eski sunucu) → fallback tek sayfa, en yeni üstte
  const usingFallback = err != null && data == null
  const fb = useMemo(() => usingFallback && fallback?.length
    ? [...fallback].sort((a, b) => (b?.closed_ts || 0) - (a?.closed_ts || 0)) : [], [usingFallback, fallback])

  if (data?.configured === false) return <div style={{ fontSize: 10, color: C.muted }}>{data.note || 'simülatör kuruluyor…'}</div>

  const rows: Trade[] = data?.trades ?? fb
  const total = usingFallback ? (fb.length || null) : data?.total ?? null
  const pages = usingFallback ? 1 : Math.max(1, data?.pages ?? 1)
  const curPage = usingFallback ? 1 : (data?.page ?? page)
  const fromSeq = usingFallback ? rows[0]?.seq : data?.from_seq
  const toSeq = usingFallback ? rows[rows.length - 1]?.seq : data?.to_seq
  const go = (p: number) => setPage(Math.max(1, Math.min(pages, p)))

  const pbtn = (key: string, lbl: string, target: number, disabled: boolean, active = false) => (
    <button key={key} type="button" className="cm-btn" onClick={() => go(target)} disabled={disabled}
      aria-current={active ? 'page' : undefined}
      style={{ fontSize: 8.5, padding: '1px 6px', background: active ? C.violet : C.surface, color: active ? C.bg : disabled ? C.border : C.muted,
               cursor: disabled ? 'default' : 'pointer' }}>{lbl}</button>
  )

  return (
    <div style={{ minWidth: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 4 }}>
        {title && <span className="cm-title">{title}</span>}
        <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>
          İŞLEM {total != null ? total.toLocaleString('tr-TR') : '—'} · sayfa {curPage}/{pages} · #{num(fromSeq)}–#{num(toSeq)}
        </span>
        {loading && <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>yükleniyor…</span>}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
          <label className="mono" style={{ fontSize: 8.5, color: C.muted }}>sayfa başına{' '}
            <select value={perPage} disabled={usingFallback} aria-label="sayfa başına işlem"
              onChange={ev => { setPerPage(Number(ev.target.value)); setPage(1) }}
              style={{ background: C.surface, color: C.text, border: `1px solid ${C.border}`, borderRadius: 4, fontSize: 9, padding: '1px 3px' }}>
              {PER_PAGE.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <nav aria-label="işlem sayfaları" style={{ display: 'flex', gap: 2 }}>
            {pbtn('first', '«', 1, curPage <= 1)}
            {pbtn('prev', '‹', curPage - 1, curPage <= 1)}
            {pageList(curPage, pages).map((p, i) => p === '…'
              ? <span key={`e${i}`} className="mono" style={{ fontSize: 8.5, color: C.muted, padding: '1px 3px' }}>…</span>
              : pbtn(`p${p}`, String(p), p, false, p === curPage))}
            {pbtn('next', '›', curPage + 1, curPage >= pages)}
            {pbtn('last', '»', pages, curPage >= pages)}
          </nav>
        </div>
      </div>

      {err && <div style={{ fontSize: 9, color: C.danger, marginBottom: 4 }}>{err}{usingFallback && fb.length ? ' — son işlemler önbellekten (tek sayfa)' : ''}</div>}
      {!rows.length && !loading && <div style={{ fontSize: 10, color: C.muted }}>henüz kapanan işlem yok</div>}

      {rows.length > 0 && (
        <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
          <table className="mono" style={{ borderCollapse: 'collapse', width: '100%', fontSize: 9, whiteSpace: 'nowrap' }}>
            <thead><tr style={{ color: C.muted, textAlign: 'left' }}>
              {['#', 'kapanış', 'parite', 'yön', 'sleeve', 'emir', 'giriş → çıkış', 'boyut $', 'brüt $', 'kom $', 'net $', 'net %', 'sebep', 'tutma', 'tepe % / PCR'].map(h => (
                <th key={h} scope="col" style={{ padding: '2px 5px', borderBottom: `1px solid ${C.border}`, fontWeight: 600, position: 'sticky', top: 0, background: '#131A2E' }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {rows.map((t, i) => {
                const neg = (t.net_pnl ?? 0) < 0
                const td = (v: any, c?: string) => <td style={{ padding: '2px 5px', color: c || C.text }}>{v}</td>
                return (
                  <tr key={t.seq ?? `${t.closed_ts}-${i}`} style={{ background: neg ? '#FF3B5C0D' : '#00FF880D', borderTop: `1px solid ${C.border}33` }}>
                    {td(t.seq != null ? `#${t.seq}` : '—', C.muted)}
                    {td(tstr(t.closed_ts), C.muted)}
                    {td(t.symbol ?? '—')}
                    {td(t.direction ?? '—', t.direction === 'LONG' ? C.neon : t.direction === 'SHORT' ? C.danger : C.text)}
                    {td(t.sleeve ?? '—', C.muted)}
                    {td(t.order_type ?? '—', t.order_type === 'maker' ? C.neon : C.warn)}
                    {td(`${prec(t.entry)} → ${prec(t.exit)}`)}
                    {td(usd(t.notional))}
                    {td(usd(t.gross_pnl), (t.gross_pnl ?? 0) >= 0 ? C.neon : C.danger)}
                    {td(usd(t.fees, 3), C.warn)}
                    {td(usd(t.net_pnl), neg ? C.danger : C.neon)}
                    {td(pct(t.net_pct_realized), neg ? C.danger : C.neon)}
                    {td(t.reason ?? '—', C.muted)}
                    {td(`${t.hold_bucket ?? '—'}${t.hold_sec != null ? ` · ${Math.round(t.hold_sec / 60)} dk` : ''}`, C.muted)}
                    {td(`${pct(t.peak_net_pct)} / ${t.peak_capture != null ? Number(t.peak_capture).toFixed(2) : '—'}`, C.muted)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

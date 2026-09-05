/**
 * 1.000 $ SANAL SİMÜLATÖR — herkese açık, salt-okunur.
 *
 * Kaynak: /api/simulator (5-20 sn), /api/simulator/journal (60 sn).
 * Sistemin kendi komite koşucusu: 12 rol · en düşük komisyonlu borsa · maker-öncelikli
 * emir · her işlemde ders. Panel sayı ÜRETMEZ; API'den gelmeyen değer "—" yazar.
 */
import { useState, useEffect, useCallback, useMemo } from 'react'
import { FlaskConical, Users, BookOpen, EyeOff, Coins, ChevronDown, ChevronUp, Activity } from 'lucide-react'
import EquityChart from './EquityChart'
import TradeTable, { type TradesPage } from './TradeTable'
import VideoSources from './VideoSources'
import RiskTracks from './RiskTracks'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E',
            surface:'#1A2240', bg:'#0A0E1A', cyan:'#22D3EE', violet:'#A78BFA' }
const j = async (p: string) => { const r = await fetch(`${API}${p}`); if (!r.ok) throw new Error(String(r.status)); return r.json() }
const usd = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) < 0 ? '−' : ''}$${Math.abs(Number(x)).toLocaleString('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const pct = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—' : `${Number(x) >= 0 ? '+' : ''}${Number(x).toFixed(d)}%`
const ago = (ts: any) => ts ? `${Math.max(0, Math.round(Date.now() / 1000 - ts))} sn önce` : '—'
const tstr = (ts: any) => ts ? new Date(ts * 1000).toLocaleString('tr-TR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'

export default function Simulator() {
  const [st, setSt] = useState<any>(null)
  const [jr, setJr] = useState<any>(null)
  const [open, setOpen] = useState(true)
  const [tab, setTab] = useState<'karar' | 'evren' | 'roller' | 'dersler' | 'gozardi' | 'arastirma' | 'video' | 'raylar' | 'islemler' | 'gunluk'>('karar')
  const [ms, setMs] = useState<any>(null)
  const [rs, setRs] = useState<any>(null)
  const [uni, setUni] = useState<any>(null)
  const pullU = useCallback(async () => { try { setUni(await j('/api/simulator/universe')) } catch {} }, [])
  useEffect(() => { if (tab === 'evren') { pullU(); const iv = setInterval(pullU, 15000); return () => clearInterval(iv) } }, [tab, pullU])
  const pullM = useCallback(async () => { try { setMs(await j('/api/simulator/missed?limit=20')) } catch {} }, [])
  const pullR = useCallback(async () => { try { setRs(await j('/api/research')) } catch {} }, [])
  useEffect(() => { if (tab === 'gozardi') pullM() }, [tab, pullM])
  useEffect(() => { if (tab === 'arastirma') pullR() }, [tab, pullR])

  const pull = useCallback(async () => { try { setSt(await j('/api/simulator')) } catch {} }, [])
  const pullJ = useCallback(async () => { try { setJr(await j('/api/simulator/journal?limit=40')) } catch {} }, [])
  useEffect(() => { pull(); const iv = setInterval(pull, st?.running ? 5000 : 20000); return () => clearInterval(iv) }, [pull, st?.running])
  useEffect(() => { pullJ(); const iv = setInterval(pullJ, 60000); return () => clearInterval(iv) }, [pullJ])

  // Zaman eksenli özsermaye: /api/simulator/equity (30 sn). 404 → st.equity_curve'e geri düş (src bilgisi yok)
  const [eq, setEq] = useState<any>(null)
  const pullE = useCallback(async () => {
    try { setEq(await j('/api/simulator/equity?max_points=2000')) }
    catch (e: any) { if (String(e?.message) === '404') setEq(null) }   // geçici hatada son iyi veri kalsın
  }, [])
  useEffect(() => { pullE(); const iv = setInterval(pullE, 30000); return () => clearInterval(iv) }, [pullE])

  // Sayfalı işlem ucu: TradeTable buradan çeker; toplam sayı sekme etiketine yazılır (bilinmiyorsa yazılmaz)
  const [tradeTotal, setTradeTotal] = useState<number | null>(null)
  const fetchTrades = useCallback(async (page: number, perPage: number): Promise<TradesPage> => {
    const d = await j(`/api/simulator/trades?page=${page}&per_page=${perPage}`)
    if (d?.configured !== false && typeof d?.total === 'number') setTradeTotal(d.total)
    return d
  }, [])
  useEffect(() => {
    const f = () => { fetchTrades(1, 1).catch(() => {}) }   // yalnız toplam için hafif sorgu
    f(); const iv = setInterval(f, 60000); return () => clearInterval(iv)
  }, [fetchTrades])
  const [notesOpen, setNotesOpen] = useState(false)

  const s = st?.stats
  const les = st?.lessons
  const venue = st?.venue
  const eqOk = eq?.configured !== false && Array.isArray(eq?.points) && eq.points.length > 0
  const eqPts = useMemo(() => eqOk ? eq.points : (st?.equity_curve || []), [eqOk, eq, st?.equity_curve])
  const eqMarks = eqOk ? eq.marks : undefined
  const eqCap = eq?.capital ?? s?.capital
  const cnt = tradeTotal != null ? ` (${tradeTotal.toLocaleString('tr-TR')})` : ''

  return (
    <section id="simulator" style={{ borderBottom: `1px solid ${C.border}`, background: '#0B1020' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '8px 16px', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <FlaskConical size={15} color={C.violet} />
          <span style={{ fontWeight: 800, letterSpacing: 1, fontSize: 12 }}>1.000 $ SANAL SİMÜLATÖR — KOMİTE STRATEJİSİ</span>
        </div>
        <Badge text="GARANTİ: YOK" col={C.warn} />
        {st?.configured ? <>
          <Badge text={st.running ? (st.halted ? 'HALT' : 'ÇALIŞIYOR ▶') : 'DURDU ■'} col={st.halted ? C.danger : st.running ? C.neon : C.muted} />
          <Badge text={`${(venue?.exchange_id || '').toUpperCase()} · maker %${((venue?.maker_bps || 0) / 100).toFixed(2)} / taker %${((venue?.taker_bps || 0) / 100).toFixed(2)}`} col={C.cyan}
                 title={venue?.note || 'en düşük komisyonlu borsa seçildi'} />
          <Badge text={`${st.config?.symbols?.length || 0} parite · her ${st.loop_sec} sn`} col={C.muted} />
          <span className="mono" style={{ fontSize: 9, color: C.muted }}>döngü #{st.cycle} · {ago(st.last_cycle_ts)}</span>
        </> : <span style={{ fontSize: 10, color: C.muted }}>{st?.note || 'simülatör yükleniyor…'}</span>}
        <button className="cm-btn" style={{ marginLeft: 'auto', background: C.surface, color: C.muted }} onClick={() => setOpen(o => !o)}>
          {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        </button>
      </div>

      {open && st?.configured && (
        <div style={{ padding: '10px 16px 12px' }}>
          {/* ── özet kutuları ── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(112px, 1fr))', gap: 6 }}>
            <Tile k="ÖZSERMAYE" v={usd(s.equity)} sub={`başlangıç ${usd(s.capital, 0)}`} />
            <Tile k="NET KÂR/ZARAR" v={usd(s.net_pnl)} c={s.net_pnl >= 0 ? C.neon : C.danger} sub={pct(s.return_pct)} />
            <Tile k="ÖDENEN KOMİSYON" v={usd(s.fees_paid)} c={C.warn} icon={<Coins size={10} color={C.warn} />}
                  sub={s.fee_share_of_gross_pct != null ? `brütün %${s.fee_share_of_gross_pct}'i` : `maker payı ${s.maker_share_pct != null ? '%' + s.maker_share_pct : '—'}`} />
            <Tile k="KAZANMA" v={`%${s.win_rate}`} sub={`${s.closed_trades} işlem · p_win ${les?.p_win ?? '—'}`} />
            <Tile k="KÂR FAKTÖRÜ" v={String(s.profit_factor)} c={s.profit_factor >= 1 ? C.neon : C.danger} />
            <Tile k="AÇIK / BEKLEYEN" v={`${s.open_positions} / ${s.pending_orders}`} sub={`maks ${st.config?.max_open}`} />
            <Tile k="GÜN GETİRİSİ" v={pct(st.day_return_pct)} c={(st.day_return_pct || 0) >= 0 ? C.neon : C.danger}
                  sub={`limit −%${Math.round((st.config?.daily_loss_limit_pct || 0) * 1000) / 10}`} />
            <Tile k="DRAWDOWN" v={st.drawdown_pct != null ? `%${st.drawdown_pct}` : '—'} c={C.warn} sub={`maks %${Math.round((st.config?.max_drawdown_pct || 0) * 100)}`} />
            <Tile k="DERSLER" v={String(les?.lessons?.length ?? 0)} sub={`${les?.shadows_open ?? 0} gölge takipte`} icon={<BookOpen size={10} color={C.violet} />} />
            <Tile k="PEAK CAPTURE" v={s.avg_peak_capture != null ? String(s.avg_peak_capture) : '—'} sub="ort. tepe yakalama" />
            <Tile k="KAÇIRILAN / KAÇINILAN" v={`${st.missed?.n_missed ?? 0} / ${st.missed?.n_avoided ?? 0}`} sub={`${st.missed?.n_open ?? 0} gölgede · net %${st.missed?.missed_net_pct_sum ?? 0}`} c={C.warn} icon={<EyeOff size={10} color={C.warn} />} />
            <Tile k="FEE DRAG" v={s.fee_drag != null ? String(s.fee_drag) : '—'} sub={`maker payı ${s.maker_share_pct != null ? '%' + s.maker_share_pct : '—'}`} c={C.warn} />
          </div>

          {/* ── kanıt kapıları (yalnız API gönderirse çizilir) ── */}
          {st.governance && <Governance g={st.governance} />}

          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 1.4fr) minmax(260px, 1fr)', gap: 8, marginTop: 8 }} className="cm-trading-grid">
            <div className="cm-card" style={{ margin: 0, padding: 8, minWidth: 0 }}>
              {/* zaman eksenli, sürüklenebilir pencere: /api/simulator/equity (yoksa st.equity_curve) */}
              <EquityChart points={eqPts} marks={eqMarks} capital={eqCap} color={C.violet} height={170} label="ÖZSERMAYE EĞRİSİ" />
              <div style={{ fontSize: 9, color: C.muted, marginTop: 4, lineHeight: 1.5 }}>
                Gerçek anlık veri, sanal sermaye. Emirler önce <b style={{ color: C.text }}>maker</b> (limit, en iyi teklif) gönderilir;
                dolmazsa ve kenar yeterliyse taker. Komisyon her fişte yazılır.
              </div>
            </div>

            {/* açık pozisyonlar + bekleyen emirler + fiş */}
            <div className="cm-card" style={{ margin: 0, padding: 8 }}>
              <div className="cm-title" style={{ marginBottom: 4 }}>AÇIK POZİSYONLAR & İŞLEM FİŞİ</div>
              {!st.positions?.length && !st.pending?.length && <div style={{ fontSize: 10, color: C.muted }}>açık pozisyon yok — komite bekliyor</div>}
              {st.pending?.map((p: any) => (
                <div key={p.symbol} style={{ fontSize: 10, padding: '3px 0', borderTop: `1px solid ${C.border}`, color: C.warn }}>
                  ⏳ {p.symbol} maker {p.side} {usd(p.notional)} @ {Number(p.price).toPrecision(6)} · {p.bars} bar bekledi
                </div>
              ))}
              {st.positions?.map((p: any) => {
                const t = p.decision?.ticket
                return (
                  <div key={p.symbol} style={{ fontSize: 10, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
                      <b style={{ color: p.direction === 'LONG' ? C.neon : C.danger }}>{p.symbol} {p.direction} <span style={{ color: C.muted, fontWeight: 400 }}>· {p.order_type} · {p.trigger}</span></b>
                      <span className="mono" style={{ color: p.unrealized >= 0 ? C.neon : C.danger }}>{usd(p.unrealized)} ({pct(p.pnl_pct)})</span>
                    </div>
                    <div className="mono" style={{ fontSize: 9, color: C.muted }}>
                      giriş {p.entry} · stop {Number(p.stop).toPrecision(6)}{p.partial_done ? ' (BE)' : ''} · hedef {Number(p.target).toPrecision(6)} · tepe {pct(p.peak_pnl_pct)} · {p.age_min} dk
                    </div>
                    {t && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(96px, 1fr))', gap: 3, marginTop: 3, fontSize: 8.5 }}>
                        <Kv k="yatırım" v={usd(t.investment_usdt)} />
                        <Kv k="beklenen kâr" v={usd(t.expected_profit_usdt)} c={C.neon} />
                        <Kv k="azami zarar" v={usd(t.max_loss_usdt)} c={C.danger} />
                        <Kv k="komisyon" v={usd(t.fee_usdt, 3)} c={C.warn} />
                        <Kv k="kazanma olasılığı" v={`%${Math.round(t.p_win * 100)}`} />
                        <Kv k="beklenen değer" v={usd(t.ev_usdt, 3)} c={t.ev_usdt >= 0 ? C.neon : C.danger} />
                        <Kv k="başabaş kazanma" v={`%${Math.round(t.breakeven_win_rate * 100)}`} />
                        <Kv k="net R/R" v={String(t.rr_net)} />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>

          {/* ── sekmeler ── */}
          <div style={{ display: 'flex', gap: 4, marginTop: 8, flexWrap: 'wrap' }}>
            {([['karar', 'KOMİTE KARARLARI'], ['roller', '12 ROL & GÜVENİLİRLİK'], ['dersler', 'DERSLER'],
               ['evren', 'EVREN (TÜM PARİTELER)'], ['gozardi', 'KAÇIRILANLAR · NEDEN YAPILMADI'], ['arastirma', 'ARAŞTIRMA FABRİKASI (250)'],
               ['video', 'VİDEO KURULUMLARI'], ['raylar', 'RİSK RAYLARI (KALDIRAÇ)'],
               ['islemler', `İŞLEMLER${cnt}`], ['gunluk', `GÜNLÜK${cnt}`]] as const).map(([k, l]) => (
              <button key={k} className="cm-btn" onClick={() => setTab(k)}
                style={{ background: tab === k ? C.violet : C.surface, color: tab === k ? C.bg : C.muted, fontSize: 9, letterSpacing: 0.5 }}>{l}</button>
            ))}
          </div>

          <div className="cm-card" style={{ margin: '6px 0 0', padding: 8 }}>
            {tab === 'karar' && <Decisions d={st.decisions || []} />}
            {tab === 'roller' && <Roles st={st} jr={jr} />}
            {tab === 'dersler' && <Lessons les={les} />}
            {tab === 'evren' && <Universe u={uni} />}
            {tab === 'gozardi' && <Missed m={ms} les={les} />}
            {tab === 'arastirma' && <Research r={rs} />}
            {/* Videolardan çıkarılan kurulumlar: kendi verisini çeker, uç yoksa kendini gizler */}
            {tab === 'video' && <VideoSources />}
            {/* Kaldıraçlı trend rayları: kendi verisini çeker, uç yoksa kendini gizler */}
            {tab === 'raylar' && <RiskTracks />}
            {/* İŞLEMLER ve GÜNLÜK aynı sayfalı tabloyu paylaşır (sekme geçişinde sayfa korunur) */}
            {(tab === 'islemler' || tab === 'gunluk') && (
              <TradeTable fetchPage={fetchTrades} fallback={st.trades} title={tab === 'gunluk' ? 'GÜNLÜK — KAPANAN İŞLEMLER' : 'İŞLEMLER'} />
            )}
            {tab === 'gunluk' && (
              <div style={{ marginTop: 8, borderTop: `1px solid ${C.border}`, paddingTop: 6 }}>
                <button type="button" className="cm-btn" onClick={() => setNotesOpen(o => !o)} aria-expanded={notesOpen}
                        style={{ background: C.surface, color: C.muted, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  {notesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />} GÜNLÜK NOTLARI (markdown)
                </button>
                {notesOpen && <pre style={{ fontSize: 9.5, color: '#8892B0', whiteSpace: 'pre-wrap', maxHeight: 360, overflowY: 'auto', margin: '6px 0 0', fontFamily: 'inherit', lineHeight: 1.5 }}>{jr?.markdown || 'henüz kapanan işlem yok — günlük ilk kapanışta başlar'}</pre>}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

/* ── alt paneller ── */
function Decisions({ d }: { d: any[] }) {
  if (!d.length) return <div style={{ fontSize: 10, color: C.muted }}>ilk döngü bekleniyor…</div>
  return <div>
    {d.map((x: any) => (
      <div key={x.symbol} style={{ fontSize: 9.5, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <b>{x.symbol}</b>
          <span style={{ color: C.muted }}>{x.template === 'pullback' ? 'trend' : 'yatay'}{x.trigger ? ` · ${x.trigger}` : ''}{x.tier === 'light' ? ' · hafif' : ''}{x.exit_mode ? ` · ${x.exit_mode}` : ''}</span>
          {x.competition?.length > 1 && <span className="mono" style={{ fontSize: 8, color: C.muted }}>yarış: {x.competition.map((c: any) => `${c.kind} EV%${c.ev_pct}`).join(' · ')}</span>}
          {x.score != null && <span className="mono" style={{ color: C.muted }}>oy {x.score >= 0 ? '+' : ''}{x.score} · güven {x.confidence}</span>}
          <span style={{ color: String(x.result).startsWith('AÇ') ? C.neon : String(x.result).startsWith('VETO') ? C.danger : String(x.result).startsWith('MAKER') ? C.warn : C.muted }}>{x.result}</span>
        </div>
        {x.veto_review && <div style={{ fontSize: 8.5, color: x.veto_review.decision === 'AÇ' ? C.neon : C.warn, marginTop: 2 }}>
          İNCELEME · {x.veto_review.summary_tr}
        </div>}
        {x.votes?.length > 1 && (
          <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginTop: 3 }}>
            {x.votes.map((v: any, i: number) => {
              const col = v.veto ? C.danger : !v.data_ok ? C.border : v.score > 0.1 ? C.neon : v.score < -0.1 ? C.danger : C.muted
              return <span key={i} title={`${v.title}\n${(v.notes || []).join('\n')}${v.veto ? '\nVETO: ' + v.veto : ''}`}
                style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, cursor: 'help', color: col, border: `1px solid ${col}55` }}>
                {v.role.replace(/_/g, ' ')} {v.veto ? 'VETO' : !v.data_ok ? 'veri yok' : `${v.score >= 0 ? '+' : ''}${v.score}`}{v.size_mult !== 1 && v.data_ok && !v.veto ? ` ×${v.size_mult}` : ''}
              </span>
            })}
          </div>
        )}
        {x.plan && <div className="mono" style={{ fontSize: 8.5, color: C.muted, marginTop: 2 }}>
          stop %{x.plan.stop_pct} ({x.plan.stop_source}) · hedef %{x.plan.target_pct} ({x.plan.target_source}) · R/R {x.plan.rr}</div>}
      </div>
    ))}
  </div>
}

function Roles({ st, jr }: { st: any; jr: any }) {
  const card = st?.strategy_card
  const rel: Record<string, any> = {}
  for (const r of (jr?.roles || st?.lessons?.roles || [])) rel[r.role] = r
  return <div>
    <div style={{ fontSize: 9, color: C.muted, marginBottom: 6, lineHeight: 1.5 }}>
      Ağırlık = taban × (0,5 + güvenilirlik). Güvenilirlik her kapanan işlemde güncellenir (rolün oyu gerçekleşen yönle uyuştu mu, Beta öncülüyle).
      Veto rolleri oylamaya katılmaz; hepsini ezer.
    </div>
    <table style={{ width: '100%', fontSize: 9.5, borderCollapse: 'collapse' }}>
      <thead><tr style={{ color: C.muted, textAlign: 'left' }}><th>rol</th><th>taban</th><th>güvenilirlik</th><th>oy</th><th>etkin</th></tr></thead>
      <tbody>
        {(card?.roles || []).map((r: any) => {
          const x = rel[r.id] || {}
          const rr = x.reliability
          return <tr key={r.id} style={{ borderTop: `1px solid ${C.border}` }}>
            <td style={{ padding: '3px 0' }}><Users size={9} color={r.veto ? C.danger : C.cyan} style={{ verticalAlign: -1 }} /> {r.title}{r.veto && <span style={{ color: C.danger, fontSize: 8 }}> · VETO</span>}</td>
            <td className="mono">{r.veto ? '—' : r.base_weight}</td>
            <td className="mono" style={{ color: rr == null ? C.muted : rr >= 0.6 ? C.neon : rr <= 0.4 ? C.danger : C.text }}>{rr == null ? 'ölçülmedi' : rr.toFixed(2)}</td>
            <td className="mono" style={{ color: C.muted }}>{x.n ?? 0}</td>
            <td className="mono">{r.veto ? '—' : (r.effective_weight ?? r.base_weight)}</td>
          </tr>
        })}
      </tbody>
    </table>
    {card?.measured_caveats && <div style={{ fontSize: 8.5, color: C.muted, marginTop: 6 }}>
      {card.measured_caveats.map((m: string, i: number) => <div key={i}>▸ {m}</div>)}</div>}
  </div>
}

function Lessons({ les }: { les: any }) {
  const L = les?.lessons || []
  const ov = les?.overrides || {}
  return <div>
    {Object.keys(ov).length > 0 && <div style={{ fontSize: 9, marginBottom: 6, color: C.warn }}>
      Öğrenilmiş parametreler: {Object.entries(ov).map(([k, v]: any) => `${k}=${v}`).join(' · ')}</div>}
    {!L.length && <div style={{ fontSize: 10, color: C.muted }}>Henüz ders yok. Ders için kanıt eşiği gerekir (≥ 8 işlem/gölge) — kanıtsız ders yazılmaz.</div>}
    {L.map((l: any, i: number) => (
      <div key={i} style={{ fontSize: 9.5, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
        <span className="mono" style={{ color: C.muted }}>{tstr(l.ts)}</span> 📘 {l.title}
        {l.action && <span className="mono" style={{ color: C.warn }}> → {l.action.param} {l.action.from} → {l.action.to}</span>}
        <div className="mono" style={{ fontSize: 8, color: C.muted }}>kanıt: {Object.entries(l.evidence || {}).filter(([k]) => !String(k).startsWith('_')).map(([k, v]: any) => `${k}=${typeof v === 'number' ? v : JSON.stringify(v)}`).join(' · ')}</div>
      </div>
    ))}
  </div>
}

/* ── KANIT KAPILARI — sleeve kanıt durumu, drawdown/oturum çarpanları, taker politikası (yalnız API gönderirse) ── */
function Governance({ g }: { g: any }) {
  const stateCol: Record<string, string> = { PROVEN: C.neon, UNPROVEN: C.muted, PROBATION: C.warn, PAUSED: C.danger }
  const mult = (m: any) => (m == null || !isFinite(m)) ? '—' : `×${Number(m).toLocaleString('tr-TR', { maximumFractionDigits: 2 })}`
  const dol = (x: any) => (x == null || !isFinite(x)) ? '—'
    : `${Number(x) < 0 ? '−' : ''}${Math.abs(Number(x)).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} $`
  const tst = (t: any) => (t == null || !isFinite(t)) ? '—'
    : Number(t).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).replace(/^-/, '−')
  const sleeves: any[] = Array.isArray(g.sleeves) ? g.sleeves : []
  const sess: [string, any][] = Object.entries(g.session || {})
  return <div className="cm-card" style={{ margin: '8px 0 0', padding: 8 }}>
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
      <div className="cm-title">KANIT KAPILARI</div>
      {g.probe_notional_usdt != null && <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>sonda boyutu {usd(g.probe_notional_usdt)}</span>}
      {g.derisk?.active && <Badge col={C.warn} text={`DRAWDOWN ${mult(g.derisk.mult)} (son ${g.derisk.trailing_n ?? '—'}: ${dol(g.derisk.trailing_net)})`} />}
      {sess.map(([k, v]) => <Badge key={k} col={C.warn} text={`${k} UTC ${mult(v?.mult)} (n${v?.n ?? '—'}, t ${tst(v?.t_stat)})`} />)}
    </div>
    {sleeves.length === 0 && <div style={{ fontSize: 9.5, color: C.muted }}>henüz sleeve ölçümü yok</div>}
    {sleeves.map((r: any) => (
      <div key={r.sleeve} className="mono" style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', fontSize: 9, padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
        <b style={{ minWidth: 130, color: C.text }}>{r.sleeve ?? '—'}</b>
        <Badge text={r.state ?? '—'} col={stateCol[r.state] || C.muted} />
        <span style={{ color: C.muted }}>n {r.n ?? '—'}</span>
        <span style={{ color: (r.net ?? 0) >= 0 ? C.neon : C.danger }}>net {usd(r.net)}</span>
        <span style={{ color: C.muted }}>ort net {pct(r.mean_net_pct, 3)}</span>
        <span style={{ color: C.muted }}>t {tst(r.t_stat)}</span>
        <span style={{ color: C.muted }}>tavan {usd(r.cap_usdt, 0)}</span>
        {r.paused_until && <span style={{ color: C.danger }}>durduruldu → {tstr(r.paused_until)}</span>}
      </div>
    ))}
    {g.taker_policy && <div style={{ fontSize: 8.5, color: C.muted, marginTop: 4 }}>{g.taker_policy}</div>}
  </div>
}

function Badge({ text, col, title }: { text: string; col: string; title?: string }) {
  return <span title={title} style={{ fontSize: 8.5, fontWeight: 800, color: col, border: `1px solid ${col}55`, borderRadius: 4, padding: '1px 6px', whiteSpace: 'nowrap', cursor: title ? 'help' : 'default' }}>{text}</span>
}
function Tile({ k, v, c, sub, icon }: { k: string; v: string; c?: string; sub?: string; icon?: any }) {
  return <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: '6px 8px' }}>
    <div style={{ fontSize: 8, color: C.muted, letterSpacing: 0.6, display: 'flex', gap: 3, alignItems: 'center' }}>{icon}{k}</div>
    <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: c || C.text }}>{v}</div>
    {sub && <div style={{ fontSize: 8, color: C.muted }}>{sub}</div>}
  </div>
}
function Kv({ k, v, c }: { k: string; v: string; c?: string }) {
  return <div style={{ background: C.surface, borderRadius: 4, padding: '2px 5px' }}>
    <div style={{ color: C.muted }}>{k}</div><div className="mono" style={{ color: c || C.text }}>{v}</div></div>
}
// Activity ikonu ileride grafik başlığında kullanılmak üzere içe aktarıldı
void Activity


/* ── KAÇIRILAN FIRSATLAR — neden yapılmadı, ne göz ardı edildi, hangi bilgi eksikti ── */
function Missed({ m, les }: { m: any; les: any }) {
  if (!m) return <div style={{ fontSize: 10, color: C.muted }}>kaçırılan-fırsat raporu yükleniyor…</div>
  const verdictCol = (v: string) => v?.startsWith('ZARARLI') ? C.danger : v?.startsWith('KORUYOR') ? C.neon : C.muted
  return <div style={{ fontSize: 9.5 }}>
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 6, color: C.muted }}>
      <span>kayıt <b style={{ color: C.text }}>{m.n_records}</b></span>
      <span>gölgede <b style={{ color: C.text }}>{m.n_open}</b></span>
      <span>kaçırılan kazanç <b style={{ color: C.warn }}>{m.n_missed}</b> (net toplam %{m.missed_net_pct_sum})</span>
      <span>doğru kaçınma <b style={{ color: C.neon }}>{m.n_avoided}</b></span>
      {m.blind && Object.keys(m.blind).length > 0 && <span>gölgesiz kör nokta: {Object.entries(m.blind).map(([k, v]) => `${k} ${v}`).join(' · ')}</span>}
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 8 }}>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>KAPI İSABETİ — kazananı mı engelliyor, kaybedeni mi?</div>
        {(m.gates || []).length === 0 && <div style={{ color: C.muted }}>henüz veto/eleme kaydı yok</div>}
        {(m.gates || []).map((g: any) => (
          <div key={g.gate} style={{ display: 'flex', gap: 6, flexWrap: 'wrap', padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
            <b style={{ minWidth: 150 }}>{g.gate_tr}</b>
            <span className="mono" style={{ color: C.muted }}>n {g.n} · kaçırılan {g.missed} · kaçınılan {g.avoided} · ufuk {g.timeout} · açık {g.open}</span>
            <span className="mono" style={{ color: verdictCol(g.verdict) }}>{g.verdict}{g.precision != null ? ` · isabet ${Math.round(g.precision * 100)}% [${Math.round(g.wilson[0] * 100)}–${Math.round(g.wilson[1] * 100)}]` : ''}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>GÖZ ARDI EDİLEN ÖZELLİKLER — lift = P(özellik|kaçırılan) / P(özellik|kaçınılan)</div>
        {(m.features || []).length === 0 && <div style={{ color: C.muted }}>ölçüm için ≥ 6 çözülmüş aday gerekir</div>}
        {(m.features || []).slice(0, 10).map((f: any) => (
          <div key={f.feature} style={{ display: 'flex', gap: 6, padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
            <b style={{ minWidth: 170 }}>{f.feature_tr}</b>
            <span className="mono" style={{ color: f.lift >= 1.5 ? C.warn : f.lift <= 0.67 ? C.danger : C.muted }}>lift {f.lift}</span>
            <span className="mono" style={{ color: C.muted }}>{f.in_missed}/{f.n_missed} vs {f.in_avoided}/{f.n_avoided}</span>
            <span style={{ color: C.muted }}>{f.note}</span>
          </div>
        ))}
        <div className="cm-title" style={{ margin: '6px 0 3px' }}>BİLGİ VARLIĞI / YOKLUĞU</div>
        {(m.info || []).length === 0 && <div style={{ color: C.muted }}>ölçülmedi</div>}
        {(m.info || []).map((i: any) => (
          <div key={i.info} style={{ display: 'flex', gap: 6, padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
            <b style={{ minWidth: 170 }}>{i.info_tr}</b>
            <span className="mono" style={{ color: C.muted }}>varken kaçırma {i.missed_rate_with != null ? Math.round(i.missed_rate_with * 100) + '%' : '—'} ({i.n_with}) · yokken {i.missed_rate_without != null ? Math.round(i.missed_rate_without * 100) + '%' : '—'} ({i.n_without})</span>
            <span style={{ color: C.muted }}>{i.note}</span>
          </div>
        ))}
      </div>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 8, marginTop: 8 }}>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>KAZANAN PROFİLİ — hedefe NASIL ulaştılar</div>
        {!m.winner_profile?.n ? <div style={{ color: C.muted }}>henüz hedefe ulaşan yok</div> : <div style={{ fontSize: 9 }}>
          <div className="mono" style={{ color: C.muted }}>n {m.winner_profile.n} · medyan {m.winner_profile.median_minutes} dk · MAE/stop medyan {m.winner_profile.median_mae_over_stop} · stop'a yaklaşan pay {m.winner_profile.near_stop_share}</div>
          <div style={{ color: C.muted }}>kurulum: {Object.entries(m.winner_profile.by_setup || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
          <div style={{ color: C.muted }}>engelleyen kapı: {Object.entries(m.winner_profile.by_gate || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
          <div style={{ color: C.muted }}>göz ardı edilen ortak özellik: {Object.entries(m.winner_profile.common_supportive || {}).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}</div>
          <div style={{ color: C.warn }}>DERS: {m.winner_profile.lesson}</div>
        </div>}
      </div>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>STOP PROFİLİ — hedefe NEDEN ulaşamadılar</div>
        {!m.stop_profile?.n ? <div style={{ color: C.muted }}>henüz stop olan yok</div> : <div style={{ fontSize: 9 }}>
          <div className="mono" style={{ color: C.muted }}>n {m.stop_profile.n} · medyan {m.stop_profile.median_minutes} dk · MFE/hedef medyan {m.stop_profile.median_mfe_over_target} · hedefe yaklaşan pay {m.stop_profile.near_target_share}</div>
          <div style={{ color: C.muted }}>kurulum: {Object.entries(m.stop_profile.by_setup || {}).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
          <div style={{ color: C.muted }}>uyaranlar: {Object.entries(m.stop_profile.common_warnings || {}).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}</div>
          <div style={{ color: C.warn }}>DERS: {m.stop_profile.lesson}</div>
        </div>}
        {m.interest_weights && <div className="mono" style={{ fontSize: 8.5, color: C.muted, marginTop: 4 }}>Tier-A öğrenilmiş ağırlık: dip ×{m.interest_weights.dip} (n {m.interest_weights.dip_n}) · kırılım ×{m.interest_weights.breakout} (n {m.interest_weights.breakout_n})</div>}
      </div>
    </div>
    {m.proposals && Object.keys(m.proposals).length > 0 && (
      <div style={{ marginTop: 6, color: C.warn }}>🥊 challenger önerisi (kanıtlı): {Object.entries(m.proposals).map(([k, v]) => `${k} → ${v}`).join(' · ')}</div>
    )}
    <div className="cm-title" style={{ margin: '8px 0 3px' }}>SON ÇÖZÜLEN ADAYLAR — nasıl ve neden yapılmadı</div>
    {(m.recent || []).length === 0 && <div style={{ color: C.muted }}>henüz çözülen aday yok (ufuk dolmadı)</div>}
    {(m.recent || []).map((r: any) => {
      const a = r.attribution || {}
      const col = r.outcome === 'TARGET' ? C.warn : r.outcome === 'STOP' ? C.neon : C.muted
      return <div key={r.id} style={{ padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          <b>{r.symbol}</b><span style={{ color: C.muted }}>{r.direction} · {r.sleeve || r.kind} · {ago(r.ts)}</span>
          <span style={{ color: col, fontWeight: 700 }}>{a.verdict}</span>
          <span className="mono" style={{ color: C.muted }}>MFE %{r.mfe_pct}{a.net_missed_pct != null ? ` · net kaçırılan %${a.net_missed_pct}` : ''}</span>
        </div>
        {a.path && <div style={{ color: C.text, fontSize: 9 }}>yol: {a.path}</div>}
        <div style={{ color: C.muted, fontSize: 9 }}>{a.how}</div>
      </div>
    })}
    <div style={{ marginTop: 6, fontSize: 8.5, color: C.muted }}>{m.note}</div>
    {les?.veto_stats && <div style={{ marginTop: 4, fontSize: 8.5, color: C.muted }}>ders motoru veto sayacı: {Object.entries(les.veto_stats).map(([k, v]: any) => `${k} ${v.blocked}/${v.would_win}/${v.would_lose}`).join(' · ')}</div>}
  </div>
}

/* ── ARAŞTIRMA FABRİKASI — 250 hipotez, huni, gölge modüller ── */
function Research({ r }: { r: any }) {
  if (!r) return <div style={{ fontSize: 10, color: C.muted }}>araştırma kütüphanesi yükleniyor…</div>
  const lib = r.library || {}
  const st = lib.by_status || {}
  const stCol: any = { IMPLEMENTED: C.neon, SHADOW: C.cyan, RESEARCH: C.muted, DATA_NOT_WIRED: C.warn, NO_DATA: C.danger }
  return <div style={{ fontSize: 9.5 }}>
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 6 }}>
      {Object.entries(st).map(([k, v]: any) => <span key={k} className="mono" style={{ color: stCol[k] || C.muted }}>{k} <b>{v}</b></span>)}
      <span style={{ color: C.muted }}>huni: {(lib.funnel || []).map((f: any) => `${f.stage} ${f.n}`).join(' → ')}</span>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>AİLELER</div>
        {(lib.families || []).map((f: any) => (
          <div key={f.family} style={{ display: 'flex', gap: 6, padding: '1px 0', borderTop: `1px solid ${C.border}` }}>
            <b style={{ minWidth: 190 }}>{f.family} · {f.family_tr}</b>
            <span className="mono" style={{ color: C.muted }}>{f.n} · <span style={{ color: C.neon }}>{f.implemented} kod</span> · <span style={{ color: C.cyan }}>{f.shadow} gölge</span> · <span style={{ color: C.danger }}>{f.no_data} veri yok</span></span>
          </div>
        ))}
      </div>
      <div>
        <div className="cm-title" style={{ marginBottom: 3 }}>GÖLGE MODÜLLER (emir yok)</div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Kointegrasyon çiftleri</b> <span className="mono" style={{ color: C.muted }}>{r.pairs?.pairs?.length ?? 0} çift · açık {r.pairs?.open?.length ?? 0} · kapanan {r.pairs?.n_closed ?? 0} · kazanma {r.pairs?.win_rate ?? '—'}</span>
          {(r.pairs?.pairs || []).slice(0, 3).map((p: any) => <div key={p.a + p.b} className="mono" style={{ fontSize: 8.5, color: C.muted }}>{p.a}–{p.b} β {p.beta_kalman} · ADF {p.adf_t} · z {p.z} · yarı-ömür {p.half_life_bars} bar</div>)}
        </div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Funding carry</b> <span className="mono" style={{ color: C.muted }}>{r.carry?.enabled ? 'açık' : 'kapalı (opt-in)'} · fırsat {r.carry?.opportunities?.length ?? 0} · açık {r.carry?.open?.length ?? 0}{r.carry?.error ? ` · ${r.carry.error}` : ''}</span>
          {(r.carry?.opportunities || []).slice(0, 3).map((o: any) => <div key={o.symbol} className="mono" style={{ fontSize: 8.5, color: C.muted }}>{o.symbol} funding %{o.funding_8h_pct}/8s · net 3g %{o.net_pct} {o.qualifies ? '✓' : ''}</div>)}
        </div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Üçgen arbitraj</b> <span className="mono" style={{ color: C.muted }}>{r.triangular?.n_scans ?? 0} tarama · bulunan {r.triangular?.n_found_total ?? 0}</span>
          {(r.triangular?.recent || []).slice(0, 2).map((t: any, i: number) => <div key={i} className="mono" style={{ fontSize: 8.5, color: C.muted }}>{(t.path || []).join('→')} +{t.net_bps} bps</div>)}
        </div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Avellaneda–Stoikov MM</b>
          {(r.market_making?.rows || []).map((m: any) => <div key={m.symbol} className="mono" style={{ fontSize: 8.5, color: C.muted }}>{m.symbol} dolum {m.n_fills} · P&L {m.pnl_usd} $ · envanter {m.inv_usd} $ · ters seçim {m.adverse_selection_bps_avg ?? '—'} bps</div>)}
          {(r.market_making?.rows || []).length === 0 && <span className="mono" style={{ fontSize: 8.5, color: C.muted }}> henüz kotasyon yok</span>}
        </div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Meta-tahsisçi</b> <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>{(r.allocator?.rows || []).filter((x: any) => x.regime === '*').slice(0, 6).map((x: any) => `${x.sleeve} n${x.n} rel ${x.reliability}${x.measured ? ' ×' + x.weight : ''}`).join(' · ') || 'henüz kapanan işlem yok'}</span></div>
        <div style={{ padding: '2px 0', borderTop: `1px solid ${C.border}` }}><b>Hasat kutusu</b> <span className="mono" style={{ fontSize: 8.5, color: C.muted }}>{r.inbox?.n ?? 0} dosya · {Object.entries(r.inbox?.counts || {}).map(([k, v]) => `${k} ${v}`).join(' · ') || 'boş'}</span></div>
        {r.errors && Object.keys(r.errors).length > 0 && <div style={{ color: C.danger, fontSize: 8.5 }}>hatalar: {Object.entries(r.errors).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div>}
      </div>
    </div>
    <div style={{ marginTop: 6, fontSize: 8.5, color: C.muted }}>{lib.note}</div>
  </div>
}


/* ── EVREN — bütün paritelerin tek tabloda görünürlüğü ── */
function Universe({ u }: { u: any }) {
  if (!u?.rows) return <div style={{ fontSize: 10, color: C.muted }}>evren yükleniyor…</div>
  const fmt = (v: any, d = 2) => (v == null ? '—' : typeof v === 'number' ? v.toFixed(d) : String(v))
  return <div style={{ fontSize: 9 }}>
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 4, color: C.muted }}>
      <span>{u.n} parite</span>
      {u.extra?.length > 0 && <span>ölçülmüş ek pariteler: {u.extra.join(', ')}</span>}
      {u.paused_sleeves && Object.keys(u.paused_sleeves).length > 0 && <span style={{ color: C.warn }}>devre kesici: {Object.keys(u.paused_sleeves).join(', ')}</span>}
      <span>bugün rotasyon {u.rotations_today}</span>
    </div>
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 8.5 }}>
        <thead><tr style={{ color: C.muted, textAlign: 'left' }}>
          {['parite', 'katman', 'tazelik', 'ilgi', 'rejim', 'z', 'RSI', 'trend', 'CVD', 'OBI', 'tetik', 'EV%', 'stop-risk', 'pozisyon', 'son karar / inceleme'].map(h => <th key={h} style={{ padding: '2px 4px', borderBottom: `1px solid ${C.border}` }}>{h}</th>)}
        </tr></thead>
        <tbody>
          {u.rows.map((r: any) => (
            <tr key={r.symbol} style={{ borderBottom: `1px solid ${C.border}22` }}>
              <td style={{ padding: '2px 4px', fontWeight: 700 }}>{r.symbol.replace('/USDT', '')}</td>
              <td style={{ color: C.muted }}>{r.tier === 'heavy' ? 'ağır' : 'hafif'}</td>
              <td style={{ color: r.freshness === 'LIVE' ? C.neon : C.warn }}>{r.freshness || '—'}</td>
              <td className="mono">{fmt(r.interest)}</td>
              <td style={{ color: C.muted }}>{r.regime || '—'}</td>
              <td className="mono">{fmt(r.z)}</td><td className="mono">{fmt(r.rsi, 0)}</td><td className="mono">{fmt(r.trend_score)}</td>
              <td className="mono" style={{ color: (r.cvd || 0) > 0.2 ? C.neon : (r.cvd || 0) < -0.2 ? C.danger : C.muted }}>{fmt(r.cvd)}</td>
              <td className="mono">{fmt(r.obi)}</td>
              <td style={{ color: r.trigger ? C.cyan : C.muted }}>{r.trigger || '—'}</td>
              <td className="mono" style={{ color: (r.ev_pct || 0) > 0 ? C.neon : C.muted }}>{fmt(r.ev_pct, 3)}</td>
              <td className="mono" style={{ color: (r.stop_risk || 0) >= 2 ? C.warn : C.muted }}>{r.stop_risk ?? '—'}</td>
              <td style={{ color: r.position ? C.neon : C.muted }}>{r.position ? `${r.position.sleeve} ${r.position.net_pct}% (tepe ${r.position.peak_net_pct}%, devam p ${r.position.cont_prob ?? '—'}, kalan EV ${r.position.remaining_ev_pct ?? '—'})` : (r.shadows_open ? `${r.shadows_open} gölge` : '—')}</td>
              <td style={{ color: String(r.result).startsWith('AÇ') ? C.neon : String(r.result).startsWith('VETO') ? C.danger : C.muted, maxWidth: 420 }}>{r.result}{r.review ? ` · ${r.review}` : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
}

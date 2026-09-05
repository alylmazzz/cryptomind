import { useState, useEffect, useCallback, useRef } from 'react'
import { Activity, Clock, Zap, TrendingUp, TrendingDown, Gauge, Shield, Target, Wallet, BookOpen, AlertTriangle, Shapes, Waves, BarChart3, ZoomIn, ZoomOut, Maximize2, Radio } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot, ReferenceArea, CartesianGrid, AreaChart, Area } from 'recharts'
import { Net1Section } from './Net1'
import Trading from './Trading'
import Simulator from './Simulator'
import TradeLog from './TradeLog'

// Panel nginx altında /cryptomind/ ile yayımlanır; API aynı köke bağlıdır.
// Vite BASE_URL sondaki bölüyle gelir ('/cryptomind/'), onu kırpıyoruz.
const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')

const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E',
            surface:'#1A2240', bg:'#0A0E1A' }

interface Signal {
  symbol: string; direction: string; bias: string; confidence: number; entry: number
  stop_loss: number; take_profits: number[]; risk_reward: number; timeframe: string
  actionable: boolean; signal_class: string; momentum_score: number; volatility: string
  buy_pressure_pct: number; sell_pressure_pct: number; pressure_label: string
  forecast: any; reasons: string[]; layer_breakdown: any[]
}

const j = async (path: string) => {
  const r = await fetch(`${API}${path}`)
  return await r.json()
}

export default function App() {
  const [signals, setSignals] = useState<Signal[]>([])
  const [meta, setMeta] = useState<any>(null)        // analiz anlık görüntüsü meta
  const [sel, setSel] = useState('BTC/USDT')
  const [tf, setTf] = useState('4h')
  const [chart, setChart] = useState<any>(null)
  const [chartLoading, setChartLoading] = useState(true)
  const [trend, setTrend] = useState<any>(null)
  const [risk, setRisk] = useState<any>(null)
  const [strat, setStrat] = useState<any>(null)
  const [prices, setPrices] = useState<any[]>([])
  const [events, setEvents] = useState<any[]>([])
  const [corr, setCorr] = useState<any>(null)
  const [pat, setPat] = useState<any>(null)                 // /api/patterns yanıtı
  const [offPat, setOffPat] = useState<Set<string>>(new Set())   // kapatılan formasyonlar
  const [harm, setHarm] = useState<any>(null)               // /api/harmonics yanıtı
  const [selHarm, setSelHarm] = useState<string | null>(null)    // seçili harmonik buton
  const [mover, setMover] = useState<any>(null)             // /api/mover yanıtı
  const [social, setSocial] = useState<any>(null)           // /api/social yanıtı
  const [board, setBoard] = useState<any>(null)             // /api/indicators yanıtı
  const [cdl, setCdl] = useState<any>(null)                 // /api/candles yanıtı
  const [clocks, setClocks] = useState<{market:string;flag:string;time:string;open:boolean}[]>([])

  const updateClocks = useCallback(() => {
    const n = new Date()
    const t = (tz:string) => n.toLocaleString('tr-TR',{timeZone:tz,hour:'2-digit',minute:'2-digit',hour12:false})
    const h = (tz:string) => parseInt(n.toLocaleString('en-US',{timeZone:tz,hour:'2-digit',hour12:false}))
    const wd = (tz:string) => { const x = n.toLocaleString('en-US',{timeZone:tz,weekday:'short'}); return x !== 'Sat' && x !== 'Sun' }
    setClocks([
      {market:'NYSE',flag:'US',time:t('America/New_York'),open:wd('America/New_York')&&h('America/New_York')>=9&&h('America/New_York')<16},
      {market:'BIST',flag:'TR',time:t('Europe/Istanbul'),open:wd('Europe/Istanbul')&&h('Europe/Istanbul')>=10&&h('Europe/Istanbul')<18},
      {market:'KRİPTO',flag:'₿',time:t('UTC'),open:true},
    ])
  }, [])

  // analiz anlık görüntüsü — sunucuda arka planda üretilir, burada sadece okunur
  const pullAnalyze = useCallback(async () => {
    try {
      const d = await j('/api/analyze')
      setSignals(d.signals || [])
      setMeta({ pending: !!d.pending, age: d.age_sec, at: d.generated_at,
                refresh: d.refresh_sec, env: d.env })
    } catch { /* ağ hatası: eski veriyi koru */ }
  }, [])

  const pullChart = useCallback(async (symbol: string, timeframe: string) => {
    setChartLoading(true)
    try {
      const d = await j(`/api/chart?symbol=${encodeURIComponent(symbol)}&tf=${timeframe}&bars=160`)
      setChart(d?.error ? null : d)
    } catch { setChart(null) }
    setChartLoading(false)
  }, [])

  const pullPatterns = useCallback(async (symbol: string, timeframe: string) => {
    try {
      const d = await j(`/api/patterns?symbol=${encodeURIComponent(symbol)}&tf=${timeframe}&bars=160&top=5`)
      setPat(d?.error ? null : d)
      setOffPat(new Set())          // yeni parite/TF → hepsi açık
    } catch { setPat(null) }
  }, [])

  const pullHarmonics = useCallback(async (symbol: string, timeframe: string) => {
    try {
      const d = await j(`/api/harmonics?symbol=${encodeURIComponent(symbol)}&tf=${timeframe}&bars=160`)
      setHarm(d?.error ? null : d)
    } catch { setHarm(null) }
  }, [])

  const pullBoard = useCallback(async (symbol: string, timeframe: string) => {
    try {
      const d = await j(`/api/indicators?symbol=${encodeURIComponent(symbol)}&tf=${timeframe}`)
      setBoard(d?.available ? d : null)
    } catch { setBoard(null) }
  }, [])

  const pullCandles = useCallback(async (symbol: string, timeframe: string) => {
    try {
      const d = await j(`/api/candles?symbol=${encodeURIComponent(symbol)}&tf=${timeframe}&lookback=3`)
      setCdl(d?.available ? d : null)
    } catch { setCdl(null) }
  }, [])

  useEffect(() => { updateClocks(); const iv = setInterval(updateClocks, 20000); return () => clearInterval(iv) }, [updateClocks])
  useEffect(() => {
    pullChart(sel, tf); pullPatterns(sel, tf); pullHarmonics(sel, tf)
    pullBoard(sel, tf); pullCandles(sel, tf)
    setSelHarm(null); setBoard(null); setCdl(null)
  }, [sel, tf, pullChart, pullPatterns, pullHarmonics, pullBoard, pullCandles])
  useEffect(() => {
    pullAnalyze(); const iv = setInterval(pullAnalyze, 60000); return () => clearInterval(iv)
  }, [pullAnalyze])
  useEffect(() => {
    const f = async () => {
      try { setTrend(await j('/api/trend')) } catch {}
      try { setRisk(await j('/api/risk')) } catch {}
      try { setPrices((await j('/api/prices')).prices || []) } catch {}
    }
    f(); const iv = setInterval(f, 60000); return () => clearInterval(iv)
  }, [])
  useEffect(() => {
    (async () => {
      try { setStrat(await j('/api/strategy')) } catch {}
      try { setMover(await j('/api/mover')) } catch {}
      try { setSocial(await j('/api/social')) } catch {}
      try { setEvents((await j('/api/events')).calendar || []) } catch {}
      try { setCorr(await j('/api/correlation?tf=1d')) } catch {}
    })()
  }, [])

  const s = signals.find(x => x.symbol === sel)
  const volNames: Record<string,string> = {low:'Düşük',medium:'Orta',high:'Yüksek',extreme:'Aşırı'}

  return (
    <div className="cm-shell">
      {/* ───────────────────────── HEADER ───────────────────────── */}
      <header style={{display:'flex',alignItems:'center',gap:12,flexWrap:'wrap',padding:'8px 16px',
                      background:'#0F1425',borderBottom:`1px solid ${C.border}`}}>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <span style={{width:22,height:22,borderRadius:6,background:'linear-gradient(135deg,#00FF88,#0099FF)',
                        display:'grid',placeItems:'center',color:'#06121F',fontWeight:900,fontSize:13}}>◆</span>
          <span style={{fontWeight:800,letterSpacing:1.5,color:'white',fontFamily:'monospace',fontSize:15}}>
            CRYPTO<span style={{color:C.neon}}>MIND</span>
          </span>
          <span title="Halka açık portföy kâğıt üzerindedir" style={{fontSize:9,color:C.muted,border:`1px solid ${C.border}`,borderRadius:4,padding:'1px 5px'}}>PAPER PORTFÖY</span>
          <a href="#otopilot" title="Borsa API anahtarınızı girin, otopiloti kurun"
            style={{fontSize:9,color:C.info,border:`1px solid ${C.info}55`,
                    borderRadius:4,padding:'2px 6px',textDecoration:'none'}}>
            🔑 BORSA & OTOPİLOT
          </a>
        </div>

        <div className="cm-strip" style={{flex:1,minWidth:220}}>
          {prices.filter(p => p.price != null).map(p => (
            <div key={p.symbol} style={{display:'flex',alignItems:'center',gap:6,padding:'2px 8px',borderRadius:6,
                 background:C.surface,border:`1px solid ${C.border}`,fontSize:11,flexShrink:0}}>
              <span style={{color:C.muted}}>{p.symbol}</span>
              <span className="mono">{Number(p.price).toLocaleString('tr-TR',{maximumFractionDigits:2})}</span>
              {p.change_pct != null && (
                <span className="mono" style={{color:p.change_pct>=0?C.neon:C.danger}}>
                  {p.change_pct>=0?'+':''}{p.change_pct}%
                </span>
              )}
            </div>
          ))}
        </div>

        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          {clocks.map(c => (
            <div key={c.market} style={{display:'flex',alignItems:'center',gap:5,padding:'2px 8px',borderRadius:6,
                 background:C.surface,border:`1px solid ${C.border}`,fontSize:10}} className="mono">
              <span>{c.flag}</span><span style={{color:C.muted}}>{c.market}</span><span>{c.time}</span>
              <span style={{width:6,height:6,borderRadius:'50%',background:c.open?C.neon:C.muted}} />
            </div>
          ))}
        </div>
      </header>

      {/* ── ÜST SIRA: solda borsa bağlantıları + simülatör, SAĞDA yapışkan canlı işlem günlüğü ── */}
      <div className="cm-toprow">
        <div style={{minWidth:0}}>
          {/* ── BORSA BAĞLANTILARI & OTOPİLOT: anahtar girişi + canlı/paper/testnet ── */}
          <Trading />
          {/* ── 1.000 $ SANAL SİMÜLATÖR: komite stratejisi, herkese açık salt-okunur kayıt ── */}
          <Simulator />
        </div>
        <aside className="cm-toplog"><TradeLog /></aside>
      </div>

      {/* ───────────────────────── GRID ───────────────────────── */}
      <div className="cm-grid">
        {/* SOL: izleme listesi + strateji */}
        <div className="cm-col" style={{padding:8}}>
          <div className="cm-title" style={{padding:'2px 4px 6px'}}>İZLEME LİSTESİ</div>
          {signals.length === 0 && <EngineWarming meta={meta} />}
          {signals.map(sig => {
            const isSel = sig.symbol === sel
            const dc = sig.direction==='LONG'?C.neon:sig.direction==='SHORT'?C.danger:C.muted
            return (
              <div key={sig.symbol} onClick={() => setSel(sig.symbol)}
                style={{padding:'7px 9px',cursor:'pointer',borderRadius:6,marginBottom:4,
                  background:isSel?'rgba(0,255,136,0.06)':C.panel,
                  border:`1px solid ${isSel?'rgba(0,255,136,0.35)':C.border}`}}>
                <div style={{display:'flex',justifyContent:'space-between'}}>
                  <span style={{fontWeight:700}}>{sig.symbol}</span>
                  <span style={{fontSize:10,color:C.muted}}>{sig.timeframe}</span>
                </div>
                <div style={{display:'flex',justifyContent:'space-between',marginTop:2}}>
                  <span className="mono" style={{fontSize:12}}>${sig.entry?.toLocaleString('tr-TR')||'-'}</span>
                  <span style={{fontSize:9,fontWeight:700,color:dc}}>
                    {(sig.signal_class||sig.direction).replace(/_/g,' ').toUpperCase()}
                  </span>
                </div>
              </div>
            )
          })}
          <MoverPanel mover={mover} onPick={setSel} />
          <SocialPanel social={social} />
          <HarmonicBar harm={harm} sel={selHarm} setSel={setSelHarm} />
          <PatternPanel pat={pat} off={offPat} setOff={setOffPat} tf={tf} />
          <CandlePanel cdl={cdl} tf={tf} />
          <StrategyCard strat={strat} />
        </div>

        {/* ORTA: grafik + analiz detayı */}
        <div className="cm-col" style={{display:'flex',flexDirection:'column',minWidth:0}}>
          <div style={{padding:'6px 12px',borderBottom:`1px solid ${C.border}`,display:'flex',
                       alignItems:'center',gap:8,flexWrap:'wrap'}}>
            <Activity size={14} color={C.neon} />
            <span style={{fontWeight:700}}>{sel}</span>
            <div style={{display:'flex',gap:4,marginLeft:4}}>
              {['1h','4h','1d','1w'].map(t => (
                <button key={t} className="cm-btn" onClick={() => setTf(t)}
                  style={{background:t===tf?C.neon:C.surface,color:t===tf?'#0A0E1A':C.muted}}>{t}</button>
              ))}
            </div>
            {s && <span className="mono" style={{marginLeft:'auto',fontSize:14,fontWeight:700,
                   color:s.direction==='LONG'?C.neon:s.direction==='SHORT'?C.danger:C.muted}}>
              ${s.entry?.toLocaleString('tr-TR')}</span>}
          </div>

          <ChartView chart={chart} sig={s} loading={chartLoading} pat={pat} offPat={offPat}
                     harm={harm} selHarm={selHarm} />

          <IndicatorBoard board={board} symbol={sel} tf={tf} />

          <div style={{padding:14}}>
            {s ? (
              <div style={{maxWidth:640,margin:'0 auto'}}>
                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10,marginBottom:10}}>
                  <div style={{background:'rgba(0,255,136,0.05)',border:'1px solid rgba(0,255,136,0.22)',borderRadius:8,padding:10}}>
                    <div style={{display:'flex',alignItems:'center',gap:4,color:C.neon,fontSize:10,fontWeight:700,marginBottom:3}}>
                      <TrendingUp size={12}/> AL EŞİĞİ (destek)</div>
                    <div className="mono" style={{color:C.neon,fontSize:14}}>
                      {s.forecast?.buy_threshold?`$${Number(s.forecast.buy_threshold).toLocaleString('tr-TR',{maximumFractionDigits:2})}`:'—'}</div>
                  </div>
                  <div style={{background:'rgba(255,59,92,0.05)',border:'1px solid rgba(255,59,92,0.22)',borderRadius:8,padding:10}}>
                    <div style={{display:'flex',alignItems:'center',gap:4,color:C.danger,fontSize:10,fontWeight:700,marginBottom:3}}>
                      <TrendingDown size={12}/> SAT EŞİĞİ (direnç)</div>
                    <div className="mono" style={{color:C.danger,fontSize:14}}>
                      {s.forecast?.sell_threshold?`$${Number(s.forecast.sell_threshold).toLocaleString('tr-TR',{maximumFractionDigits:2})}`:'—'}</div>
                  </div>
                </div>

                <GatePanel gate={s.forecast?.gate} actionable={s.actionable} />

                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8,marginBottom:10}}>
                  {[[Target,'HEDEF',C.warn,s.take_profits?.[0]],
                    [Shield,'STOP',C.danger,s.stop_loss],
                    [Gauge,'R/R',C.info,null]].map(([Ico,lbl,col,val]:any,i)=>(
                    <div key={i} style={{background:C.surface,borderRadius:6,padding:8}}>
                      <div style={{display:'flex',alignItems:'center',gap:4,color:col,fontSize:10}}><Ico size={11}/> {lbl}</div>
                      <div className="mono" style={{color:col,fontWeight:700,fontSize:13}}>
                        {lbl==='R/R' ? `1:${s.risk_reward?.toFixed(1)||'-'}`
                                     : (val ? `$${Number(val).toLocaleString('tr-TR',{maximumFractionDigits:2})}` : '—')}
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'4px 16px',marginBottom:10,fontSize:10}}>
                  <Row k="Momentum" v={`${s.momentum_score||50}/100`}
                       c={(s.momentum_score||50)>60?C.neon:(s.momentum_score||50)>30?C.warn:C.danger}/>
                  <Row k="Volatilite" v={volNames[s.volatility||'medium']||'-'}/>
                  <Row k="Güven" v={`%${(s.confidence*100).toFixed(0)}`}
                       c={s.direction==='LONG'?C.neon:s.direction==='SHORT'?C.danger:C.muted}/>
                  <Row k="Alıcı/Satıcı" v={`${s.buy_pressure_pct}/${s.sell_pressure_pct}`}/>
                </div>

                <div style={{fontSize:10,color:'#8892B0',lineHeight:1.7}}>
                  {(s.reasons||[]).slice(0,6).map((r,i) => <div key={i}>▸ {r}</div>)}
                </div>

                {meta?.at && (
                  <div style={{marginTop:10,fontSize:9,color:C.muted}}>
                    Anlık görüntü: {meta.at}{meta.age!=null?` · ${Math.round(meta.age/60)} dk önce`:''}
                    {meta.refresh?` · her ${Math.round(meta.refresh/60)} dk'da bir yenilenir`:''}
                  </div>
                )}
              </div>
            ) : <EngineWarming meta={meta} big />}
          </div>
        </div>

        {/* SAĞ: paper portföy + risk */}
        <div className="cm-col" style={{padding:8}}>
          <div style={{display:'flex',alignItems:'center',gap:6,padding:'2px 4px 6px'}}>
            <Zap size={13} color={C.warn} />
            <span className="cm-title">CANLI PAPER PORTFÖY</span>
          </div>
          <TrendPanel trend={trend} />
          <RiskPanel risk={risk} />
        </div>

        {/* ALT: makro takvim + korelasyon (gerçek hesaplama) */}
        <div className="cm-wide" style={{borderTop:`1px solid ${C.border}`,padding:'8px 12px'}}>
          <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:6}}>
            <Clock size={12} color={C.info} />
            <span className="cm-title">MAKRO TAKVİM & KORELASYON</span>
          </div>
          <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
            <div className="cm-strip" style={{flex:'1 1 420px'}}>
              {events.length === 0 && <span style={{fontSize:10,color:C.muted}}>Takvim yükleniyor…</span>}
              {events.slice(0,6).map((ev,i) => (
                <div key={i} style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,
                     padding:'5px 9px',flexShrink:0,minWidth:150}}>
                  <div style={{fontSize:10,fontWeight:700}}>{ev.name}</div>
                  <div className="mono" style={{fontSize:9,color:C.muted}}>{ev.date} · {ev.in_days}g</div>
                  <div style={{fontSize:9,marginTop:2,color:String(ev.impact).includes('çok')?C.danger:C.warn}}>
                    {String(ev.impact).toUpperCase()} ETKİ
                  </div>
                </div>
              ))}
            </div>
            <CorrStrip corr={corr} />
          </div>
        </div>
      </div>

      {/* ── NET +%1 TARAYICI: matris · sabah haritası · kanıt · tahmin kaydı · yönetişim (alt bölüm) ── */}
      <Net1Section />

      {/* ───────────────────────── FOOTER / UYARI ───────────────────────── */}
      <footer style={{borderTop:`1px solid ${C.border}`,background:'#0F1425',padding:'10px 16px',
                      display:'flex',gap:10,alignItems:'flex-start',flexWrap:'wrap'}}>
        <AlertTriangle size={14} color={C.warn} style={{flexShrink:0,marginTop:1}} />
        <div style={{fontSize:10,color:C.muted,lineHeight:1.6,flex:'1 1 420px'}}>
          <b style={{color:C.warn}}>Bu bir yatırım tavsiyesi değildir.</b> Halka açık portföy kâğıt (paper) üzerindedir:
          10.000 USDT sanal sermaye ile yürütülen açık bir kayıttır. Otopilot yalnız <b>sizin</b> girdiğiniz anahtarla,
          sizin oturumunuzda ve açık onayınızla çalışır; para çekme izinli anahtar hiçbir koşulda kabul edilmez, canlı mod
          paper kanıtı + operatör kapısı + onay cümlesi ister. Geçmiş performans gelecek getiriyi garanti etmez;
          kripto ve türev piyasalarda sermayenizin tamamını kaybedebilirsiniz.
        </div>
        <div style={{fontSize:10,color:C.muted}}>
          <span style={{color:C.text}}>MindCorp Lab</span> · veri: Binance/Yahoo Finance (halka açık)
        </div>
      </footer>
    </div>
  )
}

// ── küçük yardımcılar ────────────────────────────────────────────────────
function Row({ k, v, c }: { k: string; v: string; c?: string }) {
  return (
    <div style={{display:'flex',justifyContent:'space-between'}}>
      <span style={{color:C.muted}}>{k}</span>
      <span className="mono" style={{color:c||C.text,fontWeight:700}}>{v}</span>
    </div>
  )
}

function EngineWarming({ meta, big }: { meta: any; big?: boolean }) {
  const txt = meta?.pending
    ? 'Analiz motoru ilk anlık görüntüyü hazırlıyor (~1 dk). Sayfa kendini yeniler.'
    : 'Analiz verisi bekleniyor…'
  return (
    <div className="anim-pulse" style={{padding:big?24:10,textAlign:'center',color:C.muted,
         fontSize:big?12:10,border:`1px dashed ${C.border}`,borderRadius:8}}>
      {txt}
    </div>
  )
}

// ── fiyat grafiği: kapanış + trend çizgileri + S/R + FORMASYON GEOMETRİSİ ──
function ChartView({ chart, sig, loading, pat, offPat, harm, selHarm }:
    { chart: any; sig: any; loading: boolean; pat: any; offPat: Set<string>
      harm: any; selHarm: string | null }) {
  if (!chart || !chart.candles?.length) {
    return <div style={{height:340,display:'flex',alignItems:'center',justifyContent:'center',
                        color:C.muted,fontSize:11,borderBottom:`1px solid ${C.border}`}}
                className={loading ? 'anim-pulse' : ''}>
      {loading ? 'Grafik yükleniyor…' : 'Grafik verisi alınamadı'}
    </div>
  }
  const data = chart.candles.map((c: any, i: number) => ({ i, c: c.c }))
  const nBars = data.length
  const tls = chart.trendlines?.trendlines || []
  const horiz = (chart.trendlines?.horizontals || []).slice(0, 3)
  const tps: number[] = sig?.take_profits || chart.levels?.take_profits || []
  const lineColor = sig?.direction === 'SHORT' ? C.danger : C.neon

  // ── YAKINLAŞTIRMA / ALAN SEÇİMİ ──────────────────────────────────────────
  // Sol tıkla sürükleyerek alan seç → bırakınca o aralığa yakınlaş.
  // Tekerlek imlecin bulunduğu noktaya doğru yakınlaştırır.
  const [zoom, setZoom] = useState<{ l: number; r: number } | null>(null)
  const [drag, setDrag] = useState<{ a: number | null; b: number | null }>({ a: null, b: null })

  const shown = (pat?.patterns || []).filter((p: any) => !offPat.has(p.key))
  // Harmonik yalnız BUTONA BASILINCA çizilir — hepsi birden çizilirse grafik okunmaz.
  const harmShown = selHarm
    ? (harm?.patterns || []).filter((p: any) => p.key === selHarm)
    : []
  // projekte D barı sağa taşabilir; alanı ona göre genişlet
  const maxX = harmShown.reduce((m: number, p: any) =>
    Math.max(m, ...(p.points || []).map((pt: any) => pt.i)), nBars - 1)
  const clamp = (x: number) => Math.max(0, Math.min(maxX, x))

  // görünür x aralığı
  const vl = zoom ? Math.max(0, zoom.l) : 0
  const vr = zoom ? Math.min(maxX, zoom.r) : maxX
  // görünür veriye göre y aralığı (yakınlaşınca dikey de otomatik sığar)
  const vis = data.filter((d: any) => d.i >= vl && d.i <= vr).map((d: any) => d.c)
  const yPad = vis.length ? (Math.max(...vis) - Math.min(...vis)) * 0.08 || 1 : 0
  const yDomain: [any, any] = vis.length && zoom
    ? [Math.min(...vis) - yPad, Math.max(...vis) + yPad]
    : ['auto', 'auto']

  // React `onWheel`'i PASİF dinleyici olarak ekler → preventDefault çalışmaz ve
  // tekerlekle yakınlaşırken sayfa da kayar. Bu yüzden dinleyici elle,
  // { passive: false } ile bağlanır.
  const wheelRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = wheelRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const box = el.getBoundingClientRect()
      const frac = Math.min(1, Math.max(0, (e.clientX - box.left - 58) / (box.width - 58 - 96)))
      const anchor = vl + (vr - vl) * frac
      const f = e.deltaY < 0 ? 0.75 : 1 / 0.75
      const nl = anchor - (anchor - vl) * f
      const nr = anchor + (vr - anchor) * f
      if (nr - nl < 6) return
      if (nl <= 0 && nr >= maxX) { setZoom(null); return }
      setZoom({ l: Math.max(0, Math.round(nl)), r: Math.min(maxX, Math.round(nr)) })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [vl, vr, maxX])

  const applyZoom = (a: number, b: number) => {
    const lo = Math.round(Math.min(a, b)), hi = Math.round(Math.max(a, b))
    if (hi - lo < 4) return                      // kazara tıklama → yoksay
    setZoom({ l: Math.max(0, lo), r: Math.min(maxX, hi) })
  }
  const zoomBy = (factor: number) => {
    const c = (vl + vr) / 2
    const half = Math.max(3, ((vr - vl) / 2) * factor)
    setZoom({ l: Math.max(0, Math.round(c - half)), r: Math.min(maxX, Math.round(c + half)) })
  }

  return (
    <div style={{padding:'8px 10px 2px',borderBottom:`1px solid ${C.border}`,
                 position:'sticky', top:0, zIndex:5, background:C.bg}}>
      {/* araç çubuğu */}
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:4}}>
        <span style={{fontSize:9,color:C.muted}}>
          {zoom ? `bar ${vl}–${vr} (${vr - vl + 1})` : `tüm aralık (${nBars} bar)`}
        </span>
        <span style={{fontSize:9,color:C.muted,marginLeft:6}}>
          sol tıkla sürükle = alan seç · tekerlek = yakınlaştır
        </span>
        <div style={{marginLeft:'auto',display:'flex',gap:4}}>
          <button className="cm-btn" onClick={()=>zoomBy(0.6)} title="yakınlaştır"
            style={{background:C.surface,color:C.text}}><ZoomIn size={11}/></button>
          <button className="cm-btn" onClick={()=>zoomBy(1/0.6)} title="uzaklaştır"
            style={{background:C.surface,color:C.text}}><ZoomOut size={11}/></button>
          <button className="cm-btn" onClick={()=>setZoom(null)} disabled={!zoom}
            title="tümünü göster"
            style={{background: zoom ? C.neon : C.surface, color: zoom ? '#0A0E1A' : C.muted}}>
            <Maximize2 size={11}/>
          </button>
        </div>
      </div>

      <div ref={wheelRef} style={{height:300, userSelect:'none'}}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 96, bottom: 0, left: 0 }}
          onMouseDown={(e: any) => e?.activeLabel != null && setDrag({ a: +e.activeLabel, b: null })}
          onMouseMove={(e: any) => drag.a != null && e?.activeLabel != null &&
                                   setDrag(d => ({ ...d, b: +e.activeLabel }))}
          onMouseUp={() => {
            if (drag.a != null && drag.b != null) applyZoom(drag.a, drag.b)
            setDrag({ a: null, b: null })
          }}
          onMouseLeave={() => setDrag({ a: null, b: null })}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,42,69,0.35)" />
          <XAxis dataKey="i" type="number" domain={[vl, vr]} allowDataOverflow hide />
          <YAxis domain={yDomain} allowDataOverflow tick={{ fill:C.muted, fontSize:9 }} width={58}
                 tickFormatter={(v:number)=>v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(2)} />

          {/* sürüklerken seçili alan */}
          {drag.a != null && drag.b != null && (
            <ReferenceArea x1={Math.min(drag.a, drag.b)} x2={Math.max(drag.a, drag.b)}
              fill={C.info} fillOpacity={0.18} stroke={C.info} strokeOpacity={0.6} />
          )}
          <Tooltip contentStyle={{ background:C.panel, border:`1px solid ${C.border}`, borderRadius:8, fontSize:11 }}
                   labelFormatter={()=>''} formatter={(v:any)=>[`$${Number(v).toLocaleString('tr-TR')}`,'fiyat']} />

          {/* ── FORMASYON GÖLGE ALANLARI (en altta) ── */}
          {shown.map((p: any) => (
            <ReferenceArea key={'z'+p.key} x1={clamp(p.start_i)} x2={clamp(p.end_i)}
              fill={p.color} fillOpacity={0.07} stroke="none" ifOverflow="hidden" />
          ))}

          <Line type="monotone" dataKey="c" stroke={lineColor} dot={false} strokeWidth={1.7} />

          {/* otomatik trend çizgileri (formasyondan bağımsız) */}
          {tls.map((tl: any, k: number) => (
            <ReferenceLine key={'tl'+k} ifOverflow="extendDomain"
              segment={[{ x: tl.x0, y: tl.y0 }, { x: tl.x1, y: tl.y1 }]}
              stroke="rgba(107,115,148,0.55)" strokeWidth={1} strokeDasharray="4 3" />
          ))}
          {horiz.map((h: any, k: number) => (
            <ReferenceLine key={'h'+k} y={h.price} stroke="rgba(124,58,237,0.35)" strokeDasharray="2 4" />
          ))}

          {/* ── FORMASYON ÇİZGİLERİ ── */}
          {shown.flatMap((p: any) =>
            (p.lines || []).map((ln: any, k: number) => (
              <ReferenceLine key={`pl-${p.key}-${k}`} ifOverflow="extendDomain"
                segment={[{ x: clamp(ln.x0), y: ln.y0 }, { x: clamp(ln.x1), y: ln.y1 }]}
                stroke={p.color} strokeWidth={2}
                strokeDasharray={ln.dashed ? '6 3' : undefined} />
            )))}

          {/* fincan eğrisi — ardışık kısa parçalarla */}
          {shown.flatMap((p: any) =>
            (p.curve || []).slice(0, -1).map((c0: any, k: number) => (
              <ReferenceLine key={`pc-${p.key}-${k}`} ifOverflow="hidden"
                segment={[{ x: clamp(c0.x), y: c0.y },
                          { x: clamp(p.curve[k + 1].x), y: p.curve[k + 1].y }]}
                stroke={p.color} strokeWidth={2} />
            )))}

          {/* ── ÇAPA NOKTALARI + ETİKETLERİ (Sol Omuz / Baş / Sağ Omuz…) ── */}
          {shown.flatMap((p: any) =>
            (p.points || []).map((pt: any, k: number) => (
              <ReferenceDot key={`pd-${p.key}-${k}`} x={clamp(pt.i)} y={pt.price}
                r={3.5} fill={p.color} stroke="#0A0E1A" strokeWidth={1} ifOverflow="hidden"
                label={{ value: pt.label, position: 'top', fill: p.color,
                         fontSize: 9, fontWeight: 700 }} />
            )))}

          {/* ── KIRILIM ve HEDEF seviyeleri ── */}
          {shown.map((p: any) => (
            <ReferenceLine key={'bo'+p.key} y={p.breakout} stroke={p.color}
              strokeDasharray="2 3" strokeOpacity={0.55} />
          ))}
          {shown.map((p: any) => (
            <ReferenceLine key={'tg'+p.key} y={p.target} stroke={p.color} strokeWidth={1.5}
              label={{ value: `${p.name}  ${p.target_pct >= 0 ? '+' : ''}${p.target_pct}%`,
                       position: 'right', fill: p.color, fontSize: 9, fontWeight: 700 }} />
          ))}

          {/* ── HARMONİK XABCD — neon bacaklar + oran etiketleri ── */}
          {harmShown.flatMap((p: any) => (p.legs || []).map((lg: any, k: number) => ([
            // neon parlama (kalın, saydam) + üstüne keskin çizgi
            <ReferenceLine key={`hg-${p.key}-${k}`} ifOverflow="extendDomain"
              segment={[{ x: clamp(lg.x0), y: lg.y0 }, { x: clamp(lg.x1), y: lg.y1 }]}
              stroke={p.color} strokeWidth={7} strokeOpacity={0.16} />,
            <ReferenceLine key={`hl-${p.key}-${k}`} ifOverflow="extendDomain"
              segment={[{ x: clamp(lg.x0), y: lg.y0 }, { x: clamp(lg.x1), y: lg.y1 }]}
              stroke={p.color} strokeWidth={2.2}
              strokeDasharray={lg.projected ? '7 4' : undefined}
              label={{ value: lg.ratio_name === 'XA (baz)' ? 'XA'
                              : `${lg.ratio_name} ${lg.projected ? lg.ideal : lg.ratio}`,
                       position: 'center', fill: p.color, fontSize: 8, fontWeight: 700 }} />,
          ])).flat())}

          {/* PRZ — D'nin beklendiği bölge */}
          {harmShown.filter((p: any) => p.prz).map((p: any) => (
            <ReferenceArea key={`przz-${p.key}`}
              x1={clamp(p.prz.i_from)} x2={clamp(p.prz.i)}
              y1={p.prz.lo} y2={p.prz.hi}
              fill={p.color} fillOpacity={0.16} stroke={p.color} strokeOpacity={0.5}
              strokeDasharray="4 3" ifOverflow="extendDomain"
              label={{ value: 'PRZ', position: 'insideTopRight', fill: p.color,
                       fontSize: 9, fontWeight: 700 }} />
          ))}

          {/* X A B C D noktaları */}
          {harmShown.flatMap((p: any) =>
            (p.points || []).map((pt: any, k: number) => (
              <ReferenceDot key={`hd-${p.key}-${k}`} x={clamp(pt.i)} y={pt.price}
                r={pt.projected ? 5 : 4.5} fill={pt.projected ? '#0A0E1A' : p.color}
                stroke={p.color} strokeWidth={2} ifOverflow="extendDomain"
                label={{ value: pt.label, position: 'top', fill: p.color,
                         fontSize: 11, fontWeight: 800 }} />
            )))}

          {/* harmonik işlem seviyeleri */}
          {harmShown.map((p: any) => (
            <ReferenceLine key={`he-${p.key}`} y={p.entry} stroke={p.color}
              strokeDasharray="2 2" strokeOpacity={0.7}
              label={{ value: `giriş ${p.entry_pct >= 0 ? '+' : ''}${p.entry_pct}%`,
                       position: 'right', fill: p.color, fontSize: 9, fontWeight: 700 }} />
          ))}
          {harmShown.flatMap((p: any) => (p.targets || []).slice(0, 2).map((t: number, k: number) => (
            <ReferenceLine key={`ht-${p.key}-${k}`} y={t} stroke={p.color}
              strokeDasharray="1 5" strokeOpacity={0.5}
              label={{ value: `H${k + 1} ${p.target_pcts[k] >= 0 ? '+' : ''}${p.target_pcts[k]}%`,
                       position: 'right', fill: p.color, fontSize: 8 }} />
          )))}

          {/* ── KULLANICI HEDEFİ: ±%1 seviyeleri ── */}
          {pat?.one_pct_levels && (<>
            <ReferenceLine y={pat.one_pct_levels.up} stroke="#E2E8F0" strokeDasharray="1 4"
              strokeOpacity={0.5}
              label={{ value:'+%1', position:'right', fill:'#E2E8F0', fontSize:8 }} />
            <ReferenceLine y={pat.one_pct_levels.down} stroke="#E2E8F0" strokeDasharray="1 4"
              strokeOpacity={0.5}
              label={{ value:'−%1', position:'right', fill:'#E2E8F0', fontSize:8 }} />
          </>)}

          {/* işlem seviyeleri */}
          {sig?.entry && <ReferenceLine y={sig.entry} stroke={C.text} strokeOpacity={0.7}
            label={{ value:'giriş', fill:C.text, fontSize:8, position:'insideRight' }} />}
          {sig?.stop_loss && <ReferenceLine y={sig.stop_loss} stroke={C.danger} strokeOpacity={0.7}
            label={{ value:'SL', fill:C.danger, fontSize:8, position:'insideRight' }} />}
          {tps.slice(0,1).map((t: number, k: number) => (
            <ReferenceLine key={'tp'+k} y={t} stroke={C.warn} strokeDasharray="3 3" strokeOpacity={0.7}
              label={{ value:'TP', fill:C.warn, fontSize:8, position:'insideRight' }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  )
}

// ── paper portföy: equity eğrisi + pozisyonlar ──────────────────────────
function TrendPanel({ trend }: { trend: any }) {
  if (!trend) return null
  if (!trend.available) {
    return <div className="cm-card" style={{fontSize:10,color:C.muted}}>
      Paper portföy durumu henüz oluşmadı ({trend.reason||''}).
    </div>
  }
  const ret = trend.return_pct || 0
  const col = ret >= 0 ? C.neon : C.danger
  const curve = (trend.equity_curve||[]).filter((p:any) => p.equity != null)
  return (
    <div className="cm-card" style={{background:'#0F1A14',border:'1px solid rgba(0,255,136,0.25)'}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{display:'flex',alignItems:'center',gap:5,fontSize:10,fontWeight:700,color:C.neon,letterSpacing:1}}>
          <Wallet size={12}/> TREND PORTFÖY
        </span>
        <span className="mono" style={{fontSize:9,color:C.muted}}>{trend.last_rebalance||''}</span>
      </div>

      <div style={{display:'flex',gap:16,marginBottom:8,flexWrap:'wrap'}}>
        <div><div style={{fontSize:9,color:C.muted}}>Sermaye</div>
          <div className="mono" style={{fontSize:15,fontWeight:700}}>
            ${(trend.equity||0).toLocaleString('tr-TR',{maximumFractionDigits:2})}</div></div>
        <div><div style={{fontSize:9,color:C.muted}}>Getiri</div>
          <div className="mono" style={{fontSize:15,fontWeight:700,color:col}}>{ret>=0?'+':''}{ret.toFixed(2)}%</div></div>
        <div><div style={{fontSize:9,color:C.muted}}>Yatırılan</div>
          <div className="mono" style={{fontSize:15,fontWeight:700}}>%{trend.invested_pct||0}</div></div>
        <div><div style={{fontSize:9,color:C.muted}}>Gün</div>
          <div className="mono" style={{fontSize:15,fontWeight:700}}>{trend.days||0}</div></div>
      </div>

      {curve.length > 1 && (
        <div style={{height:110,marginBottom:8}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={curve} margin={{top:4,right:4,bottom:0,left:0}}>
              <defs>
                <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={col} stopOpacity={0.35}/>
                  <stop offset="100%" stopColor={col} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <YAxis domain={['dataMin','dataMax']} hide />
              <XAxis dataKey="date" hide />
              <Tooltip contentStyle={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,fontSize:11}}
                       formatter={(v:any)=>[`$${Number(v).toLocaleString('tr-TR',{maximumFractionDigits:2})}`,'sermaye']} />
              <ReferenceLine y={trend.initial} stroke={C.muted} strokeDasharray="3 3" />
              <Area type="monotone" dataKey="equity" stroke={col} strokeWidth={1.6} fill="url(#eq)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {trend.cash ? (
        <div style={{fontSize:10,color:C.warn,background:'rgba(255,182,39,0.08)',borderRadius:4,padding:'5px 7px'}}>
          🛡️ %100 NAKİT — izlenen varlıkların hiçbiri trend filtresini geçmiyor. Strateji sermayeyi koruyor.
        </div>
      ) : (
        <div style={{display:'flex',flexWrap:'wrap',gap:4}}>
          {(trend.in_market||[]).map((p:string) => (
            <span key={p} className="mono" style={{fontSize:9,color:C.neon,background:'rgba(0,255,136,0.1)',
                  borderRadius:3,padding:'2px 6px'}}>
              🟢 {p} %{((trend.weights?.[p]||0)*100).toFixed(1)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ── risk & sağlık (drift, VaR/CVaR, maruziyet) ──────────────────────────
function RiskPanel({ risk }: { risk: any }) {
  if (!risk || !risk.available) return null
  const hc: Record<string,string> = { green:C.neon, yellow:C.warn, red:C.danger }
  const col = hc[risk.health] || C.muted
  return (
    <div className="cm-card" style={{border:`1px solid ${col}55`}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
        <span className="cm-title">RİSK & SAĞLIK</span>
        <span style={{fontSize:9,fontWeight:700,color:col}}>● {(risk.health||'').toUpperCase()}</span>
      </div>
      {risk.samples >= 20 ? (
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'3px 14px',fontSize:9}}>
          <Row k="Gerçekleşen Sharpe" v={`${risk.realized_sharpe}`} c={risk.realized_sharpe>=1?C.neon:C.warn}/>
          <Row k="Beklenen (OOS)" v={`${risk.expected_sharpe}`}/>
          <Row k="Max düşüş" v={`%${risk.max_drawdown_pct}`}/>
          <Row k="Güncel düşüş" v={`%${risk.current_drawdown_pct}`}/>
          <Row k="VaR %95" v={`%${risk.var95_pct}`} c={C.warn}/>
          <Row k="CVaR %95" v={`%${risk.cvar95_pct}`} c={C.danger}/>
          <Row k="Yıllık vol" v={`%${risk.ann_vol_pct}`}/>
          <Row k="Maruziyet" v={`%${risk.gross_exposure_pct}`}/>
        </div>
      ) : (
        <div style={{fontSize:9,color:C.muted}}>
          Metrikler için ≥20 gün geçmiş gerekli ({risk.samples} gün var)
        </div>
      )}
      {(risk.alerts||[]).slice(0,3).map((a:string,i:number) => (
        <div key={i} style={{fontSize:9,color:col,marginTop:4}}>{a}</div>
      ))}
    </div>
  )
}

// ── GÖSTERGE TABLOSU — 129 göstergenin AL/SAT/NÖTR dağılımı ─────────────
// Grafiğin ALTINDA durur. İki sayım birden gösterir: ham (kullanıcının istediği
// "kaçı al kaçı sat") ve AİLE (birbirini tekrar edenler tek oya indirgenmiş).
// Ölçülmüş kanıt panelin içinde: bu konsensüsü takip etmek para kaybettirdi.
function IndicatorBoard({ board, symbol, tf }:
  { board: any; symbol: string; tf: string }) {
  const [open, setOpen] = useState(false)
  const [cat, setCat] = useState<string | null>(null)

  if (!board) return (
    <div style={{padding:'10px 14px',borderTop:`1px solid ${C.border}`,
                 fontSize:10,color:C.muted}}>
      Gösterge tablosu yükleniyor… ({symbol} · {tf})
    </div>
  )

  const raw = board.raw || {}, fam = board.family || {}
  const ev = board.evidence || {}
  const total = board.total || 0
  const bias = board.bias || 'NÖTR'
  const biasCol = bias === 'YUKARI' ? C.neon : bias === 'AŞAĞI' ? C.danger : C.muted
  const cats: any[] = Object.entries(board.by_category || {})
    .map(([k, v]: any) => ({ key: k, ...v }))
    .sort((a, b) => b.n - a.n)
  const sigCol = (s: string) => s === 'AL' ? C.neon : s === 'SAT' ? C.danger : C.muted
  const shown: any[] = (board.indicators || [])
    .filter((i: any) => !cat || i.category === cat)

  // Yüzde barı — ham sayım üzerinden (kullanıcının istediği görünüm)
  const pct = (n: number) => total ? (n / total * 100) : 0

  return (
    <div style={{borderTop:`1px solid ${C.border}`,background:C.panel}}>
      <div style={{padding:'10px 14px'}}>
        <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:8,flexWrap:'wrap'}}>
          <Gauge size={12} color={biasCol} />
          <span className="cm-title">GÖSTERGE TABLOSU</span>
          <span style={{fontSize:9,color:C.muted}}>{symbol} · {tf} · {total} gösterge</span>
          <span style={{marginLeft:'auto',fontSize:11,fontWeight:800,color:biasCol}}>
            EĞİLİM {bias}
          </span>
        </div>

        {/* ── HAM SAYIM: kaçı AL, kaçı SAT, kaçı NÖTR ── */}
        <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8,marginBottom:8}}>
          {[['AL', raw.al, C.neon], ['SAT', raw.sat, C.danger], ['NÖTR', raw.notr, C.muted]]
            .map(([lbl, n, col]: any) => (
            <div key={lbl} style={{background:C.surface,borderRadius:6,padding:'8px 10px',
                                   borderLeft:`3px solid ${col}`}}>
              <div style={{fontSize:9,color:col,fontWeight:700}}>{lbl}</div>
              <div className="mono" style={{fontSize:20,fontWeight:800,color:col,lineHeight:1.1}}>
                {n ?? 0}
              </div>
              <div style={{fontSize:9,color:C.muted}}>%{pct(n ?? 0).toFixed(0)}</div>
            </div>
          ))}
        </div>

        {/* yığılmış bar */}
        <div style={{display:'flex',height:7,borderRadius:4,overflow:'hidden',marginBottom:8}}>
          <div style={{width:`${pct(raw.al ?? 0)}%`,background:C.neon}} />
          <div style={{width:`${pct(raw.notr ?? 0)}%`,background:'#2A3355'}} />
          <div style={{width:`${pct(raw.sat ?? 0)}%`,background:C.danger}} />
        </div>

        {/* ── AİLE SAYIMI — asıl karar sayısı ── */}
        <div style={{background:'rgba(255,182,39,0.06)',border:`1px solid ${C.warn}44`,
                     borderRadius:6,padding:'7px 9px',marginBottom:8}}>
          <div style={{fontSize:9,color:C.warn,fontWeight:700,marginBottom:3}}>
            ⚠ BAĞIMSIZ OY: AL {fam.al} · SAT {fam.sat} · NÖTR {fam.notr} (toplam {fam.total})
          </div>
          <div style={{fontSize:9,color:C.muted,lineHeight:1.45}}>
            {board.redundancy_note}
          </div>
        </div>

        {/* ── kategori kırılımı ── */}
        <div style={{display:'flex',gap:5,flexWrap:'wrap',marginBottom:6}}>
          <button onClick={() => setCat(null)} style={{
            background: cat === null ? C.surface : 'transparent',
            border:`1px solid ${cat === null ? C.info : C.border}`,color:cat===null?C.text:C.muted,
            borderRadius:5,padding:'3px 7px',fontSize:9,cursor:'pointer'}}>TÜMÜ</button>
          {cats.map(c2 => {
            const mikro = c2.key === 'microstructure'
            return (
              <button key={c2.key} onClick={() => setCat(cat === c2.key ? null : c2.key)}
                title={mikro ? 'Borsa defteri ve türev verisinden gelir — fiyattan TÜRETİLMEZ'
                             : undefined}
                style={{
                  background: cat === c2.key ? C.surface : (mikro ? 'rgba(34,211,238,0.08)' : 'transparent'),
                  border:`1px solid ${cat === c2.key ? C.info : (mikro ? '#22D3EE66' : C.border)}`,
                  color: cat === c2.key ? C.text : (mikro ? '#22D3EE' : C.muted),
                  borderRadius:5,padding:'3px 7px',fontSize:9,cursor:'pointer',
                  fontWeight: mikro ? 700 : 400}}>
                {mikro && '◈ '}{c2.label}
                <span style={{color:C.neon,marginLeft:4}}>{c2.al}</span>
                <span style={{color:C.muted}}>/</span>
                <span style={{color:C.danger}}>{c2.sat}</span>
                <span style={{color:C.muted}}>/{c2.notr}</span>
              </button>
            )
          })}
        </div>

        {/* Mikroyapı katmanı — tablodaki tek fiyattan-türetilmeyen sınıf */}
        {board.microstructure !== undefined && (
          <div style={{fontSize:9,lineHeight:1.5,marginBottom:6,padding:'5px 8px',
                       borderRadius:5,
                       background: board.microstructure ? 'rgba(34,211,238,0.06)' : 'transparent',
                       border:`1px solid ${board.microstructure ? '#22D3EE44' : C.border}`,
                       color:C.muted}}>
            <b style={{color: board.microstructure ? '#22D3EE' : C.warn}}>
              ◈ MİKROYAPI {board.microstructure ? 'AÇIK' : 'KAPALI'}
            </b>{' — '}{board.microstructure_note}
            {board.microstructure && (
              <span> Bunlar borsa <b>defterinden</b> (derinlik eğrisi, spread,
                dengesizlik) ve <b>türev</b> verisinden (funding, açık pozisyon,
                pozisyonlanma) gelir; diğer {board.total - (board.by_category?.microstructure?.n || 0)}
                {' '}gösterge aynı fiyat serisinin türevleridir.</span>
            )}
          </div>
        )}

        {/* Mikroyapı AYRICA ölçüldü — sonucu fiyat türevlerinden farklı çıktı */}
        {board.micro_evidence && (
          <div style={{background:'rgba(34,211,238,0.05)',border:`1px solid #22D3EE44`,
                       borderRadius:6,padding:'7px 9px',marginBottom:8}}>
            <div style={{fontSize:9,fontWeight:800,color:'#22D3EE',marginBottom:3}}>
              ◈ MİKROYAPI AYRICA ÖLÇÜLDÜ — {board.micro_evidence.headline}
            </div>
            <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginBottom:4}}>
              {board.micro_evidence.differs_from_price}
            </div>
            <table style={{width:'100%',fontSize:9,borderCollapse:'collapse',marginBottom:4}}>
              <thead><tr style={{color:C.muted}}>
                <th style={{textAlign:'left',padding:'1px 4px'}}>sinyal</th>
                <th style={{textAlign:'right',padding:'1px 4px'}}>brüt</th>
                <th style={{textAlign:'right',padding:'1px 4px'}}>maliyet</th>
                <th style={{textAlign:'right',padding:'1px 4px'}}>NET</th>
                <th style={{textAlign:'right',padding:'1px 4px'}}>maliyet/brüt</th>
              </tr></thead>
              <tbody>
                {board.micro_evidence.signals.map((s: any) => (
                  <tr key={s.name} style={{color:C.text}}>
                    <td style={{padding:'1px 4px'}}>{s.name}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px',color:C.neon}}>
                      %{s.gross_pct}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px'}}>
                      %{s.cost_pct}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px',color:C.danger}}>
                      %{s.net_pct}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px',color:C.warn}}>
                      {s.cost_multiple}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>
              Gürültü çıkanlar: {board.micro_evidence.noise.join(' · ')}
            </div>
            <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginTop:3}}>
              {board.micro_evidence.spread_nature}
            </div>
            <div style={{fontSize:9,color:C.warn,marginTop:3}}>
              ⚠ {board.micro_evidence.power_warning}
            </div>
            <div style={{fontSize:9,color:C.muted,marginTop:2}}>
              {board.micro_evidence.ladder_pending}
            </div>
            <div style={{fontSize:9,color:C.danger,fontWeight:700,marginTop:3}}>
              {board.micro_evidence.verdict}
            </div>
          </div>
        )}

        <button onClick={() => setOpen(!open)} style={{
          background:'transparent',border:`1px solid ${C.border}`,color:C.info,
          borderRadius:5,padding:'3px 9px',fontSize:9,cursor:'pointer',marginBottom:6}}>
          {open ? '▲ listeyi gizle' : `▼ ${shown.length} göstergeyi tek tek gör`}
        </button>

        {open && (
          <div style={{maxHeight:280,overflowY:'auto',border:`1px solid ${C.border}`,
                       borderRadius:6,marginBottom:8}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:9}}>
              <tbody>
                {shown.map((i: any, k: number) => (
                  <tr key={k} style={{borderBottom:`1px solid ${C.border}55`}}>
                    <td style={{padding:'3px 7px',color:C.text,whiteSpace:'nowrap'}}>{i.name}</td>
                    <td className="mono" style={{padding:'3px 7px',color:C.muted,textAlign:'right'}}>
                      {Number.isFinite(i.value) ? Number(i.value).toLocaleString('tr-TR',
                        {maximumFractionDigits:4}) : '—'}
                    </td>
                    <td style={{padding:'3px 7px',color:sigCol(i.signal),fontWeight:700,
                                whiteSpace:'nowrap'}}>{i.signal}</td>
                    <td style={{padding:'3px 7px',color:C.muted}}>{i.rule}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── ÖLÇÜLMÜŞ KANIT — bu tablo işe yarıyor mu? ── */}
        {ev.measured && (
          <div style={{background:'rgba(255,59,92,0.06)',border:`1px solid ${C.danger}44`,
                       borderRadius:6,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:C.danger,fontWeight:800,marginBottom:4}}>
              ÖLÇÜLDÜ — {ev.headline}
            </div>
            <table style={{width:'100%',fontSize:9,borderCollapse:'collapse',marginBottom:5}}>
              <thead>
                <tr style={{color:C.muted}}>
                  <th style={{textAlign:'left',padding:'1px 4px'}}>TF</th>
                  <th style={{textAlign:'right',padding:'1px 4px'}}>korelasyon</th>
                  <th style={{textAlign:'right',padding:'1px 4px'}}>takip getirisi</th>
                  <th style={{textAlign:'right',padding:'1px 4px'}}>kazanma</th>
                  <th style={{textAlign:'right',padding:'1px 4px'}}>negatif parite</th>
                </tr>
              </thead>
              <tbody>
                {(ev.rows || []).map((r: any) => (
                  <tr key={r.tf} style={{color:C.text}}>
                    <td className="mono" style={{padding:'1px 4px'}}>{r.tf}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px',color:C.danger}}>
                      {r.corr_fwd1.toFixed(3)}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px',color:C.danger}}>
                      %{r.follow_ret_pct.toFixed(2)}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px'}}>
                      %{r.follow_winrate.toFixed(1)}</td>
                    <td className="mono" style={{textAlign:'right',padding:'1px 4px'}}>{r.neg_pairs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginBottom:4}}>{ev.detail}</div>
            {ev.count_comparison && (
              <div style={{background:'rgba(255,182,39,0.07)',border:`1px solid ${C.warn}44`,
                           borderRadius:5,padding:'5px 7px',marginBottom:4}}>
                <div style={{fontSize:9,color:C.warn,fontWeight:700,marginBottom:2}}>
                  {ev.count_comparison.before.n} → {ev.count_comparison.after.n} gösterge:
                  sonuç DEĞİŞMEDİ ({ev.count_comparison.tf})
                </div>
                <div className="mono" style={{fontSize:9,color:C.muted}}>
                  {[ev.count_comparison.before, ev.count_comparison.mid,
                    ev.count_comparison.after].filter(Boolean).map((x:any,i:number)=>(
                    <div key={i}>
                      {x.n}: r {x.corr} · takip %{x.follow_ret_pct} · kazanma %{x.winrate}
                    </div>
                  ))}
                </div>
                <div style={{fontSize:9,color:C.muted,marginTop:2,lineHeight:1.45}}>
                  {ev.count_comparison.note}
                </div>
              </div>
            )}
            <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginBottom:4}}>
              <b style={{color:C.warn}}>Peki tersini alsak?</b> {ev.why_not_inverted}
            </div>
            {ev.dsr && (
              <div style={{fontSize:9,color:C.muted,marginBottom:4}}>
                Deflated Sharpe — 4h <b style={{color:C.danger}}>{ev.dsr['4h']}</b> ·
                1d <b style={{color:C.danger}}>{ev.dsr['1d']}</b> ·
                eşik {ev.dsr.threshold} · denenen hipotez {ev.dsr.n_trials} ·
                şansla beklenen en iyi Sharpe {ev.dsr.sr0_annual}
              </div>
            )}
            <div style={{fontSize:9,color:C.warn,fontWeight:700}}>{ev.verdict}</div>
            <div style={{fontSize:8,color:C.muted,marginTop:3}}>örneklem: {ev.sample}</div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── X / HABER İSTİHBARATI — ölçülmüş hesap etkisi ───────────────────────
function SocialPanel({ social }: { social: any }) {
  if (!social) return null
  const c = social.collector || {}
  const measured: any[] = social.measured_accounts || []
  return (
    <div className="cm-card" style={{marginBottom:8}}>
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:6}}>
        <Radio size={12} color={c.x_live ? C.neon : C.muted} />
        <span className="cm-title">X / HABER İSTİHBARATI</span>
        <span style={{marginLeft:'auto',fontSize:9,fontWeight:700,
          color: c.x_live ? C.neon : C.muted}}>
          X {c.x_live ? 'AÇIK' : 'KAPALI'}
        </span>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'2px 12px',fontSize:9}}>
        <Row k="Toplanan olay" v={`${c.events ?? 0}`}/>
        <Row k="Kaynak" v={(c.sources || []).join(', ') || '—'}/>
        <Row k="Ölçülen hesap" v={`${social.n_measured ?? 0} / ${social.n_accounts ?? 0}`}
             c={(social.n_measured ?? 0) > 0 ? C.neon : C.muted}/>
        <Row k="Eşik" v={`${social.min_events ?? 20} olay`}/>
      </div>

      {measured.length > 0 ? (
        <div style={{marginTop:6}}>
          {measured.map((a: any) => (
            <div key={a.handle} style={{display:'flex',alignItems:'center',gap:6,
                  fontSize:9,padding:'2px 0'}}>
              <span className="mono" style={{flex:1}}>@{a.handle}</span>
              <span style={{color:C.muted}}>{a.best_horizon}</span>
              <span className="mono" style={{fontWeight:700,color:C.warn}}>
                {a.impact_score.toFixed(1)}/10
              </span>
              <span style={{color:C.muted}}>n={a.n_events}</span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{marginTop:6,fontSize:9,color:C.muted,lineHeight:1.6}}>
          Henüz istatistiksel kapıyı geçen hesap yok. Toplayıcı ileriye dönük veri
          biriktiriyor; {social.min_events ?? 20} olaya ulaşan hesaplar otomatik ölçülür.
          <b style={{color:C.text}}> Ölçülmemiş hesap ağırlık almaz</b> — accounts.py'deki
          elle yazılmış değerler modele girmiyor.
        </div>
      )}

      <div style={{fontSize:9,color:C.muted,marginTop:6,lineHeight:1.5,
            borderTop:`1px solid ${C.border}`,paddingTop:6}}>
        {social.x_note}
      </div>
    </div>
  )
}

// ── GÜNÜN %1 HAREKET ADAYI — kanıtlı sıralama (yalnız büyüklük) ─────────
function MoverPanel({ mover, onPick }: { mover: any; onPick: (s: string) => void }) {
  if (!mover || !mover.picks?.length) return null
  return (
    <div className="cm-card" style={{marginBottom:8}}>
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:6}}>
        <Gauge size={12} color={C.warn} />
        <span className="cm-title">BUGÜN %1 OYNAMA OLASILIĞI</span>
      </div>

      {mover.picks.map((p: any) => {
        const trusted = p.model_trusted
        const strong = trusted && p.lift >= 1.10
        return (
          <div key={p.symbol} onClick={() => onPick(p.symbol)}
            style={{cursor:'pointer',borderRadius:6,padding:'6px 8px',marginBottom:4,
              background: strong ? 'rgba(255,182,39,0.08)' : C.panel,
              border:`1px solid ${strong ? C.warn+'66' : C.border}`}}>
            <div style={{display:'flex',alignItems:'center',gap:6}}>
              <span style={{fontSize:9,color:C.muted,width:12}}>{p.rank}.</span>
              <span style={{fontSize:10,fontWeight:700,flex:1}}>{p.symbol}</span>
              <span className="mono" style={{fontSize:12,fontWeight:700,
                color: trusted ? (strong ? C.warn : C.text) : C.muted}}>
                %{(p.probability*100).toFixed(0)}
              </span>
            </div>
            <div style={{display:'flex',gap:8,marginLeft:18,marginTop:2,fontSize:9,flexWrap:'wrap'}}>
              <span style={{color:C.muted}}>taban %{(p.base_rate*100).toFixed(0)}</span>
              <span className="mono" style={{fontWeight:700,
                color: !trusted ? C.muted : p.lift >= 1.05 ? C.neon : p.lift <= 0.95 ? C.danger : C.muted}}>
                lift {p.lift}×
              </span>
              <span style={{color:C.muted}}>bekl. |hareket| %{p.expected_move_pct}</span>
              {!trusted && <span style={{fontSize:8,color:C.danger,border:`1px solid ${C.danger}55`,
                borderRadius:3,padding:'0 3px'}}>MODEL GEÇERSİZ</span>}
            </div>
            {trusted && p.evidence?.length > 0 && (
              <div style={{marginLeft:18,marginTop:3,fontSize:8,color:C.muted,lineHeight:1.5}}>
                kanıt: {p.evidence.slice(0,2).map((e: any) =>
                  `${e.tr} (${e.contribution >= 0 ? '+' : ''}${e.contribution})`).join(' · ')}
              </div>
            )}
          </div>
        )
      })}

      <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginTop:6,
            borderTop:`1px solid ${C.border}`,paddingTop:6}}>
        <b style={{color:C.warn}}>Bu sıralama YÖN vermez.</b> Ölçüldü: hareket
        büyüklüğü için AUC 0,57 (gerçek ama mütevazı), yön için 0,47–0,50 (sıfır bilgi).
        Sistem hangi paritenin oynayacağını sıralar; hangi yöne gideceğini formasyon
        ve seviye tetikleyicileri belirler. <b style={{color:C.text}}>lift</b> = tahmin
        ÷ taban oranı; 1,00'e yakınsa model o gün ek bilgi vermiyordur.
      </div>
    </div>
  )
}

// ── HARMONİK FORMASYON BUTONLARI + seçilenin XABCD künyesi ──────────────
function HarmonicBar({ harm, sel, setSel }:
    { harm: any; sel: string | null; setSel: (k: string | null) => void }) {
  if (!harm) return null
  const avail: Record<string, any> = harm.available || {}
  const keys = Object.keys(avail)
  if (!keys.length) return null
  const chosen = (harm.patterns || []).find((p: any) => p.key === sel)

  return (
    <div className="cm-card" style={{marginBottom:8}}>
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:7}}>
        <Waves size={12} color="#22D3EE" />
        <span className="cm-title">HARMONİK FORMASYONLAR (XABCD)</span>
        {harm.min_quality && (
          <span style={{marginLeft:'auto',fontSize:8,color:C.muted}}>
            uyum eşiği %{Math.round(harm.min_quality * 100)}
          </span>
        )}
      </div>

      {/* ÖLÇÜLMÜŞ KANIT — iki ayrı soru, iki ayrı cevap */}
      {harm.evidence?.tested && (
        <div style={{background:'rgba(34,211,238,0.06)',border:`1px solid #22D3EE44`,
                     borderRadius:6,padding:'6px 8px',marginBottom:7}}>
          <div style={{fontSize:9,fontWeight:800,color:'#22D3EE',marginBottom:3}}>
            ÖLÇÜLDÜ — ŞEKİL GERÇEK, YÖN ÜSTÜNLÜĞÜ YOK
          </div>
          <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginBottom:3}}>
            <b style={{color:C.neon}}>1. Şekil gerçek mi?</b> Evet — tamamlanmış harmonik
            gerçek veride %{harm.evidence.frequency.real_pct}, eşleştirilmiş rastgele
            yürüyüşte %{harm.evidence.frequency.random_pct} oranında görülüyor
            (<b>+{harm.evidence.frequency.gap_pts} puan</b>). Çift tepe/dip bu sınavı
            hiçbir ayarda geçememişti.
          </div>
          <div style={{fontSize:9,color:C.muted,lineHeight:1.5,marginBottom:3}}>
            <b style={{color:C.danger}}>2. Yön bilgisi var mı?</b> Hayır —{' '}
            {harm.evidence.direction.fwd_bars} bar ileri getiri, kontrol{' '}
            %{harm.evidence.direction.control_ret_pct}:{' '}
            LONG {harm.evidence.direction.long.n} olay %{harm.evidence.direction.long.ret_pct}{' '}
            (t={harm.evidence.direction.long.t}) ·{' '}
            SHORT {harm.evidence.direction.short.n} olay %{harm.evidence.direction.short.ret_pct}{' '}
            (t={harm.evidence.direction.short.t}) — ters yönde ve anlamsız.
          </div>
          <div style={{fontSize:9,color:C.warn,fontWeight:700}}>
            {harm.evidence.verdict}
          </div>
        </div>
      )}

      <div style={{display:'flex',flexWrap:'wrap',gap:5,marginBottom:6}}>
        {keys.map(k => {
          const a = avail[k]
          const n = (a.complete || 0) + (a.forming || 0)
          const on = sel === k
          const has = n > 0
          return (
            <button key={k} disabled={!has}
              onClick={() => setSel(on ? null : k)}
              title={has ? `${a.complete} tamamlanmış · ${a.forming} oluşmakta`
                         : 'bu pencerede bulunamadı'}
              style={{
                fontSize:9, fontWeight:700, padding:'4px 8px', borderRadius:6,
                cursor: has ? 'pointer' : 'not-allowed',
                background: on ? a.color : 'transparent',
                color: on ? '#0A0E1A' : (has ? a.color : C.muted),
                border: `1px solid ${has ? a.color + (on ? '' : '66') : C.border}`,
                opacity: has ? 1 : 0.4,
                boxShadow: on ? `0 0 10px ${a.color}66` : 'none',
              }}>
              {a.tr}{has ? ` (${n})` : ''}
            </button>
          )
        })}
      </div>

      {!sel && (
        <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>
          Bir formasyona tıklayın — XABCD bacakları grafiğe neon çizilir, her bacakta
          gerçekleşen/ideal Fibonacci oranı görünür. Tamamlanmamış olanlarda
          <b style={{color:C.text}}> D noktası projekte edilir</b> ve PRZ bölgesi taranır.
        </div>
      )}

      {chosen && (
        <div style={{borderTop:`1px solid ${C.border}`,paddingTop:6}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:5}}>
            <span style={{fontSize:10,fontWeight:700,color:chosen.color}}>{chosen.name}</span>
            <span style={{fontSize:9,fontWeight:700,
              color: chosen.status === 'tamamlandı' ? C.neon : C.warn}}>
              {chosen.status.toUpperCase()} · {chosen.direction === 'LONG' ? 'ALIŞ' : 'SATIŞ'}
            </span>
          </div>

          {/* bacak oranları — kullanıcının istediği "uzunluklar net görülsün" */}
          <div style={{fontSize:9,marginBottom:5}}>
            {(chosen.legs || []).map((l: any, i: number) => (
              <div key={i} style={{display:'flex',justifyContent:'space-between',
                    color: l.projected ? C.warn : C.muted, marginBottom:1}}>
                <span className="mono">{l.frm}→{l.to} {l.ratio_name}</span>
                <span className="mono">
                  {l.ratio_name === 'XA (baz)' ? 'baz bacak'
                    : l.projected ? `hedef ${l.ideal}`
                    : `${l.ratio} / ${l.ideal}`}
                  {!l.projected && l.ratio_name !== 'XA (baz)' && (
                    <b style={{color: l.fit >= 0.9 ? C.neon : l.fit >= 0.6 ? C.warn : C.danger}}>
                      {' '}%{Math.round(l.fit*100)}
                    </b>
                  )}
                </span>
              </div>
            ))}
          </div>

          {chosen.prz && (
            <div style={{fontSize:9,color:C.warn,background:'rgba(255,182,39,0.08)',
                  borderRadius:4,padding:'4px 6px',marginBottom:5,lineHeight:1.5}}>
              <b>PRZ (D bölgesi simülasyonu):</b>{' '}
              {Number(chosen.prz.lo).toLocaleString('tr-TR',{maximumFractionDigits:2})} –{' '}
              {Number(chosen.prz.hi).toLocaleString('tr-TR',{maximumFractionDigits:2})}
              {chosen.prz.overlap
                ? <b style={{color:C.neon}}> · iki projeksiyon KESİŞİYOR (güçlü)</b>
                : <b style={{color:C.danger}}> · örtüşme YOK (zayıf)</b>}
            </div>
          )}

          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'2px 10px',fontSize:9}}>
            <Row k="Giriş (D)" v={`${Number(chosen.entry).toLocaleString('tr-TR',{maximumFractionDigits:2})} (${chosen.entry_pct >= 0 ? '+' : ''}${chosen.entry_pct}%)`} c={chosen.color}/>
            <Row k="Stop" v={Number(chosen.stop).toLocaleString('tr-TR',{maximumFractionDigits:2})} c={C.danger}/>
            <Row k="Hedef 1" v={`${chosen.target_pcts?.[0] >= 0 ? '+' : ''}${chosen.target_pcts?.[0]}%`} c={C.neon}/>
            <Row k="Hedef 2" v={`${chosen.target_pcts?.[1] >= 0 ? '+' : ''}${chosen.target_pcts?.[1]}%`}/>
            <Row k="R/R" v={`1:${chosen.rr}`} c={chosen.rr >= 2 ? C.neon : C.warn}/>
            <Row k="Kalite" v={`%${Math.round(chosen.quality*100)}`}/>
          </div>

          <div style={{marginTop:6,fontSize:9,lineHeight:1.6,color:C.text,
                background:`${chosen.color}14`,borderLeft:`2px solid ${chosen.color}`,
                borderRadius:4,padding:'5px 7px'}}>
            <b style={{color:chosen.color}}>KARAR: </b>{chosen.action}
          </div>
          <div style={{marginTop:4,fontSize:8,color:C.muted,lineHeight:1.5}}>
            Projeksiyon "fiyat buraya gelecek" demek değildir; "kural gerçekleşirse D
            burada olurdu" demektir. Harmonik oranlar geleneksel kabullerdir, örneklem
            dışı kâr kanıtı yoktur — karar desteğidir, otomatik emir değildir.
          </div>
        </div>
      )}
    </div>
  )
}

// ── MUM FORMASYONLARI — bağlam şartlı, ölçülmüş kanıtla ─────────────────
function CandlePanel({ cdl, tf }: { cdl: any; tf: string }) {
  const [open, setOpen] = useState(false)
  if (!cdl) return null
  const list: any[] = cdl.patterns || []
  const s = cdl.summary || {}
  const m = cdl.measured || {}
  const col = (d: string) => d === 'LONG' ? C.neon : d === 'SHORT' ? C.danger : C.muted
  const biasCol = s.bias === 'YUKARI' ? C.neon : s.bias === 'AŞAĞI' ? C.danger : C.muted

  return (
    <div className="cm-card" style={{marginBottom:8}}>
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:6}}>
        <BarChart3 size={12} color={biasCol} />
        <span className="cm-title">MUM FORMASYONLARI ({tf})</span>
        <span style={{marginLeft:'auto',fontSize:9,fontWeight:700,color:biasCol}}>
          {s.n ? `${s.long}▲ ${s.short}▼ ${s.notr}◆` : 'YOK'}
        </span>
      </div>

      {list.length === 0 && (
        <div style={{fontSize:10,color:C.muted}}>
          Son barlarda bağlam şartını sağlayan mum formasyonu yok. (Dönüş
          formasyonları yalnız öncesinde ≥1,5 ATR trend varsa raporlanır.)
        </div>
      )}

      {list.slice(0, 6).map((p, k) => (
        <div key={k} style={{borderRadius:6,padding:'6px 8px',marginBottom:4,
                             background:`${col(p.direction)}0E`,
                             border:`1px solid ${col(p.direction)}44`}}>
          <div style={{display:'flex',alignItems:'center',gap:6,flexWrap:'wrap'}}>
            <span style={{fontSize:10,fontWeight:700,color:C.text}}>{p.name}</span>
            <span style={{fontSize:9,fontWeight:800,color:col(p.direction)}}>
              {p.direction}
            </span>
            <span style={{fontSize:8,color:C.muted,border:`1px solid ${C.border}`,
                          borderRadius:3,padding:'1px 4px'}}>{p.family}</span>
            <span style={{fontSize:8,color:C.muted}}>{p.bars} mum · güç %{Math.round(p.strength*100)}</span>
            {p.evidence?.edge === 'ters' && (
              <span style={{marginLeft:'auto',fontSize:8,fontWeight:800,color:C.danger,
                            background:'rgba(255,59,92,0.15)',borderRadius:3,padding:'1px 5px'}}>
                ÖLÇÜLDÜ: TERS
              </span>
            )}
            {p.evidence?.edge === 'yok' && (
              <span style={{marginLeft:'auto',fontSize:8,color:C.warn}}>ölçüldü: kanıt yok</span>
            )}
          </div>
          <div style={{fontSize:9,color:C.muted,marginTop:2,lineHeight:1.45}}>
            {p.note} <span style={{color:C.info}}>({p.context})</span>
          </div>
          {p.evidence && (
            <div className="mono" style={{fontSize:8,color:C.muted,marginTop:2}}>
              {p.evidence.n} olay · getiri %{p.evidence.ret_pct} vs kontrol
              %{p.evidence.control_pct} · t={p.evidence.t} — {p.evidence.note}
            </div>
          )}
        </div>
      ))}

      {m.verdict && (
        <>
          <button onClick={() => setOpen(!open)} style={{
            background:'transparent',border:`1px solid ${C.border}`,color:C.info,
            borderRadius:5,padding:'3px 8px',fontSize:9,cursor:'pointer',marginTop:4}}>
            {open ? '▲ ölçümü gizle' : '▼ mum formasyonları ölçüldü mü?'}
          </button>
          {open && (
            <div style={{background:'rgba(255,59,92,0.06)',border:`1px solid ${C.danger}44`,
                         borderRadius:6,padding:'7px 9px',marginTop:5}}>
              <div style={{fontSize:9,fontWeight:800,color:C.danger,marginBottom:3}}>
                ÖLÇÜLDÜ — YÖN ÜSTÜNLÜĞÜ YOK
              </div>
              <div className="mono" style={{fontSize:9,color:C.muted,marginBottom:3}}>
                {m.fwd_bars} bar ileri · {m.n_control} kontrol gözlemi ·
                ölçülebilen {m.n_measurable} formasyon
                <br/>
                kontrolden iyi: {m.better_than_control}/{m.n_measurable} ·
                |t|≥2: {m.significant} — <b style={{color:C.danger}}>
                {m.significant_wrong_way} tanesi TERS yönde</b>
              </div>
              <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>{m.verdict}</div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ── OPTİMAL İŞLEM ÇIKARIMI — "hangi yönde yüzde kaçlık işlem?" ──────────
// Yalnız HÂLÂ GEÇERLİ formasyonlardan hesaplanır. Fiyat stop'u ihlal etmiş ya
// da hedefi aşmış kurulumlar hesaba KATILMAZ (denetimde bulunan hata buydu).
function PatternVerdict({ rec, nInvalid }: { rec: any; nInvalid?: number }) {
  if (!rec) return null
  if (!rec.available) return (
    <div style={{fontSize:9,color:C.muted,borderTop:`1px solid ${C.border}`,
                 paddingTop:6,marginTop:4}}>
      İşlem çıkarımı yok — {rec.reason}
      {!!rec.excluded_no_edge?.length && (
        <div style={{color:C.warn,marginTop:2}}>
          Dışlanan (ölçüldü, kanıt yok): {rec.excluded_no_edge.join(' · ')}
        </div>
      )}
    </div>
  )
  const up = rec.direction === 'LONG'
  const col = up ? C.neon : C.danger
  const weak = /BEKLE|ZAYIF|DÜŞÜK GÜVEN/.test(rec.verdict || '')
  const vcol = weak ? C.warn : col
  return (
    <div style={{borderTop:`1px solid ${C.border}`,marginTop:6,paddingTop:7}}>
      <div style={{fontSize:9,color:C.muted,fontWeight:700,marginBottom:5}}>
        GRAFİĞE GÖRE OPTİMAL İŞLEM
      </div>
      <div style={{display:'flex',alignItems:'center',gap:8,flexWrap:'wrap',marginBottom:6}}>
        <span style={{background:`${col}18`,border:`1px solid ${col}66`,borderRadius:5,
                      padding:'3px 9px',color:col,fontWeight:800,fontSize:12}}>
          {up ? <TrendingUp size={11}/> : <TrendingDown size={11}/>} {rec.direction}
        </span>
        <span className="mono" style={{fontSize:13,fontWeight:800,color:col}}>
          hedef %{rec.target_pct > 0 ? '+' : ''}{rec.target_pct}
        </span>
        <span className="mono" style={{fontSize:11,color:C.danger}}>
          stop %{rec.stop_pct > 0 ? '+' : ''}{rec.stop_pct}
        </span>
        <span className="mono" style={{fontSize:11,color:C.info}}>R/R {rec.rr}</span>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:6,marginBottom:6}}>
        {[['GÜVEN', `${(rec.confidence * 100).toFixed(0)}%`, C.info],
          ['BAŞABAŞ KAZANMA', `%${rec.breakeven_winrate}`, C.warn],
          ['DESTEK / KARŞI', `${rec.n_supporting} / ${rec.n_opposing}`, C.muted]]
          .map(([l, v, c]: any) => (
          <div key={l} style={{background:C.surface,borderRadius:5,padding:'5px 7px'}}>
            <div style={{fontSize:8,color:C.muted}}>{l}</div>
            <div className="mono" style={{fontSize:12,fontWeight:700,color:c}}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{background:`${vcol}10`,border:`1px solid ${vcol}44`,borderRadius:5,
                   padding:'6px 8px',fontSize:10,color:vcol,fontWeight:700,marginBottom:5}}>
        {rec.verdict}
      </div>

      <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>
        <div>Ulaşılabilirlik: {rec.reachability} · tipik bar hareketi %{rec.atr_pct}</div>
        {!!nInvalid && (
          <div style={{color:C.warn}}>
            {nInvalid} formasyon hesaba KATILMADI — fiyat ya iptal seviyesini ihlal etti
            ya da hedefi çoktan geçti.
          </div>
        )}
        {!!rec.excluded_no_edge?.length && (
          <div style={{color:C.danger}}>
            Karara katılmadı (yön öngörüsü ölçüldü, kanıtlanamadı):{' '}
            {rec.excluded_no_edge.join(' · ')}
          </div>
        )}
        <div style={{marginTop:3}}>{rec.note}</div>
      </div>
    </div>
  )
}

// ── TESPİT EDİLEN FORMASYONLAR — sıralı liste + grafik açma/kapama ──────
function PatternPanel({ pat, off, setOff, tf }:
    { pat: any; off: Set<string>; setOff: (s: Set<string>) => void; tf: string }) {
  if (!pat) return null
  const list: any[] = pat.patterns || []
  const toggle = (k: string) => {
    const n = new Set(off)
    n.has(k) ? n.delete(k) : n.add(k)
    setOff(n)
  }
  const statusLabel: Record<string, string> = {
    'kırılım': 'KIRILIM', 'tamamlandı': 'TAMAMLANDI',
    'oluşuyor': 'OLUŞUYOR', 'potansiyel': 'POTANSİYEL',
  }
  return (
    <div className="cm-card" style={{marginBottom:8}}>
      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:6}}>
        <span style={{display:'flex',alignItems:'center',gap:5}}>
          <Shapes size={12} color={C.info} />
          <span className="cm-title">TESPİT EDİLEN FORMASYONLAR ({tf})</span>
        </span>
        {pat.consensus && (
          <span style={{fontSize:9,fontWeight:700,
            color: pat.consensus.bias === 'YUKARI' ? C.neon
                 : pat.consensus.bias === 'AŞAĞI' ? C.danger : C.muted}}>
            {pat.consensus.bias}
          </span>
        )}
      </div>

      {list.length === 0 && (
        <div style={{fontSize:10,color:C.muted}}>
          Bu pencerede belirgin formasyon yok. (Motor 14 tür tarıyor: üçgen · kama ·
          bayrak · flama · fincan-kulp · dikdörtgen · OBO · çift tepe/dip)
        </div>
      )}

      <PatternVerdict rec={pat.recommendation} nInvalid={pat.n_invalid} />

      {list.map((p, idx) => {
        const on = !off.has(p.key)
        const up = p.direction === 'LONG'
        return (
          <div key={p.key}
            style={{borderRadius:6,padding:'7px 8px',marginBottom:5,
              background: on ? `${p.color}12` : 'transparent',
              border:`1px solid ${on ? p.color + '55' : C.border}`}}>
            <div style={{display:'flex',alignItems:'center',gap:7}}>
              {/* açma/kapama anahtarı */}
              <button onClick={() => toggle(p.key)} aria-label={`${p.name} çiz`}
                style={{width:26,height:14,borderRadius:8,border:'none',cursor:'pointer',padding:0,
                  background: on ? p.color : '#2A3350', position:'relative', flexShrink:0}}>
                <span style={{position:'absolute',top:2,left: on ? 14 : 2,width:10,height:10,
                  borderRadius:'50%',background:'#0A0E1A',transition:'left .15s'}} />
              </button>
              <span style={{fontSize:10,fontWeight:700,color:C.text,flex:1}}>
                {idx + 1}. {p.name}
              </span>
              {up ? <TrendingUp size={11} color={C.neon}/> : <TrendingDown size={11} color={C.danger}/>}
            </div>

            <div style={{display:'flex',gap:8,marginTop:3,marginLeft:33,flexWrap:'wrap',alignItems:'center'}}>
              <span style={{fontSize:9,fontWeight:700,color:p.color}}>
                {statusLabel[p.status] || p.status}
              </span>
              <span className="mono" style={{fontSize:10,fontWeight:700,
                color: p.target_pct >= 0 ? C.neon : C.danger}}>
                Hedef {p.target_pct >= 0 ? '+' : ''}{p.target_pct}%
              </span>
              {p.valid === false
                ? <span style={{fontSize:8,fontWeight:700,color:C.warn,
                    background:'rgba(255,182,39,0.14)',borderRadius:3,padding:'1px 4px'}}>
                    {p.validity === 'stop_ihlal' ? 'GEÇERSİZ' : 'HEDEF ALINDI'}
                  </span>
                : p.clears_min_move
                ? <span style={{fontSize:8,fontWeight:700,color:C.neon,
                    background:'rgba(0,255,136,0.12)',borderRadius:3,padding:'1px 4px'}}>≥%1 ✓</span>
                : <span style={{fontSize:8,color:C.muted,border:`1px solid ${C.border}`,
                    borderRadius:3,padding:'1px 4px'}}>%1 altı</span>}
            </div>

            {p.valid === false && p.validity_note && (
              <div style={{marginLeft:33,marginTop:3,fontSize:9,color:C.warn,lineHeight:1.45}}>
                {p.validity_note}
              </div>
            )}

            {/* ÖLÇÜLMÜŞ KANIT — sınanmış ve yön üstünlüğü bulunamamış aileler */}
            {p.evidence?.edge === 'yok' && (
              <div style={{marginLeft:33,marginTop:4,padding:'5px 7px',
                           background:'rgba(255,59,92,0.07)',
                           border:`1px solid ${C.danger}44`,borderRadius:5}}>
                <div style={{fontSize:9,color:C.danger,fontWeight:800,marginBottom:2}}>
                  ÖLÇÜLDÜ — YÖN ÜSTÜNLÜĞÜ YOK · karara katılmıyor
                </div>
                <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>
                  {p.evidence.note}
                </div>
                <div className="mono" style={{fontSize:8,color:C.muted,marginTop:2}}>
                  {p.evidence.n_events} olay · {p.evidence.fwd_bars} bar ileri ·
                  olay %{p.evidence.event_ret_pct} vs kontrol %{p.evidence.control_ret_pct} ·
                  t={p.evidence.t}
                </div>
              </div>
            )}

            {on && (
              <div style={{marginLeft:33,marginTop:4,display:'grid',
                gridTemplateColumns:'1fr 1fr',gap:'2px 10px',fontSize:9}}>
                <Row k="Kırılım" v={`$${Number(p.breakout).toLocaleString('tr-TR',{maximumFractionDigits:2})}`}/>
                <Row k="Hedef" v={`$${Number(p.target).toLocaleString('tr-TR',{maximumFractionDigits:2})}`} c={p.color}/>
                <Row k="Geçersiz" v={`$${Number(p.stop).toLocaleString('tr-TR',{maximumFractionDigits:2})}`} c={C.danger}/>
                <Row k="R/R" v={`1:${p.rr}`} c={p.rr >= 2 ? C.neon : C.warn}/>
                <Row k="Kalite" v={`%${Math.round(p.quality*100)}`}/>
                <Row k="Tamamlanma" v={`%${Math.round(p.completion*100)}`}/>
              </div>
            )}
            {on && p.note && (
              <div style={{marginLeft:33,marginTop:4,fontSize:9,color:C.muted,lineHeight:1.5}}>
                {p.note}
              </div>
            )}
          </div>
        )
      })}

      {list.length > 0 && (
        <div style={{fontSize:9,color:C.muted,marginTop:6,lineHeight:1.5,
          borderTop:`1px solid ${C.border}`,paddingTop:6}}>
          Formasyonlar <b style={{color:C.text}}>tek başına işlem açtırmaz</b> — karar
          motoruna özellik olarak girer. Grafikteki kesikli beyaz çizgiler mevcut
          fiyatın ±%1 seviyeleridir; hedefi bu bandın dışında kalan formasyonlar
          <b style={{color:C.neon}}> ≥%1 ✓</b> ile işaretlenir.
        </div>
      )}
    </div>
  )
}

// ── strateji künyesi (gerçek OOS ölçümleri) ─────────────────────────────
function StrategyCard({ strat }: { strat: any }) {
  const [open, setOpen] = useState(false)
  if (!strat) return null
  const m = strat.oos_metrics || {}
  return (
    <div className="cm-card" style={{marginTop:8}}>
      <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:6}}>
        <BookOpen size={12} color={C.info} /><span className="cm-title">STRATEJİ</span>
      </div>
      <div style={{fontSize:10,color:C.text,marginBottom:6}}>{strat.name}</div>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'3px 12px',fontSize:9,marginBottom:6}}>
        <Row k="Sharpe (OOS)" v={`${m.sharpe ?? '-'}`} c={C.neon}/>
        <Row k="CAGR" v={m.cagr!=null?`%${m.cagr}`:'-'}/>
        <Row k="Max düşüş" v={m.dd!=null?`%${m.dd}`:'-'}/>
        <Row k="Calmar" v={`${m.calmar ?? '-'}`}/>
        <Row k="Varlık" v={`${strat.universe_size||0}`}/>
        <Row k="Fit parametre" v={`${strat.fitted_parameters}`} c={C.neon}/>
      </div>
      <div style={{fontSize:9,color:C.muted,lineHeight:1.6}}>{strat.rule}</div>
      <button className="cm-btn" onClick={() => setOpen(!open)}
        style={{background:C.surface,color:C.info,marginTop:8,width:'100%'}}>
        {open ? 'GİZLE' : 'DÜRÜSTLÜK NOTLARI'}
      </button>
      {open && (
        <div style={{marginTop:6,fontSize:9,color:C.muted,lineHeight:1.7}}>
          {strat.one_percent && (
            <div style={{marginBottom:8,border:`1px solid ${C.border}`,borderRadius:6,padding:7}}>
              <div style={{fontSize:9,fontWeight:700,color:C.text,marginBottom:5}}>
                "%1" — iki ayrı şey
              </div>
              <div style={{color:C.neon,marginBottom:2}}>
                ✓ TEK İŞLEMDE %1 — mümkün
              </div>
              <div style={{marginBottom:6}}>{strat.one_percent.per_trade?.explain}</div>
              <div style={{color:C.danger,marginBottom:2}}>
                ✗ HER GÜN %1 (bileşik) — matematiksel olarak imkânsız
              </div>
              <div style={{marginBottom:5}}>{strat.one_percent.per_day_compound?.explain}</div>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'2px 8px'}}>
                <Row k="Yıllık karşılığı" v={`%${strat.one_percent.per_day_compound?.annual_equivalent_pct}`} c={C.danger}/>
                <Row k="Gereken kaldıraç" v={`${strat.one_percent.per_day_compound?.required_leverage}×`} c={C.danger}/>
                <Row k="Gereken yıllık vol" v={`%${strat.one_percent.per_day_compound?.required_annual_vol_pct}`} c={C.danger}/>
                <Row k="Günlük dalgalanma" v={`%${strat.one_percent.per_day_compound?.daily_vol_pct}`} c={C.danger}/>
              </div>
              <div style={{marginTop:6,color:C.warn}}>
                Gerçekçi hedef: aylık %{strat.one_percent.realistic_target?.monthly_pct} ·
                yıllık %{strat.one_percent.realistic_target?.annual_pct} ·
                maks düşüş %{strat.one_percent.realistic_target?.max_drawdown_pct}
              </div>
            </div>
          )}
          <div style={{marginBottom:4}}>Test penceresi: {strat.oos_window}</div>
          <div style={{marginBottom:6}}>Maliyet: {strat.costs_modeled}</div>
          {(strat.honest_notes||[]).map((n:string,i:number) => <div key={i}>▸ {n}</div>)}
        </div>
      )}
    </div>
  )
}

// ── konsensüs kapısı ────────────────────────────────────────────────────
function GatePanel({ gate, actionable }: { gate: any; actionable: boolean }) {
  if (!gate) return null
  const ag = Math.round((gate.agreement || 0) * 100)
  const minAg = Math.round((gate.min_agreement || 0) * 100)
  const ok = actionable
  const color = ok ? C.neon : C.warn
  return (
    <div style={{background:C.panel,border:`1px solid ${ok?'rgba(0,255,136,0.3)':C.border}`,
                 borderRadius:8,padding:10,marginBottom:10}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:6}}>
        <span className="cm-title">KONSENSÜS KAPISI</span>
        <span style={{fontSize:10,fontWeight:700,color}}>{ok?'✓ ORTAK KARAR':'BEKLE'}</span>
      </div>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'3px 14px',fontSize:9}}>
        <Row k="Hizalanma" v={`%${ag} / %${minAg}`} c={ag>=minAg?C.neon:C.warn}/>
        <Row k="Aktif katman" v={`${gate.n_active||0} / ${gate.min_layers||3}`}/>
        <Row k="Hedef modu" v={String(gate.tp_mode||'-').toUpperCase()} c={gate.tp_mode==='atr'?C.info:C.warn}/>
        <Row k="Trend (ADX)" v={gate.adx!=null?`${gate.adx}${gate.trend_ok?' ✓':' ✗'}`:'-'}
             c={gate.trend_ok?C.neon:C.danger}/>
        {gate.veto && <Row k="Haber" v="VETO" c={C.danger}/>}
      </div>
    </div>
  )
}

// ── korelasyon şeridi (gerçek getiri korelasyonu) ───────────────────────
function CorrStrip({ corr }: { corr: any }) {
  if (!corr || !corr.symbols?.length || !corr.matrix?.length) {
    return <div style={{fontSize:10,color:C.muted,alignSelf:'center'}}>Korelasyon hesaplanıyor…</div>
  }
  const pairs: {a:string;b:string;v:number}[] = []
  for (let i = 0; i < corr.symbols.length; i++)
    for (let k = i + 1; k < corr.symbols.length; k++)
      pairs.push({ a: corr.symbols[i].split('/')[0], b: corr.symbols[k].split('/')[0], v: corr.matrix[i][k] })
  pairs.sort((x, y) => Math.abs(y.v) - Math.abs(x.v))
  return (
    <div style={{display:'flex',alignItems:'center',gap:6,flexWrap:'wrap',
                 borderLeft:`1px solid ${C.border}`,paddingLeft:12}}>
      {pairs.slice(0, 5).map((p, i) => (
        <div key={i} className="mono" style={{textAlign:'center',padding:'3px 7px',borderRadius:4,fontSize:9,
             background:p.v>0?'rgba(0,255,136,0.08)':'rgba(255,59,92,0.08)',
             color:p.v>0?C.neon:C.danger}}>
          <div>{p.a}↔{p.b}</div><div>{(p.v*100).toFixed(0)}%</div>
        </div>
      ))}
      <div style={{fontSize:9,color:C.muted}}>
        yoğunlaşma: <b style={{color:corr.concentration==='YÜKSEK'?C.danger:C.text}}>{corr.concentration}</b>
        <div>{corr.bars} × {corr.tf} bar</div>
      </div>
    </div>
  )
}

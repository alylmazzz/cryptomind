/**
 * NET +%1 FIRSAT TARAYICISI — panel katmanı.
 *
 * TASARIM KURALI: bu dosya HİÇBİR SAYI ÜRETMEZ. Gösterdiği her değer API'den
 * gelir; gelmiyorsa "—" ya da "ÖLÇÜLMEDİ" yazar. Eksik veriyi doldurmak için
 * varsayılan üretmek, kullanıcıya olmayan bir kesinlik satmaktır.
 *
 * İKİ SATIR HER ZAMAN AYRI DURUR:
 *     GARANTİ: YOK
 *     NET +%1 HEDEF OLASILIĞI: %X   (ya da: DOĞRULANMIŞ UFUK YOK)
 * Birincisi gizlenemez; ikincisi ancak kanıt kapıları geçilirse sayı taşır.
 */
import { useState, useEffect, useCallback } from 'react'
import { Target, ShieldCheck, AlertTriangle, Sunrise, ChevronDown, ChevronRight,
         Layers, FlaskConical, Ban } from 'lucide-react'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E',
            surface:'#1A2240', deep:'#0A7A46', cyan:'#22D3EE' }

// Şartname 92 — durum renk sistemi. "guaranteed" için yeşil rozet YOKTUR.
const STATUS_COLOR: Record<string, string> = {
  NO_DATA: C.muted, RESEARCH_ONLY: C.muted, UNVERIFIED: C.warn,
  NO_EDGE: C.danger, DEGRADED: C.danger,
  QUALIFIED: C.neon, HIGH_CONFIDENCE: C.deep,
}
const STATUS_TR: Record<string, string> = {
  NO_DATA: 'VERİ YOK', RESEARCH_ONLY: 'ARAŞTIRMA', UNVERIFIED: 'DOĞRULANMADI',
  NO_EDGE: 'KENAR YOK', DEGRADED: 'BOZULDU',
  QUALIFIED: 'NİTELENDİ', HIGH_CONFIDENCE: 'YÜKSEK GÜVEN',
}

// Yakınsama — "bu sayı kesinleşiyor mu?" sorusunun cevabı. Durumdan AYRI
// tutulur: bir hücre kararlı olup yine de kenarsız olabilir (ve tersi).
const CONV_TR: Record<string, string> = {
  CONVERGED: 'kararlı', CONVERGING: 'yakınsıyor',
  REGIME_DEPENDENT: 'rejime koşullu', UNSTABLE: 'kayıyor',
  UNMEASURED: 'ölçülmedi',
}
const CONV_COLOR: Record<string, string> = {
  CONVERGED: C.neon, CONVERGING: C.warn, REGIME_DEPENDENT: C.info,
  UNSTABLE: C.danger, UNMEASURED: C.muted,
}

const j = async (p: string) => (await fetch(`${API}${p}`)).json()
const pct = (x: any, d = 1) => (x == null || !isFinite(x)) ? '—' : `%${(x * 100).toFixed(d)}`
const num = (x: any, d = 4) => (x == null || !isFinite(x)) ? '—' : Number(x).toFixed(d)
const usd = (x: any) => (x == null || !isFinite(x)) ? '—'
  : `$${Number(x).toLocaleString('tr-TR', { maximumFractionDigits: x < 10 ? 4 : 2 })}`

export function Net1Section() {
  const [q, setQ] = useState<any>(null)
  const [morning, setMorning] = useState<any>(null)
  const [ev, setEv] = useState<any>(null)
  const [led, setLed] = useState<any>(null)
  const [uni, setUni] = useState<any>(null)
  const [health, setHealth] = useState<any>(null)
  const [attr, setAttr] = useState<any>(null)
  const [reg, setReg] = useState<any>(null)
  const [openPair, setOpenPair] = useState<string | null>(null)
  const [tab, setTab] = useState<'matris' | 'sabah' | 'kanit' | 'kayit' |
                                 'kurumsal'>('matris')

  const pull = useCallback(async () => {
    try { setQ(await j('/api/qualification')) } catch {}
    try { setMorning(await j('/api/morning')) } catch {}
    try { setHealth(await j('/api/system-health')) } catch {}
  }, [])
  useEffect(() => { pull(); const iv = setInterval(pull, 60000); return () => clearInterval(iv) }, [pull])
  useEffect(() => {
    (async () => {
      try { setEv(await j('/api/qualification/evidence')) } catch {}
      try { setLed(await j('/api/qualification/ledger?days=30')) } catch {}
      try { setUni(await j('/api/qualification/universe')) } catch {}
      try { setAttr(await j('/api/qualification/attribution')) } catch {}
      try { setReg(await j('/api/qualification/models')) } catch {}
    })()
  }, [])

  const sc = q?.scanner || {}
  const cards: any[] = q?.cards || []
  const kalifiye = cards.filter(k => k.best_horizon)
  const best = kalifiye[0]

  return (
    <section style={{borderBottom:`1px solid ${C.border}`, background:'#0C1120'}}>
      {/* ── BAŞLIK BANDI (şartname 119) ───────────────────────────────── */}
      <div style={{display:'flex',alignItems:'center',gap:14,flexWrap:'wrap',
                   padding:'8px 16px',borderBottom:`1px solid ${C.border}`}}>
        <div style={{display:'flex',alignItems:'center',gap:7}}>
          <Target size={15} color={C.neon} />
          <span style={{fontWeight:800,letterSpacing:1,fontSize:12}}>
            NET +%1 FIRSAT TARAYICISI
          </span>
        </div>
        {/* Bu rozet GİZLENEMEZ (şartname 51) */}
        <span style={{fontSize:10,fontWeight:800,color:C.warn,
                      border:`1px solid ${C.warn}66`,borderRadius:4,padding:'2px 8px'}}>
          GARANTİ: YOK
        </span>
        <Stat k="TARANAN" v={sc.markets_scanned ?? '—'} />
        <Stat k="UYGUN" v={sc.markets_eligible ?? uni?.eligible ?? '—'} />
        <Stat k="KOMBİNASYON" v={sc.combinations ?? '—'} />
        <Stat k="QUALIFIED" v={kalifiye.length} c={kalifiye.length ? C.neon : C.muted} />
        <Stat k="YÜKSEK GÜVEN"
              v={kalifiye.filter(k => k.status === 'HIGH_CONFIDENCE').length} />
        {health && <HealthBadge h={health} />}
        {q?.scanner?.schema_violations != null && (
          <span title={(q.scanner.schema_violation_sample || []).join(' · ') ||
                       'birim ve aralık kontrolleri temiz'}
            style={{fontSize:8.5,fontWeight:800,cursor:'help',
                    color:q.scanner.schema_violations ? C.danger : C.neon,
                    border:`1px solid ${q.scanner.schema_violations ? C.danger : C.neon}55`,
                    borderRadius:4,padding:'1px 6px'}}>
            ŞEMA {q.scanner.schema_violations ? `${q.scanner.schema_violations} İHLAL` : 'TEMİZ'}
          </span>
        )}
        {q?.fast_mode && (
          <span title="Yayımlanmış fırsat varken tarama sıklaşır (§40)"
            style={{fontSize:8.5,fontWeight:800,color:C.cyan,
                    border:`1px solid ${C.cyan}66`,borderRadius:4,padding:'1px 6px'}}>
            HIZLI MOD
          </span>
        )}
        <div style={{marginLeft:'auto',fontSize:9,color:C.muted}} className="mono">
          {q?.generated_at || 'tarama bekleniyor'}
          {q?.age_sec != null && ` · ${Math.round(q.age_sec)} sn önce`}
          {q?.last_scan_sec != null && ` · tarama ${q.last_scan_sec} sn`}
        </div>
      </div>

      {/* ── #1 EN İYİ KURULUM ya da FIRSAT YOK ───────────────────────── */}
      {best ? <BestSetup card={best} /> : <NoOpportunity scanner={sc} cards={cards}
                                                         morning={morning} />}

      {/* ── SEKMELER ─────────────────────────────────────────────────── */}
      <div style={{display:'flex',gap:4,padding:'6px 16px 0',flexWrap:'wrap'}}>
        {([['matris','PARİTE × UFUK MATRİSİ'],['sabah','SABAH %1 HARİTASI'],
           ['kanit','KANIT & DOĞRULAMA'],['kayit','TAHMİN KAYDI'],
           ['kurumsal','SİSTEM & MODEL YÖNETİŞİMİ']] as const).map(([k, l]) => (
          <button key={k} className="cm-btn" onClick={() => setTab(k)}
            style={{background:tab===k?C.neon:C.surface,color:tab===k?'#0A0E1A':C.muted,
                    fontSize:9,letterSpacing:0.5}}>{l}</button>
        ))}
      </div>

      <div style={{padding:'8px 16px 12px'}}>
        {tab === 'matris' && (
          <Matrix cards={cards} open={openPair} setOpen={setOpenPair} />)}
        {tab === 'sabah' && <MorningMap m={morning} />}
        {tab === 'kanit' && <Evidence ev={ev} uni={uni}
                                      limits={q?.scope_limits || []} />}
        {tab === 'kayit' && <LedgerPanel led={led} />}
        {tab === 'kurumsal' && <Governance health={health} attr={attr}
                                           reg={reg} q={q} />}
      </div>
    </section>
  )
}

const HEALTH_COLOR: Record<string, string> = {
  GREEN: C.neon, WATCH: C.warn, DEGRADED: C.danger, RED: C.danger,
  UNKNOWN: C.muted, NOT_CONFIGURED: C.info,
}
const HEALTH_TR: Record<string, string> = {
  GREEN: 'sağlıklı', WATCH: 'izlemede', DEGRADED: 'bozulmuş',
  RED: 'kritik', UNKNOWN: 'ölçülmedi',
  // "kurulmadı" ≠ "bozuldu". Bu ayrım olmadığında panel, kasıtlı olarak
  // kurulmamış yürütme katmanı yüzünden "SİSTEM bozulmuş" yazıyordu.
  NOT_CONFIGURED: 'kurulmadı',
}

// Otopilot kapalıysa SEBEBİ yazılır. Sebepsiz "KAPALI" arıza gibi okunur.
function autopilotLabel(h: any) {
  if (h.autopilot) return { text: 'OTOPİLOT AÇILABİLİR', col: C.neon }
  const yalnizcaKurulmamis = h.blocking?.length > 0 &&
    h.not_configured?.length > 0 &&
    h.blocking.every((b: string) => h.not_configured.includes(b))
  return yalnizcaKurulmamis
    ? { text: 'OTOMATİK İŞLEM KAPALI (ölçüm modu)', col: C.info }
    : { text: 'OTOPİLOT KAPALI', col: C.danger }
}

function HealthBadge({ h }: { h: any }) {
  const col = HEALTH_COLOR[h.overall] || C.muted
  const olcum = h.mode === 'MEASUREMENT_ONLY'
  return (
    <span title={(h.autopilot_reason || (h.autopilot ? 'otopilot açılabilir' : '')) +
                 (h.blocking?.length ? ` · engelleyen: ${h.blocking.join(', ')}` : '')}
      style={{fontSize:8.5,fontWeight:800,color:col,border:`1px solid ${col}66`,
              borderRadius:4,padding:'1px 6px',cursor:'help'}}>
      SİSTEM {HEALTH_TR[h.overall] || h.overall}
      {olcum && <span style={{color:C.info}}> · ÖLÇÜM MODU</span>}
      {!h.autopilot && !olcum && ' · OTOPİLOT KAPALI'}
      {!h.autopilot && olcum && <span style={{color:C.muted}}> · işlem yok</span>}
    </span>
  )
}

function Governance({ health, attr, reg, q }: any) {
  return (
    <div>
      <Section title="SİSTEM SAĞLIĞI — BİR BİLEŞEN KIRMIZIYSA OTOPİLOT KAPANIR">
        {health ? (
          <div>
            <div style={{display:'flex',gap:10,flexWrap:'wrap',alignItems:'center',
                         marginBottom:6}}>
              <span style={{fontSize:11,fontWeight:800,
                            color:HEALTH_COLOR[health.overall]}}>
                GENEL: {HEALTH_TR[health.overall] || health.overall}</span>
              {health.mode_label && (
                <span style={{fontSize:9.5,fontWeight:700,color:C.info,
                              border:`1px solid ${C.info}55`,borderRadius:4,
                              padding:'1px 6px'}}>
                  {health.mode_label.toUpperCase()}</span>)}
              <span style={{fontSize:10,fontWeight:700,
                            color:autopilotLabel(health).col}}>
                {autopilotLabel(health).text}</span>
              {health.blocking?.length > 0 && (
                <span style={{fontSize:9,color:C.muted}}>
                  engelleyen: {health.blocking.join(' · ')}</span>)}
            </div>
            {health.autopilot_reason && (
              <div style={{fontSize:9,color:C.muted,marginBottom:6,
                           lineHeight:1.6}}>
                {health.autopilot_reason}</div>)}
            <div style={{display:'grid',
                         gridTemplateColumns:'repeat(auto-fit,minmax(210px,1fr))',
                         gap:6}}>
              {(health.components || []).map((c: any) => (
                <div key={c.name} style={{background:C.surface,
                     border:`1px solid ${C.border}`,borderRadius:6,padding:'5px 8px'}}>
                  <div style={{fontSize:9.5,fontWeight:700}}>{c.name}</div>
                  <div style={{fontSize:9,fontWeight:700,
                               color:HEALTH_COLOR[c.state] || C.muted}}>
                    {HEALTH_TR[c.state] || c.state}</div>
                  <div style={{fontSize:8.5,color:C.muted,marginTop:2,
                               lineHeight:1.5}}>{c.detail}</div>
                </div>
              ))}
            </div>
            <div style={{marginTop:5,fontSize:9,color:C.muted,lineHeight:1.6}}>
              {health.note}
            </div>
          </div>
        ) : <Empty text="Sistem sağlığı okunamadı." />}
      </Section>

      {q?.scanner?.field_coverage?.always_null?.length > 0 && (
        <Section title="HİÇBİR HÜCREDE DOLMAYAN ALANLAR">
          <div style={{display:'grid',gap:4}}>
            {q.scanner.field_coverage.always_null.map((f: any) => (
              <div key={f.field} style={{fontSize:9,lineHeight:1.6,
                   color: f.expected ? C.muted : C.warn}}>
                <b>{f.field}</b> — {f.reason}
                {!f.diagnosed && <span style={{color:C.danger}}> · TEŞHİS EDİLMEDİ</span>}
              </div>
            ))}
          </div>
          <div style={{fontSize:8.5,color:C.muted,marginTop:5,lineHeight:1.6}}>
            {q.scanner.field_coverage.note}</div>
        </Section>)}

      {attr?.shadow && (attr.shadow.n > 0 || (attr.shadow_pending ?? 0) > 0 ||
        attr.shadow.verdict !== 'UNMEASURED') && (
        <Section title="GÖLGE KAYIT MUTABAKATI — SİNYAL DEĞİL, SINAV">
          <div style={{display:'grid',
                       gridTemplateColumns:'repeat(auto-fit,minmax(140px,1fr))',
                       gap:6}}>
            <Cell k="ÇÖZÜLMÜŞ GÖLGE" v={attr.shadow.n} />
            <Cell k="TAHMİN EDİLEN EV" v={num(attr.shadow.predicted_ev_mean, 4)} />
            <Cell k="GERÇEKLEŞEN NET" v={num(attr.shadow.realized_net_mean, 4)}
                  c={(attr.shadow.realized_net_mean ?? -1) > 0 ? C.neon : C.danger} />
            <Cell k="ORAN" v={num(attr.shadow.ratio, 3)} big
                  c={attr.shadow.verdict === 'ALIGNED' ? C.neon : C.warn} />
            <Cell k="TAHMİN EDİLEN P" v={num(attr.shadow.predicted_p_mean, 4)} />
            <Cell k="GERÇEKLEŞEN TP ORANI" v={num(attr.shadow.realized_tp_rate, 4)} />
          </div>
          <div style={{fontSize:9,color:C.muted,marginTop:5,lineHeight:1.6}}>
            {(attr.shadow.reasons || []).join(' · ')}
          </div>
          <div style={{fontSize:9,color:C.info,marginTop:4,lineHeight:1.6}}>
            Gölge kayıtlar kalifiye OLMAYAN en iyi hücrelerdir. İşlem önermez ve
            karneye girmez; yalnız "olasılık tahminim doğru mu" sorusunu canlı
            veriyle sınar.
          </div>
        </Section>)}

      <Section title="GERÇEKLEŞEN ÷ TAHMİN EDİLEN NET EV — ANA ÜRÜN METRİĞİ">
        {attr?.overall ? (
          <div>
            <div style={{display:'grid',
                         gridTemplateColumns:'repeat(auto-fit,minmax(140px,1fr))',
                         gap:6}}>
              <Cell k="ÇÖZÜLMÜŞ TAHMİN" v={attr.overall.n} />
              <Cell k="TAHMİN EDİLEN EV"
                    v={num(attr.overall.predicted_ev_mean, 4)} />
              <Cell k="GERÇEKLEŞEN NET"
                    v={num(attr.overall.realized_net_mean, 4)}
                    c={(attr.overall.realized_net_mean ?? -1) > 0 ? C.neon : C.danger} />
              <Cell k="ORAN" v={num(attr.overall.ratio, 3)} big
                    c={attr.overall.verdict === 'ALIGNED' ? C.neon : C.warn} />
              <Cell k="OLASILIK SAPMASI" v={num(attr.overall.probability_gap, 4)} />
              <Cell k="MALİYET SAPMASI" v={num(attr.overall.cost_gap, 4)} />
              <Cell k="YÜRÜTME SAPMASI" v={num(attr.overall.execution_gap, 4)} />
              <Cell k="AÇIKLANAMAYAN" v={num(attr.overall.unexplained_gap, 4)} />
            </div>
            <div style={{marginTop:5,fontSize:10,fontWeight:700,
                         color:attr.overall.verdict === 'ALIGNED' ? C.neon : C.warn}}>
              {attr.verdict_tr?.[attr.overall.verdict] || attr.overall.verdict}
            </div>
            {(attr.overall.reasons || []).map((r: string, i: number) => (
              <div key={i} style={{fontSize:9,color:C.muted}}>▸ {r}</div>
            ))}
            <div style={{marginTop:5,fontSize:9,color:C.muted,lineHeight:1.6}}>
              {attr.why}
            </div>
          </div>
        ) : <Empty text="Mutabakat için henüz çözülmüş tahmin yok." />}
      </Section>

      <Section title="MODEL KAYIT DEFTERİ — KİM YAPTI, KİM DOĞRULADI, KİM ONAYLADI">
        {reg?.models?.length ? (
          <div>
            <div style={{overflowX:'auto'}}>
              <table style={{borderCollapse:'collapse',fontSize:9.5,minWidth:700}}>
                <thead><tr style={{color:C.muted,textAlign:'left'}}>
                  {['MODEL','UFUK','YÖN','RİSK','DURUM','YAPAN','DOĞRULAYAN',
                    'ONAYLAYAN'].map(h =>
                    <th key={h} style={{padding:'3px 8px 3px 0',
                        borderBottom:`1px solid ${C.border}`}}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {reg.models.slice(0, 24).map((m: any) => (
                    <tr key={m.model_id} style={{borderBottom:`1px solid ${C.border}44`}}>
                      <td className="mono" style={{padding:'3px 8px 3px 0'}}>
                        {m.model_id}</td>
                      <td className="mono" style={{padding:'3px 8px 3px 0'}}>
                        {m.horizon}</td>
                      <td style={{padding:'3px 8px 3px 0',
                           color:m.direction==='LONG'?C.neon:C.danger}}>
                        {m.direction}</td>
                      <td className="mono" style={{padding:'3px 8px 3px 0',
                           color:m.risk_tier===1?C.danger:C.muted}}>T{m.risk_tier}</td>
                      <td style={{padding:'3px 8px 3px 0',fontWeight:700}}>
                        {reg.status_tr?.[m.status] || m.status}</td>
                      <td style={{padding:'3px 8px 3px 0',color:C.muted}}>
                        {m.built_by}</td>
                      <td style={{padding:'3px 8px 3px 0',
                           color:m.validated_by?C.neon:C.warn}}>
                        {m.validated_by || 'YOK'}</td>
                      <td style={{padding:'3px 0',
                           color:m.approved_by?C.neon:C.warn}}>
                        {m.approved_by || 'YOK'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{marginTop:5,fontSize:9,color:C.muted,lineHeight:1.6}}>
              {reg.note}
            </div>
            <ModelCardDetail m={reg.models[0]} />
          </div>
        ) : <Empty text="Model kaydı yok — araştırma koşusu kart üretmedi." />}
      </Section>

      <Section title="KOŞU KİMLİĞİ — YENİDEN ÜRETİLEBİLİRLİK">
        <RunProvenance q={q} />
      </Section>
    </div>
  )
}

function ModelCardDetail({ m }: { m: any }) {
  if (!m) return null
  const liste = (b: string, xs: string[]) => (xs?.length ? (
    <div style={{marginTop:4}}>
      <div style={{fontSize:9,color:C.muted,letterSpacing:0.4}}>{b}</div>
      {xs.map((x, i) => (
        <div key={i} style={{fontSize:9,color:'#9AA6C4'}}>▸ {x}</div>))}
    </div>
  ) : null)
  return (
    <div style={{marginTop:8,background:C.panel,border:`1px solid ${C.border}`,
                 borderRadius:6,padding:'8px 10px'}}>
      <div style={{fontSize:10,fontWeight:800,marginBottom:3}}>
        MODEL KARTI · {m.model_id}</div>
      <div style={{fontSize:9,color:C.muted}}>{m.purpose}</div>
      {liste('VARSAYIMLAR', m.assumptions)}
      {liste('ÖLÇÜLMÜŞ SINIRLAR', m.limitations)}
      {liste('YASAK KULLANIMLAR', m.prohibited_uses)}
      {liste('BİLİNEN BOZULMA BİÇİMLERİ', m.known_failure_modes)}
      {liste('EMEKLİLİK ÖLÇÜTLERİ', m.retirement_criteria)}
    </div>
  )
}

function RunProvenance({ q }: { q: any }) {
  const p = q?.provenance
  if (!p) {
    return (
      <div style={{fontSize:9.5,color:C.muted,lineHeight:1.6}}>
        Koşu kimliği canlı taramada değil, araştırma artefaktında tutulur
        (<span className="mono">runs/qualification/provenance.json</span>):
        veri parmak izi, kod özeti, ayar özeti ve tohum. Aynı üçlü aynı sonucu
        vermelidir; biri değişirse koşu yeniden üretilmiş sayılmaz.
      </div>
    )
  }
  return (
    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(150px,1fr))',
                 gap:6}}>
      <Cell k="KOŞU" v={p.run_id} />
      <Cell k="VERİ ÖZETİ" v={p.dataset?.hash} />
      <Cell k="KOD ÖZETİ" v={p.code?.source_hash} />
      <Cell k="AYAR ÖZETİ" v={p.config_hash} />
      <Cell k="TOHUM" v={p.seed} />
      <Cell k="SÜRE" v={p.duration_sec ? `${p.duration_sec} sn` : '—'} />
    </div>
  )
}

function Stat({ k, v, c }: { k: string; v: any; c?: string }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:5,fontSize:10}}>
      <span style={{color:C.muted}}>{k}</span>
      <span className="mono" style={{color:c || C.text,fontWeight:700}}>{v}</span>
    </div>
  )
}

/* ─────────────────────── #1 EN İYİ KURULUM ─────────────────────── */
function BestSetup({ card }: { card: any }) {
  const col = STATUS_COLOR[card.status] || C.muted
  return (
    <div style={{padding:'10px 16px',background:'rgba(0,255,136,0.04)',
                 borderBottom:`1px solid ${C.border}`}}>
      <div style={{display:'flex',alignItems:'center',gap:10,flexWrap:'wrap',marginBottom:8}}>
        <span style={{fontSize:10,color:C.muted,letterSpacing:1}}>#1 EN İYİ KURULUM</span>
        <span style={{fontWeight:800,fontSize:14}}>{card.symbol}</span>
        <span style={{fontWeight:800,fontSize:12,
                      color:card.direction==='LONG'?C.neon:C.danger}}>{card.direction}</span>
        <Badge text={STATUS_TR[card.status] || card.status} color={col} />
        <span style={{fontSize:10,color:C.muted}}>
          EN İYİ UFUK <b style={{color:C.text}}>{card.best_horizon}</b>
          {card.earliest_qualified_horizon &&
            card.earliest_qualified_horizon !== card.best_horizon &&
            <> · EN ERKEN <b style={{color:C.text}}>{card.earliest_qualified_horizon}</b></>}
        </span>
        <span style={{marginLeft:'auto',fontSize:9,color:C.warn,fontWeight:700}}>
          {card.guarantee_line}
        </span>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(140px,1fr))',gap:8}}>
        <Cell k="NET +%1 HEDEF OLASILIĞI" v={pct(card.p_target_first)} c={C.neon} big />
        <Cell k="%95 ALT SINIR (karar)" v={pct(card.p_target_first_lower95)} c={C.warn} big />
        <Cell k="P(STOP ÖNCE)" v={pct(card.p_stop_first)} c={C.danger} />
        <Cell k="P(ZAMAN AŞIMI)" v={pct(card.p_timeout)} />
        <Cell k="GİRİŞ BÖLGESİ"
              v={`${usd(card.entry_low)} – ${usd(card.entry_high)}`} />
        <Cell k="OPTİMAL GİRİŞ" v={usd(card.optimal_entry)} c={C.info} />
        <Cell k="NET +%1 ÇIKIŞ" v={usd(card.net_1pct_exit)} c={C.neon} />
        <Cell k="STOP" v={usd(card.stop)} c={C.danger} />
        <Cell k="MAKS TAKİP FİYATI" v={usd(card.max_chase_price)} />
        <Cell k="BEKLENEN HEDEF SÜRESİ"
              v={card.expected_target_time_hours != null
                 ? `${card.expected_target_time_hours.toFixed(1)} sa` : '—'} />
        <Cell k="ROBUST EV" v={num(card.robust_expected_value, 4)}
              c={(card.robust_expected_value ?? -1) > 0 ? C.neon : C.danger} />
        <Cell k="MAKS SERMAYE" v={usd(card.max_capacity_usd)} />
        <Cell k="TABAN / GEREKEN / GERÇEK LIFT"
              v={`${pct(card.baseline_target_rate)} / ${pct(card.required_probability_lift)} / ${pct(card.actual_probability_lift)}`} />
        <Cell k="MALİYET MODELİ" v={card.cost_model}
              c={card.cost_model === 'MEASURED_L2_VWAP' ? C.neon : C.warn} />
        <Cell k="YAKINSAMA" v={CONV_TR[card.convergence?.verdict] || '—'}
              c={CONV_COLOR[card.convergence?.verdict] || C.muted} />
        <Cell k="VERİ KALİTESİ"
              v={card.data_quality != null ? `${(card.data_quality * 100).toFixed(1)}` : '—'}
              c={(card.data_quality ?? 0) > 0.9 ? C.neon : C.warn} />
        <Cell k="GEÇERLİ" v={card.valid_until || '—'} />
      </div>

      {card.why_this_horizon && (
        <div style={{marginTop:8,fontSize:10,color:'#9AA6C4',lineHeight:1.6}}>
          <b style={{color:C.text}}>NEDEN BU UFUK:</b> {card.why_this_horizon}
        </div>
      )}
      <BadgeRow card={card} />
    </div>
  )
}

function BadgeRow({ card }: { card: any }) {
  /* Şartname 53 — rozet YALNIZ gerçekten geçtiyse gösterilir. */
  const b: [string, boolean][] = [
    ['L2 MALİYET ÖLÇÜLDÜ', card.cost_model === 'MEASURED_L2_VWAP'],
    ['OOS DOĞRULANDI', card.status === 'QUALIFIED' || card.status === 'HIGH_CONFIDENCE'],
    ['KALİBRE', card.calibration_error != null && card.calibration_error < 0.05],
    ['GÖLGE DOĞRULANDI', card.status === 'HIGH_CONFIDENCE'],
  ]
  const gecen = b.filter(x => x[1])
  if (!gecen.length) return null
  return (
    <div style={{display:'flex',gap:5,marginTop:6,flexWrap:'wrap'}}>
      {gecen.map(([t]) => <Badge key={t} text={t} color={C.info} small />)}
    </div>
  )
}

function Badge({ text, color, small }: { text: string; color: string; small?: boolean }) {
  return (
    <span style={{fontSize:small?8:9,fontWeight:800,color,border:`1px solid ${color}66`,
                  borderRadius:4,padding:small?'1px 5px':'2px 7px',letterSpacing:0.5}}>
      {text}
    </span>
  )
}

function Cell({ k, v, c, big }: { k: string; v: any; c?: string; big?: boolean }) {
  return (
    <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:6,padding:'6px 8px'}}>
      <div style={{fontSize:8.5,color:C.muted,letterSpacing:0.4}}>{k}</div>
      <div className="mono" style={{color:c || C.text,fontWeight:700,fontSize:big?15:12}}>{v}</div>
    </div>
  )
}

/* ─────────────────────── FIRSAT YOK ─────────────────────── */
function NoOpportunity({ scanner, cards, morning }: any) {
  const yakin: any[] = morning?.report?.nearest_to_qualification || []
  return (
    <div style={{padding:'10px 16px',background:'rgba(255,182,39,0.04)',
                 borderBottom:`1px solid ${C.border}`}}>
      <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:6}}>
        <Ban size={14} color={C.warn} />
        <span style={{fontWeight:800,fontSize:12,color:C.warn}}>
          {scanner.empty_message || 'ŞU AN NET +%1 İÇİN QUALIFIED FIRSAT YOK'}
        </span>
      </div>
      <div style={{fontSize:10,color:'#9AA6C4',lineHeight:1.6,maxWidth:900}}>
        Bu sonuç bir hata değil, ölçümün kendisidir. Sistem kapıları geçen bir
        kurulum bulamadığında düşük kaliteli bir alternatif üretmez. Aşağıdaki
        matriste her ufuk için hangi kapının düştüğü yazılıdır.
      </div>
      {yakin.length > 0 && (
        <div style={{marginTop:8}}>
          <div style={{fontSize:9,color:C.muted,letterSpacing:0.5,marginBottom:4}}>
            NİTELENDİRMEYE EN YAKIN — <b style={{color:C.warn}}>İŞLEM ÖNERİSİ DEĞİLDİR</b>
          </div>
          <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
            {yakin.map((y, i) => (
              <div key={i} style={{background:C.surface,border:`1px solid ${C.border}`,
                                   borderRadius:6,padding:'5px 9px',fontSize:10}}>
                <b>{y.symbol}</b> {y.horizon} {y.direction}
                <span style={{color:C.muted}}> — {(y.missing || []).join(' · ')}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {cards.length === 0 && (
        <div style={{marginTop:6,fontSize:10,color:C.muted}}>
          Tarama henüz koşmadı ya da artefaktlar yüklenmedi.
        </div>
      )}
    </div>
  )
}

/* ─────────────────────── PARİTE × UFUK MATRİSİ ─────────────────────── */
function Matrix({ cards, open, setOpen }:
  { cards: any[]; open: string | null; setOpen: (s: string | null) => void }) {
  if (!cards.length) return <Empty text="Matris henüz üretilmedi." />
  return (
    <div>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',fontSize:10,minWidth:820}}>
          <thead>
            <tr style={{color:C.muted,textAlign:'left'}}>
              {['PARİTE','EN İYİ UFUK','EN ERKEN','YÖN','P(+%1)','ALT95','ROBUST EV',
                'HEDEF SÜRESİ','MALİYET','DURUM','YAKINSAMA','TÜM UFUKLAR'].map(h => (
                <th key={h} style={{padding:'4px 6px',borderBottom:`1px solid ${C.border}`,
                                    fontWeight:600,letterSpacing:0.3}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cards.map(k => {
              const col = STATUS_COLOR[k.status] || C.muted
              const acik = open === k.symbol
              return (
                <tr key={k.symbol} style={{borderBottom:`1px solid ${C.border}55`}}>
                  <td style={{padding:'4px 6px',fontWeight:700}}>{k.symbol}</td>
                  <td className="mono" style={{padding:'4px 6px'}}>{k.best_horizon || '—'}</td>
                  <td className="mono" style={{padding:'4px 6px',color:C.muted}}>
                    {k.earliest_qualified_horizon || '—'}</td>
                  <td style={{padding:'4px 6px',fontWeight:700,
                              color:k.direction==='LONG'?C.neon:k.direction==='SHORT'?C.danger:C.muted}}>
                    {k.direction || '—'}</td>
                  <td className="mono" style={{padding:'4px 6px'}}>{pct(k.p_target_first)}</td>
                  <td className="mono" style={{padding:'4px 6px',color:C.warn}}>
                    {pct(k.p_target_first_lower95)}</td>
                  <td className="mono" style={{padding:'4px 6px',
                       color:(k.robust_expected_value ?? -1) > 0 ? C.neon : C.danger}}>
                    {num(k.robust_expected_value, 3)}</td>
                  <td className="mono" style={{padding:'4px 6px'}}>
                    {k.expected_target_time_hours != null
                      ? `${k.expected_target_time_hours.toFixed(1)} sa` : '—'}</td>
                  <td style={{padding:'4px 6px',fontSize:8.5,
                              color:k.cost_model==='MEASURED_L2_VWAP'?C.neon:C.warn}}>
                    {k.cost_model==='MEASURED_L2_VWAP'?'L2 ÖLÇÜLDÜ':'TAHMİNİ'}</td>
                  <td style={{padding:'4px 6px'}}>
                    <span style={{color:col,fontWeight:700,fontSize:9}}>
                      {STATUS_TR[k.status] || k.status}</span></td>
                  <td style={{padding:'4px 6px'}}>
                    <ConvBadge c={k.convergence} /></td>
                  <td style={{padding:'4px 6px'}}>
                    <button className="cm-btn" onClick={() => setOpen(acik ? null : k.symbol)}
                      style={{background:C.surface,color:C.info,fontSize:8.5,
                              display:'inline-flex',alignItems:'center',gap:3}}>
                      {acik ? <ChevronDown size={10}/> : <ChevronRight size={10}/>}
                      {acik ? 'GİZLE' : 'GÖSTER'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {open && <HorizonDetail card={cards.find(k => k.symbol === open)} />}
    </div>
  )
}

function HorizonDetail({ card }: { card: any }) {
  if (!card) return null
  const hz: any[] = card.horizons || []
  const ufuklar = Array.from(new Set(hz.map(h => h.horizon)))
  const bul = (u: string, d: string) => hz.find(h => h.horizon === u && h.direction === d)
  return (
    <div style={{marginTop:8,background:C.panel,border:`1px solid ${C.border}`,
                 borderRadius:8,padding:10}}>
      <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:6}}>
        <Layers size={12} color={C.info} />
        <span style={{fontSize:10,fontWeight:800}}>{card.symbol} — TÜM UFUKLAR</span>
        <span style={{fontSize:9,color:C.warn,fontWeight:700}}>GARANTİ: YOK</span>
        {card.regime && <span style={{fontSize:9,color:C.muted}}>
          rejim: {card.regime}{card.structure ? ` · ${card.structure}` : ''}</span>}
      </div>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%',borderCollapse:'collapse',fontSize:9.5,minWidth:900}}>
          <thead>
            <tr style={{color:C.muted,textAlign:'left'}}>
              {['UFUK','LONG KÖR','LONG P','LONG ALT95','LONG EV','LONG DURUM',
                'SHORT KÖR','SHORT P','SHORT ALT95','SHORT EV','SHORT DURUM',
                'HEDEF σ','R/R','ETKİN N','YAKINSAMA','DSR','PBO',
                'DÜŞEN KAPI'].map(h => (
                <th key={h} style={{padding:'3px 5px',borderBottom:`1px solid ${C.border}`,
                                    fontWeight:600}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ufuklar.map(u => {
              const L = bul(u, 'LONG'), S = bul(u, 'SHORT')
              const ref = L?.reference_only
              const red = [...(L?.rejection_reasons_tr || []), ...(S?.rejection_reasons_tr || [])]
              return (
                <tr key={u} style={{borderBottom:`1px solid ${C.border}44`,
                                    opacity:ref?0.55:1}}>
                  <td className="mono" style={{padding:'3px 5px',fontWeight:700}}>
                    {u}{ref && <span style={{fontSize:7,color:C.muted}}> REF</span>}</td>
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {pct(L?.baseline)}</td>
                  <Prob h={L} /><td className="mono" style={{padding:'3px 5px',color:C.warn}}>
                    {pct(L?.lower95)}</td>
                  <td className="mono" style={{padding:'3px 5px',
                       color:(L?.robust_ev ?? -1) > 0 ? C.neon : C.danger}}>
                    {num(L?.robust_ev, 3)}</td>
                  <StatusTd h={L} />
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {pct(S?.baseline)}</td>
                  <Prob h={S} /><td className="mono" style={{padding:'3px 5px',color:C.warn}}>
                    {pct(S?.lower95)}</td>
                  <td className="mono" style={{padding:'3px 5px',
                       color:(S?.robust_ev ?? -1) > 0 ? C.neon : C.danger}}>
                    {num(S?.robust_ev, 3)}</td>
                  <StatusTd h={S} />
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {num(L?.target_distance_sigma, 2)}</td>
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {num(L?.rr, 2)}</td>
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {L?.n_eff != null ? Math.round(L.n_eff) : '—'}</td>
                  <td style={{padding:'3px 5px'}}><ConvBadge c={L?.convergence} /></td>
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {num(L?.dsr, 3)}</td>
                  <td className="mono" style={{padding:'3px 5px',color:C.muted}}>
                    {num(L?.pbo, 2)}</td>
                  <td style={{padding:'3px 5px',color:C.muted,fontSize:8.5}}>
                    {L?.comparability_note && (
                      <span title={L.comparability_note}
                        style={{color:L.comparability_ok ? C.warn : C.danger,
                                cursor:'help',marginRight:4}}>⚠</span>)}
                    {Array.from(new Set(red)).slice(0, 3).join(' · ') || '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div style={{marginTop:6,fontSize:9,color:C.muted,lineHeight:1.6}}>
        <b style={{color:C.text}}>KÖR</b> = model olmadan, rastgele bir anda
        girildiğinde net +%1 hedefin stop'tan önce gelme oranı.
        <b style={{color:C.text}}> P</b> = modelin seçtiği anlarda ölçülen aynı oran;
        aradaki fark modelin kattığı bilgidir.
        <b style={{color:C.text}}> ALT95</b> kararda kullanılan muhafazakâr sınırdır —
        nokta tahmin değil bu kullanılır.
        <b style={{color:C.text}}> HEDEF σ</b> hedefin kaç standart sapma uzakta
        olduğudur; büyük değer fiziksel olarak zor demektir.
        <b style={{color:C.text}}> YAKINSAMA</b> sayının kesinleşip kesinleşmediğini
        söyler: <i>kararlı</i> = dönem ve rejimler arasında tutarlı, aralık örneklemle
        düzgün daralıyor; <i>yakınsıyor</i> = kayma yok ama örneklem henüz az;
        <i style={{color:C.info}}>rejime koşullu</i> = zamanla kararlı ama düşük/yüksek
        oynaklıkta çok farklı — ölçüm yanlış değil, <b>tek sayı temsil etmiyor</b>
        (BTC 4h kör tabanı: sakin piyasada %8, panikte %48);
        <i style={{color:C.danger}}>kayıyor</i> = tahmin dönemden döneme savruluyor ve
        <b>daha çok veri bunu düzeltmez</b>.
        <b style={{color:C.text}}> DSR</b> çoklu deneme düzeltmesidir: aynı ölçekteki
        denemeler arasından seçilen bu sonuç şansla beklenen maksimumu aşıyor mu
        (kapı 0,95). <b style={{color:C.text}}>PBO</b> aşırı uyum olasılığıdır —
        örneklem-içinde en iyi aday örneklem-dışında da iyi mi (kapı 0,30).
        <b style={{color:C.warn}}> ⚠</b> işareti, o satırın bir önceki ufukla
        doğrudan karşılaştırılamayacağını söyler (farklı stop seçilmiş).
        REF satırı (48h) yalnız referanstır.
      </div>
    </div>
  )
}

function ConvBadge({ c }: { c: any }) {
  if (!c?.verdict) return <span style={{color:C.muted,fontSize:8.5}}>—</span>
  const col = CONV_COLOR[c.verdict] || C.muted
  const ipucu = [
    c.ci_width != null ? `aralık ${(c.ci_width * 100).toFixed(1)} puan` : null,
    c.period_spread != null ? `dönem farkı ${(c.period_spread * 100).toFixed(1)} puan` : null,
    c.regime_spread != null ? `rejim farkı ${(c.regime_spread * 100).toFixed(1)} puan` : null,
    c.shrink_ratio != null ? `daralma ${c.shrink_ratio.toFixed(2)} (bağımsızda 0,50)` : null,
    ...(c.reasons || []),
  ].filter(Boolean).join(' · ')
  return (
    <span title={ipucu} style={{color:col,fontSize:8.5,fontWeight:700,
                                borderBottom:`1px dotted ${col}66`,cursor:'help'}}>
      {CONV_TR[c.verdict] || c.verdict}
    </span>
  )
}

function Prob({ h }: { h: any }) {
  return <td className="mono" style={{padding:'3px 5px',fontWeight:700}}>{pct(h?.p_target_first)}</td>
}
function StatusTd({ h }: { h: any }) {
  const c = STATUS_COLOR[h?.status] || C.muted
  return <td style={{padding:'3px 5px',color:c,fontWeight:700,fontSize:8.5}}>
    {STATUS_TR[h?.status] || h?.status || '—'}</td>
}

/* ─────────────────────── SABAH HARİTASI ─────────────────────── */
function MorningMap({ m }: { m: any }) {
  if (!m) return <Empty text="Sabah motoru henüz koşmadı." />
  const rdy = m.readiness || {}
  const rap = m.report
  const durumRenk = m.state === 'REPORT READY' ? C.neon
    : m.state === 'NO QUALIFIED OPPORTUNITY' ? C.warn : C.info
  return (
    <div>
      <div style={{display:'flex',gap:12,flexWrap:'wrap',alignItems:'center',marginBottom:8}}>
        <Sunrise size={14} color={C.warn} />
        <span style={{fontWeight:800,fontSize:11}}>CRYPTOMIND — SABAH NET +%1 HARİTASI</span>
        <Badge text={m.state} color={durumRenk} />
        <Stat k="PENCERE" v={m.window} />
        <Stat k="YEREL SAAT" v={m.now_local} />
        <Stat k="SLOT" v={m.current_slot || '—'} />
        <Stat k="EN İYİ SLOT (ÖĞRENİLEN)"
              v={m.learned_slots?.best_slot || 'yeterli kanıt yok'} />
      </div>

      {/* Şartname 52 — readiness bir kâr olasılığı DEĞİLDİR */}
      <div style={{background:C.surface,border:`1px solid ${C.border}`,borderRadius:8,
                   padding:'8px 10px',marginBottom:8}}>
        <div style={{display:'flex',alignItems:'center',gap:8,flexWrap:'wrap'}}>
          <span style={{fontSize:10,fontWeight:800,color:C.info}}>{rdy.label}</span>
          <span className="mono" style={{fontSize:14,fontWeight:800}}>
            {rdy.score != null ? rdy.score.toFixed(2) : '—'}</span>
          <span style={{fontSize:9,color:C.muted}}>{rdy.disclaimer}</span>
        </div>
        <div style={{display:'flex',gap:10,flexWrap:'wrap',marginTop:5}}>
          {Object.entries(rdy.components || {}).map(([k, v]: any) => (
            <Stat key={k} k={k.toUpperCase()} v={Number(v).toFixed(2)} />
          ))}
        </div>
        <div style={{marginTop:5,fontSize:9,color:C.muted}}>
          Yayın kararı: {m.publish_decision}
          {m.slot_switch?.reason && ` · slot: ${m.slot_switch.reason}`}
        </div>
      </div>

      {rap ? (
        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <div style={{display:'flex',gap:10,flexWrap:'wrap',marginBottom:6}}>
            <span style={{fontWeight:800,fontSize:11}}>{rap.title}</span>
            <Stat k="TARİH" v={rap.date} /><Stat k="ÜRETİM" v={rap.generated} />
            <Stat k="TETİK" v={rap.live_trigger || '—'} />
            <Stat k="QUALIFIED" v={rap.qualified} />
            <Badge text={`GARANTİ: ${rap.guarantee}`} color={C.warn} />
          </div>
          {rap.empty_result ? (
            <div style={{fontSize:11,color:C.warn,fontWeight:700}}>{rap.empty_result}</div>
          ) : (
            <div style={{display:'grid',gap:6}}>
              {(rap.opportunities || []).map((o: any) => (
                <div key={o.rank} style={{background:C.surface,borderRadius:6,padding:'6px 9px',
                                          fontSize:10}}>
                  <b>#{o.rank} {o.symbol}</b> {o.direction} · {o.best_horizon}
                  {' · '}P {pct(o.p_target_first)} (alt95 {pct(o.p_target_first_lower95)})
                  {' · '}giriş {usd(o.optimal_entry)} · çıkış {usd(o.net_1pct_exit)}
                  {' · '}stop {usd(o.stop)} · rEV {num(o.robust_expected_value, 3)}
                  <div style={{color:C.muted,fontSize:9,marginTop:2}}>{o.why_this_horizon}</div>
                </div>
              ))}
            </div>
          )}
          {(rap.nearest_to_qualification || []).length > 0 && (
            <div style={{marginTop:8,fontSize:9,color:C.muted}}>
              <b style={{color:C.warn}}>Nitelendirmeye en yakın (işlem önerisi DEĞİL):</b>{' '}
              {rap.nearest_to_qualification.map((y: any, i: number) =>
                <span key={i}>{y.symbol} {y.horizon} {y.direction} — {(y.missing||[]).join(', ')}; </span>)}
            </div>
          )}
          <div style={{marginTop:6,fontSize:9,color:C.muted}}>{rap.closing_note}</div>
        </div>
      ) : (
        <Empty text={m.in_window
          ? 'Sabah penceresi açık — güçlü bir qualified fırsat bekleniyor.'
          : 'Sabah penceresi dışında. Tarama 05:00–11:00 arasında yoğunlaşır.'} />
      )}

      <SlotTable learned={m.learned_slots} />
      <MorningScorecard perf={m.performance} />
    </div>
  )
}

function MorningScorecard({ perf }: { perf: any }) {
  if (!perf) return null
  const o = perf.overall || {}
  const satirlar: any[] = perf.by_trigger_hour || []
  return (
    <Section title="SABAH KARNESİ — TETİK SAATİ BAZINDA (§43)">
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(120px,1fr))',
                   gap:6,marginBottom:6}}>
        <Cell k="YAYIMLANAN" v={o.published ?? 0} />
        <Cell k="HEDEF ÖNCE" v={o.tp_first ?? 0} c={C.neon} />
        <Cell k="STOP ÖNCE" v={o.sl_first ?? 0} c={C.danger} />
        <Cell k="ZAMAN AŞIMI" v={o.timeout ?? 0} />
        <Cell k="NET1 KESİNLİK" v={pct(o.net1_precision)} />
        <Cell k="YANLIŞ FIRSAT" v={pct(o.false_opportunity_rate)} />
        <Cell k="GERÇEK NET" v={`${num(o.realized_net_mean, 3)}%`} />
      </div>
      {satirlar.length > 0 ? (
        <table style={{borderCollapse:'collapse',fontSize:9.5}}>
          <thead><tr style={{color:C.muted,textAlign:'left'}}>
            {['SLOT','YAYIN','TP','SL','TO','NET1 KESİNLİK','GERÇEK NET'].map(h =>
              <th key={h} style={{padding:'2px 10px 2px 0'}}>{h}</th>)}
          </tr></thead>
          <tbody>
            {satirlar.map(r => (
              <tr key={r.slot}>
                <td className="mono" style={{padding:'2px 10px 2px 0',fontWeight:700}}>
                  {r.slot}</td>
                <td className="mono" style={{padding:'2px 10px 2px 0'}}>{r.published}</td>
                <td className="mono" style={{padding:'2px 10px 2px 0',color:C.neon}}>
                  {r.tp_first}</td>
                <td className="mono" style={{padding:'2px 10px 2px 0',color:C.danger}}>
                  {r.sl_first}</td>
                <td className="mono" style={{padding:'2px 10px 2px 0'}}>{r.timeout}</td>
                <td className="mono" style={{padding:'2px 10px 2px 0'}}>
                  {pct(r.net1_precision)}</td>
                <td className="mono" style={{padding:'2px 0'}}>
                  {num(r.realized_net_mean, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{fontSize:9,color:C.muted}}>
          Sabah penceresinde henüz yayımlanmış sinyal yok — tetik saati
          kırılımı için önce qualified fırsat çıkması gerekir.
        </div>
      )}
      <div style={{marginTop:5,fontSize:9,color:C.muted}}>{perf.note}</div>
    </Section>
  )
}

function SlotTable({ learned }: { learned: any }) {
  const w = learned?.windows
  if (!w?.full) return null
  const dolu = w.full.filter((r: any) => r.n > 0)
  if (!dolu.length) {
    return (
      <div style={{marginTop:8,fontSize:9,color:C.muted}}>
        Slot öğrenimi için henüz yayımlanmış sabah sinyali yok. Slot seçimi ancak
        30 gün / 90 gün / tam OOS pencerelerinin ÜÇÜNDE de tutarlı olan bir slot
        çıkınca yapılır — tek pencereye bakılarak saat seçilmez.
      </div>
    )
  }
  return (
    <div style={{marginTop:8,overflowX:'auto'}}>
      <table style={{width:'100%',borderCollapse:'collapse',fontSize:9.5,minWidth:520}}>
        <thead><tr style={{color:C.muted,textAlign:'left'}}>
          {['SLOT','N','NET1 KESİNLİK','ROBUST EV','YANLIŞ FIRSAT','KALİBRASYON','SKOR','TUTARLI']
            .map(h => <th key={h} style={{padding:'3px 5px',
              borderBottom:`1px solid ${C.border}`}}>{h}</th>)}
        </tr></thead>
        <tbody>
          {dolu.map((r: any) => (
            <tr key={r.slot} style={{borderBottom:`1px solid ${C.border}44`}}>
              <td className="mono" style={{padding:'3px 5px',fontWeight:700}}>{r.slot}</td>
              <td className="mono" style={{padding:'3px 5px'}}>{r.n}</td>
              <td className="mono" style={{padding:'3px 5px'}}>{pct(r.net1_precision)}</td>
              <td className="mono" style={{padding:'3px 5px'}}>{num(r.robust_ev_mean, 3)}</td>
              <td className="mono" style={{padding:'3px 5px'}}>{pct(r.false_opportunity_rate)}</td>
              <td className="mono" style={{padding:'3px 5px'}}>{num(r.calibration_error, 3)}</td>
              <td className="mono" style={{padding:'3px 5px',fontWeight:700}}>{num(r.score, 3)}</td>
              <td style={{padding:'3px 5px',color:learned.consistent?.[r.slot]?C.neon:C.muted}}>
                {learned.consistent?.[r.slot] ? 'EVET' : 'hayır'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ─────────────────────── KANIT ─────────────────────── */
function ScopeLimits({ limits }: { limits: any[] }) {
  if (!limits?.length) return null
  const renk = (d: string) => d === 'KAPSAM DIŞI' ? C.muted
    : d === 'YOK' ? C.danger : C.warn
  return (
    <Section title="KAPSAM SINIRLARI — İSTENİP DE YAPILMAYAN HER ŞEY">
      <div style={{fontSize:9,color:C.muted,marginBottom:5,lineHeight:1.6}}>
        Bir şartname maddesinin sessizce atlanması, o özelliğin çalıştığı
        izlenimi bırakır. Aşağıdakiler <b style={{color:C.text}}>bilerek</b> dışarıda
        ve gerekçeleri yazılı. Hiçbiri "etkisi yok" anlamına gelmez.
      </div>
      <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
        {limits.map((l, i) => (
          <div key={i} style={{background:C.surface,border:`1px solid ${C.border}`,
               borderRadius:6,padding:'5px 8px',fontSize:9.5,maxWidth:290}}>
            <div style={{fontWeight:700}}>{l.item}</div>
            <div style={{color:renk(l.state),fontWeight:700,fontSize:9}}>{l.state}</div>
            <div style={{color:C.muted,fontSize:8.5,marginTop:2,lineHeight:1.5}}>
              {l.why}</div>
          </div>
        ))}
      </div>
    </Section>
  )
}

function Evidence({ ev, uni, limits }: { ev: any; uni: any; limits: any[] }) {
  const [sel, setSel] = useState<string>('')
  if (!ev) return <Empty text="Kanıt yükleniyor…" />
  if (!ev.available) {
    return (
      <div>
        <div style={{fontSize:10,color:C.warn}}>
          <b>DOĞRULAMA RAPORU YOK</b> — {ev.reason}. Rapor olmadan hiçbir model
          QUALIFIED olamaz (şartname 85).
        </div>
        <ScopeLimits limits={limits} />
      </div>
    )
  }
  const anahtarlar = Object.keys(ev.models || {})
  const k = sel || anahtarlar[0] || ''
  const m = (ev.models || {})[k] || {}
  const lt = m.locked_test || {}
  return (
    <div>
      <div style={{display:'flex',gap:12,flexWrap:'wrap',alignItems:'center',marginBottom:8}}>
        <FlaskConical size={13} color={C.cyan} />
        <span style={{fontWeight:800,fontSize:11}}>KANIT & DOĞRULAMA</span>
        <Stat k="TABAN ÇÖZÜNÜRLÜK" v={ev.base_resolution} />
        <Stat k="HÜCRE" v={ev.n_cells} />
        <Stat k="DENEME KAYDI" v={ev.n_trials_registry} c={C.warn} />
        <Stat k="DENEME SHARPE DAĞILIMI" v={ev.trial_dispersion_sharpe} />
      </div>
      <div style={{fontSize:9.5,color:C.muted,marginBottom:8,lineHeight:1.6,maxWidth:900}}>
        Bölme: eğitim {ev.split?.train} · doğrulama {ev.split?.validation} ·{' '}
        <b style={{color:C.text}}>kilitli test {ev.split?.locked_test}</b>. Kilitli test
        eşik aramak için KULLANILMAZ; seçim eşiği doğrulama döneminden gelir.
        Deflated Sharpe {ev.n_trials_registry} denemeyi hesaba katar — deneme kaydı
        olmadan DSR anlamsızdır.
      </div>

      <div style={{display:'flex',gap:8,flexWrap:'wrap',marginBottom:8}}>
        {Object.entries(ev.status_distribution || {}).map(([s, n]: any) => (
          <Badge key={s} text={`${STATUS_TR[s] || s}: ${n}`}
                 color={STATUS_COLOR[s] || C.muted} />
        ))}
      </div>

      <div style={{display:'flex',gap:4,flexWrap:'wrap',marginBottom:8}}>
        {anahtarlar.map(a => (
          <button key={a} className="cm-btn" onClick={() => setSel(a)}
            style={{background:a===k?C.info:C.surface,color:a===k?'#0A0E1A':C.muted,fontSize:8.5}}>
            {a}</button>
        ))}
      </div>

      {m.ok ? (
        <div style={{display:'grid',gap:8}}>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(130px,1fr))',gap:6}}>
            <Cell k="OOS ÖRNEK" v={m.n_oos} />
            <Cell k="KATLAMA" v={m.folds} />
            <Cell k="OOS TABAN" v={pct(m.oos_baseline)} />
            <Cell k="OOS BRIER" v={num(m.oos_brier, 5)}
                  c={m.oos_brier < m.oos_brier_base ? C.neon : C.danger} />
            <Cell k="TABAN BRIER" v={num(m.oos_brier_base, 5)} />
            <Cell k="OOS ECE" v={num(m.oos_ece, 4)} />
            <Cell k="KALİBRASYON EĞİMİ" v={num(m.oos_calibration?.slope, 3)} />
            <Cell k="PSI (SÜRÜKLENME)" v={num(m.psi, 3)} />
          </div>

          <Section title="KİLİTLİ TEST — MODEL SEÇTİĞİNDE GERÇEKLEŞEN">
            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(130px,1fr))',gap:6}}>
              <Cell k="ÖRNEK" v={lt.n} />
              <Cell k="TABAN P" v={pct(lt.baseline)} />
              <Cell k="SEÇİLEN ORAN" v={pct(lt.selected_frac)} />
              <Cell k="SEÇİLENDE P(HEDEF)" v={pct(lt.selected?.tp_rate)} c={C.neon} />
              <Cell k="SEÇİLENDE P(STOP)" v={pct(lt.selected?.sl_rate)} c={C.danger} />
              <Cell k="GERÇEKLEŞEN NET" v={`${num(lt.selected?.net_mean_pct, 4)}%`}
                    c={(lt.selected?.net_mean_pct ?? -1) > 0 ? C.neon : C.danger} />
              <Cell k="t" v={num(lt.selected?.t_stat, 2)} />
              <Cell k="SEÇİLENDE ORT. STOP" v={`${num(lt.selected?.stop_pct_mean, 3)}%`} />
              <Cell k="POZİTİF ALT DÖNEM" v={pct(lt.positive_subperiod_frac)} />
            </div>
            {lt.selected && (
              <div style={{marginTop:6,fontSize:9.5,color:'#9AA6C4',lineHeight:1.6}}>
                Model hedef olasılığını tabandan <b style={{color:C.text}}>
                {pct(lt.baseline)} → {pct(lt.selected.tp_rate)}</b> yükseltiyor; bu
                gerçek bir lift. Ama seçilen bölgede stop da genişliyor
                (ort. %{num(lt.selected.stop_pct_mean, 2)}), bu yüzden
                gerçekleşen net getiri <b style={{color:(lt.selected.net_mean_pct ?? -1) > 0 ? C.neon : C.danger}}>
                %{num(lt.selected.net_mean_pct, 4)}</b>. Olasılığı yükseltmek tek başına
                kâr üretmez — bu yüzden EV olasılıklardan değil, gerçekleşen getiriden ölçülür.
              </div>
            )}
            {(lt.subperiods || []).length > 0 && (
              <div style={{marginTop:6,display:'flex',gap:6,flexWrap:'wrap'}}>
                {lt.subperiods.map((s: any) => (
                  <span key={s.period} className="mono"
                    style={{fontSize:9,padding:'2px 6px',borderRadius:4,background:C.surface,
                            color:s.net_mean > 0 ? C.neon : C.danger}}>
                    {s.period}: {num(s.net_mean, 3)}% (n={s.n})</span>
                ))}
              </div>
            )}
          </Section>

          {m.ablation && (
            <Section title="ABLASYON — ZAMAN DİLİMİ VE ÖZELLİK AİLESİ">
              <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
                <div>
                  <div style={{fontSize:9,color:C.muted,marginBottom:3}}>ZAMAN DİLİMİ KÜMESİ</div>
                  <table style={{borderCollapse:'collapse',fontSize:9.5}}>
                    <tbody>
                      {(m.ablation.timeframe_sets || []).map((r: any) => (
                        <tr key={r.tf_set}>
                          <td className="mono" style={{padding:'2px 8px 2px 0',fontWeight:700}}>
                            {r.tf_set}</td>
                          <td style={{padding:'2px 8px 2px 0',color:C.muted}}>
                            {r.n_features} özellik</td>
                          <td className="mono" style={{padding:'2px 8px 2px 0'}}>
                            Brier {num(r.oos_brier, 5)}</td>
                          <td className="mono" style={{padding:'2px 0',color:C.muted}}>
                            üst desil {pct(r.top_decile)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <div style={{fontSize:9,color:C.muted,marginBottom:3}}>
                    AİLE ÇIKARILINCA (ΔBrier &gt; 0 = katkı var)</div>
                  <table style={{borderCollapse:'collapse',fontSize:9.5}}>
                    <tbody>
                      {(m.ablation.families || []).map((r: any) => (
                        <tr key={r.removed}>
                          <td className="mono" style={{padding:'2px 8px 2px 0'}}>−{r.removed}</td>
                          <td className="mono" style={{padding:'2px 8px 2px 0',
                               color:r.contributes ? C.neon : C.danger}}>
                            {r.delta_brier > 0 ? '+' : ''}{num(r.delta_brier, 6)}</td>
                          <td style={{padding:'2px 0',fontSize:8.5,
                               color:r.contributes ? C.neon : C.danger}}>
                            {r.contributes ? 'KATKI VAR' : 'KATKI YOK → BUDANDI'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              {(m.pruned_families || []).length > 0 && (
                <div style={{marginTop:5,fontSize:9,color:C.muted}}>
                  Budanan aileler: <b style={{color:C.text}}>{m.pruned_families.join(', ')}</b> —
                  önem skoru değil, OOS katkı ölçütüyle çıkarıldı (şartname 35).
                </div>
              )}
            </Section>
          )}

          <Section title="GÜVENİLİRLİK — MODEL '%X' DEDİĞİNDE GERÇEKTE NE OLDU">
            <RelTable rows={lt.reliability || m.oos_reliability} />
          </Section>
        </div>
      ) : (
        <div style={{fontSize:10,color:C.warn}}>Bu ufuk için model kurulamadı: {m.reason}</div>
      )}

      <Section title="VERİ HAZIRLIK MATRİSİ — 'VERİ YOK' İLE 'ETKİ YOK' AYNI ŞEY DEĞİLDİR">
        <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
          {(ev.data_readiness || []).map((r: any) => (
            <div key={r.source} style={{background:C.surface,border:`1px solid ${C.border}`,
                 borderRadius:6,padding:'5px 8px',fontSize:9.5,minWidth:180}}>
              <div style={{fontWeight:700}}>{r.source}</div>
              <div style={{color:r.state.includes('MISSING') ? C.danger
                          : r.state === 'PARTIAL' ? C.warn : C.neon,fontWeight:700,fontSize:9}}>
                {r.state}</div>
              <div style={{color:C.muted,fontSize:8.5,marginTop:2}}>{r.note}</div>
            </div>
          ))}
        </div>
      </Section>

      <ScopeLimits limits={limits} />

      {uni?.gates && (
        <Section title="EVREN TARAMASI — HANGİ KAPI KAÇ MARKETİ DÜŞÜRDÜ">
          <div style={{display:'flex',gap:12,flexWrap:'wrap',marginBottom:5}}>
            <Stat k="TARANAN" v={uni.scanned} /><Stat k="UYGUN" v={uni.eligible} c={C.neon} />
            <Stat k="DIŞLANAN" v={uni.excluded} c={C.danger} />
          </div>
          <div style={{display:'flex',gap:5,flexWrap:'wrap'}}>
            {uni.gates.map((g: any) => (
              <span key={g.gate} style={{fontSize:9,padding:'2px 7px',borderRadius:4,
                    background:C.surface,border:`1px solid ${C.border}`}}>
                {g.description}: <b style={{color:g.failed_count ? C.danger : C.neon}}>
                  {g.failed_count}</b>
              </span>
            ))}
          </div>
          <div style={{marginTop:5,fontSize:9,color:C.muted}}>{uni.note}</div>
        </Section>
      )}
    </div>
  )
}

function RelTable({ rows }: { rows: any[] }) {
  const dolu = (rows || []).filter(r => r.n > 0)
  if (!dolu.length) return <span style={{fontSize:9.5,color:C.muted}}>yeterli örnek yok</span>
  return (
    <table style={{borderCollapse:'collapse',fontSize:9.5}}>
      <thead><tr style={{color:C.muted,textAlign:'left'}}>
        {['KOVA','N','TAHMİN','GERÇEK','FARK'].map(h =>
          <th key={h} style={{padding:'2px 10px 2px 0'}}>{h}</th>)}
      </tr></thead>
      <tbody>
        {dolu.map(r => {
          const f = (r.actual != null && r.predicted != null) ? r.actual - r.predicted : null
          return (
            <tr key={r.bucket}>
              <td className="mono" style={{padding:'2px 10px 2px 0'}}>{r.bucket}</td>
              <td className="mono" style={{padding:'2px 10px 2px 0'}}>{r.n}</td>
              <td className="mono" style={{padding:'2px 10px 2px 0'}}>{pct(r.predicted)}</td>
              <td className="mono" style={{padding:'2px 10px 2px 0'}}>{pct(r.actual)}</td>
              <td className="mono" style={{padding:'2px 0',
                   color:f == null ? C.muted : Math.abs(f) < 0.05 ? C.neon : C.warn}}>
                {f == null ? '—' : `${f > 0 ? '+' : ''}${(f * 100).toFixed(1)} p`}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/* ─────────────────────── TAHMİN KAYDI ─────────────────────── */
function LedgerPanel({ led }: { led: any }) {
  if (!led) return <Empty text="Kayıt yükleniyor…" />
  const bugun = led.today || {}
  const hucreler: any[] = led.scorecard_full?.cells || []
  return (
    <div>
      <div style={{display:'flex',gap:12,flexWrap:'wrap',alignItems:'center',marginBottom:8}}>
        <ShieldCheck size={13} color={C.info} />
        <span style={{fontWeight:800,fontSize:11}}>DEĞİŞMEZ TAHMİN KAYDI</span>
        <Stat k="YAYIMLANAN" v={led.n_predictions} />
        <Stat k="GÖLGE" v={led.n_shadow ?? 0} />
        <Stat k="ÇÖZÜLEN" v={led.n_resolved} />
      </div>
      <div style={{fontSize:9.5,color:C.muted,marginBottom:8}}>{led.note}</div>
      {led.shadow_note && (
        <div style={{fontSize:9,color:C.info,marginBottom:8,lineHeight:1.6,
                     borderLeft:`2px solid ${C.info}55`,paddingLeft:8}}>
          {led.shadow_note}</div>)}

      <Section title={`BUGÜN (${bugun.date || '—'} · ${bugun.timezone || ''})`}>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(110px,1fr))',gap:6}}>
          <Cell k="YAYIMLANAN" v={bugun.qualified_seen ?? 0} />
          <Cell k="AKTİF" v={bugun.active ?? 0} />
          <Cell k="SÜRESİ DOLAN" v={bugun.expired ?? 0} />
          <Cell k="HEDEF ÖNCE" v={bugun.tp_first ?? 0} c={C.neon} />
          <Cell k="STOP ÖNCE" v={bugun.sl_first ?? 0} c={C.danger} />
          <Cell k="ZAMAN AŞIMI" v={bugun.timeout ?? 0} />
          <Cell k="GERÇEKLEŞEN NET" v={`${num(bugun.realized_net_pct, 3)}%`} />
          <Cell k="YANLIŞ FIRSAT ORANI" v={pct(bugun.false_opportunity_rate)} />
        </div>
      </Section>

      {hucreler.length > 0 ? (
        <Section title="PARİTE × UFUK KARNESİ (payda: YAYIMLANAN sinyal)">
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:9.5,minWidth:700}}>
              <thead><tr style={{color:C.muted,textAlign:'left'}}>
                {['PARİTE','UFUK','YÖN','YAYIN','ÇÖZÜLEN','TP','SL','TO','İŞLEMSİZ',
                  'NET1 KESİNLİK','YANLIŞ FIRSAT','GERÇEK NET','KALİBRASYON HATASI']
                  .map(h => <th key={h} style={{padding:'3px 5px',
                    borderBottom:`1px solid ${C.border}`}}>{h}</th>)}
              </tr></thead>
              <tbody>
                {hucreler.map((r, i) => (
                  <tr key={i} style={{borderBottom:`1px solid ${C.border}44`}}>
                    <td style={{padding:'3px 5px',fontWeight:700}}>{r.symbol}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{r.horizon}</td>
                    <td style={{padding:'3px 5px',color:r.direction==='LONG'?C.neon:C.danger}}>
                      {r.direction}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{r.published}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{r.resolved}</td>
                    <td className="mono" style={{padding:'3px 5px',color:C.neon}}>{r.tp_first}</td>
                    <td className="mono" style={{padding:'3px 5px',color:C.danger}}>{r.sl_first}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{r.timeout}</td>
                    <td className="mono" style={{padding:'3px 5px',color:C.muted}}>{r.not_traded}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{pct(r.net1_precision)}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{pct(r.false_opportunity_rate)}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{num(r.realized_net_mean_pct,3)}</td>
                    <td className="mono" style={{padding:'3px 5px'}}>{num(r.calibration_error,3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      ) : (
        <Empty text="Henüz yayımlanmış tahmin yok — qualified bir kurulum çıkmadı." />
      )}

      <Section title="CANLI KALİBRASYON TABLOSU">
        <RelTable rows={led.calibration_board} />
      </Section>
    </div>
  )
}

/* ─────────────────────── ortak ─────────────────────── */
function Section({ title, children }: { title: string; children: any }) {
  return (
    <div style={{marginTop:8}}>
      <div style={{fontSize:9,color:C.muted,letterSpacing:0.6,marginBottom:4,
                   borderBottom:`1px solid ${C.border}`,paddingBottom:3}}>{title}</div>
      {children}
    </div>
  )
}

function Empty({ text }: { text: string }) {
  return (
    <div style={{display:'flex',alignItems:'center',gap:6,fontSize:10,color:C.muted,
                 padding:'8px 0'}}>
      <AlertTriangle size={12} color={C.muted} /> {text}
    </div>
  )
}

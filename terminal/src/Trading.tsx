/**
 * BORSA BAĞLANTILARI & OTOPİLOT — panelin en üst bölümü.
 *
 * Kaynak: /account/trading/* (oturum + CSRF). Bu bileşen HİÇBİR anahtarı
 * saklamaz ya da gösterir; sunucu yalnız son 4 haneyi döndürür.
 *
 * Üç mod, üç ayrı kapı:
 *   PAPER    sanal sermaye (videodaki "önce 5.000 $ ile self-test")
 *   TESTNET  borsanın sandbox'ı — İŞLEM kapsamlı anahtar gerekir
 *   CANLI    operatör kapısı + onay cümlesi + paper kanıtı + anahtar yeniden doğrulama
 *
 * Panel sayı ÜRETMEZ; her değer API'den gelir, gelmiyorsa "—".
 */
import { useState, useEffect, useCallback } from 'react'
import { KeyRound, Lock, LogOut, Trash2, Plug, ShieldCheck, ShieldAlert, Play, Square,
         RotateCcw, Unlock, Bot, Video, AlertTriangle, ChevronDown, ChevronUp,
         Coins, Activity } from 'lucide-react'
import EquityChart from './EquityChart'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E',
            surface:'#1A2240', bg:'#0A0E1A', cyan:'#22D3EE', violet:'#A78BFA' }

const csrf = () =>
  document.cookie.split('; ').find(c => c.startsWith('cm_csrf='))?.split('=')[1] || ''

async function api(path: string, method = 'GET', body?: any) {
  try {
    const r = await fetch(`${API}/account${path}`, {
      method, credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json',
                 ...(method !== 'GET' ? { 'X-CSRF-Token': csrf() } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    })
    const d = await r.json().catch(() => null)
    // JSON olmayan gövde (nginx 502/504 HTML) = API yeniden başlıyor → GEÇİCİ
    return { ok: r.ok && d !== null, status: r.status, data: d ?? {}, transient: d === null }
  } catch {
    return { ok: false, status: 0, data: {}, transient: true }
  }
}

const usd = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—'
  : `${Number(x) < 0 ? '−' : ''}$${Math.abs(Number(x)).toLocaleString('tr-TR', { maximumFractionDigits: d, minimumFractionDigits: d })}`
const pct = (x: any, d = 2) => (x == null || !isFinite(x)) ? '—' : `${Number(x) >= 0 ? '+' : ''}${Number(x).toFixed(d)}%`
const ago = (ts: any) => ts ? `${Math.max(0, Math.round(Date.now() / 1000 - ts))} sn önce` : '—'
const MODE_TR: Record<string, string> = { paper: 'PAPER', testnet: 'TESTNET', live: 'CANLI' }
const MODE_COL: Record<string, string> = { paper: C.info, testnet: C.warn, live: C.danger }

export default function Trading() {
  const [cat, setCat] = useState<any>(null)
  const [sel, setSel] = useState('binance')
  const [st, setSt] = useState<any>(null)
  const [keys, setKeys] = useState<any>({ keys: [], scopes: {} })
  const [email, setEmail] = useState(''); const [pw, setPw] = useState('')
  const [msg, setMsg] = useState<{ t: string; ok: boolean } | null>(null)
  const [busy, setBusy] = useState('')
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [scope, setScope] = useState<'read' | 'trade'>('trade')
  const [ack, setAck] = useState(false)
  const [form, setForm] = useState<any>(null)
  const [confirm, setConfirm] = useState('')
  const [open, setOpen] = useState(true)
  const [showRules, setShowRules] = useState(false)
  const [showParams, setShowParams] = useState(false)
  const [net, setNet] = useState<'ok' | 'down'>('ok')

  // KURAL: geçici bir API hatası (yeniden başlatma, 502, ağ) mevcut durumu SİLMEZ.
  // Önceki sürümde tek başarısız istek `cat`'i boşaltıyor, kartlar kaybolup giriş
  // formu geri geliyordu. Artık yalnız GEÇERLİ yanıt durumu değiştirir.
  const pullCatalog = useCallback(async () => {
    const r = await api('/trading/catalog')
    if (!r.ok || typeof r.data?.logged_in !== 'boolean') { setNet('down'); return }
    setNet('ok'); setCat(r.data)
    if (r.data.logged_in) {
      const k = await api('/keys')
      if (k.ok) setKeys(k.data || { keys: [], scopes: {} })
    } else setKeys({ keys: [], scopes: {} })
  }, [])
  const pullState = useCallback(async () => {
    if (!cat?.logged_in) return
    const r = await api(`/trading/state?exchange=${encodeURIComponent(sel)}`)
    if (r.ok) setSt(r.data)
  }, [cat?.logged_in, sel])

  useEffect(() => {
    pullCatalog()
    const iv = setInterval(pullCatalog, 30000)          // kopan bağlantı kendiliğinden toparlanır
    const vis = () => { if (document.visibilityState === 'visible') pullCatalog() }
    document.addEventListener('visibilitychange', vis)
    return () => { clearInterval(iv); document.removeEventListener('visibilitychange', vis) }
  }, [pullCatalog])
  useEffect(() => {
    pullState()
    const iv = setInterval(pullState, st?.running ? 5000 : 15000)
    return () => clearInterval(iv)
  }, [pullState, st?.running])
  useEffect(() => {
    if (cat?.defaults && !form) setForm({ ...cat.defaults, mode: 'paper' })
  }, [cat, form])
  useEffect(() => {
    // kurulu koşucu varsa formu onun ayarlarıyla doldur
    if (st?.configured && st.config) setForm((f: any) => ({ ...(f || {}), ...st.config }))
  }, [st?.configured, st?.exchange, st?.config?.mode]) // eslint-disable-line

  const ex = cat?.exchanges?.find((e: any) => e.id === sel)
  const exScope = keys.scopes?.[sel]
  const hasKey = !!ex?.has_key
  const liveGate = cat?.live_gate
  const running = !!st?.running

  const doAuth = async (path: string) => {
    setBusy(path); setMsg(null)
    const r = await api(path, 'POST', { email, password: pw })
    setBusy('')
    setMsg({ t: r.ok ? 'Giriş başarılı.' : (r.data?.error || 'hata'), ok: r.ok })
    if (r.ok) { setPw(''); await pullCatalog() }
  }

  const saveKey = async () => {
    if (!ex) return
    const f = ex.fields
    const apiKey = (draft[f[0].key] || '').trim(), secret = (draft[f[1]?.key] || '').trim()
    const password = f[2] ? (draft[f[2].key] || '').trim() : undefined
    if (!apiKey || !secret) { setMsg({ t: 'API key ve secret gerekli', ok: false }); return }
    if (scope === 'trade' && !ack) { setMsg({ t: 'İŞLEM kapsamı için "para çekme izni KAPALI" onayı gerekli', ok: false }); return }
    setBusy('save'); setMsg(null)
    const r = await api('/keys', 'POST', {
      provider: ex.id, exchange_id: ex.ccxt_id, scope, withdraw_disabled_ack: ack,
      fields: { apiKey, secret, ...(password ? { password } : {}) } })
    setBusy('')
    setMsg({ t: r.ok ? `${ex.name}: ${r.data?.permissions?.reason || 'kaydedildi'}` : (r.data?.error || 'hata'), ok: r.ok })
    if (r.ok) { setDraft({}); await pullCatalog() }
  }
  const testKey = async () => {
    if (!ex) return
    setBusy('test')
    const r = await api('/keys/test', 'POST', { provider: ex.id, exchange_id: ex.ccxt_id })
    setBusy('')
    setMsg({ t: `${ex.name}: ${r.data?.permissions?.reason || r.data?.error || 'test edildi'}`, ok: !!r.ok })
  }
  const delKey = async () => {
    if (!ex) return
    setBusy('del')
    const r = await api('/keys/delete', 'POST', { provider: ex.id })
    setBusy('')
    setMsg({ t: r.ok ? `${ex.name}: anahtar silindi` : (r.data?.error || 'hata'), ok: r.ok })
    await pullCatalog()
  }

  const act = async (path: string, body: any = {}) => {
    setBusy(path); setMsg(null)
    const r = await api(`/trading/${path}`, 'POST', { exchange: sel, ...body })
    setBusy('')
    if (r.ok) { setSt(r.data.state || st); setMsg({ t: `${path}: tamam`, ok: true }) }
    else {
      const bl = r.data?.blockers ? ' — ' + r.data.blockers.join(' · ') : ''
      setMsg({ t: (r.data?.error || `${path} başarısız`) + bl, ok: false })
    }
    await pullCatalog(); await pullState()
  }
  const start = () => act('start', { mode: form.mode, config: form,
                                     ...(form.mode === 'live' ? { confirm_phrase: confirm } : {}) })

  const setP = (k: string, v: any) => setForm((f: any) => ({ ...f, params: { ...f.params, [k]: v } }))
  const setCh = (k: string, v: any) => setForm((f: any) => ({ ...f, chain: { ...f.chain, [k]: v } }))
  const setF = (k: string, v: any) => setForm((f: any) => ({ ...f, [k]: v }))
  const loadVideoPreset = () => {
    const p = cat?.strategy?.video_preset || {}
    setForm((f: any) => ({ ...f, params: { ...f.params, ...p },
                           capital_usdt: cat?.strategy?.video_capital_usdt || f.capital_usdt,
                           daily_loss_limit_pct: cat?.strategy?.video_daily_loss_pct || f.daily_loss_limit_pct }))
    setMsg({ t: 'Videodaki ayarlar yüklendi (1:1,6 · %50 geri-verme · 15-60 dk · 30 sn · 5.000 $ · günlük −%20). Günlük −%20 videonun seçimidir; CryptoMind varsayılanı −%5.', ok: true })
  }

  const stats = st?.stats
  const rd = st?.readiness

  return (
    <section id="otopilot" style={{ borderBottom: `1px solid ${C.border}`, background: '#0C1120' }}>
      {/* ── BAŞLIK BANDI ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
                    padding: '8px 16px', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Bot size={15} color={C.cyan} />
          <span style={{ fontWeight: 800, letterSpacing: 1, fontSize: 12 }}>BORSA BAĞLANTILARI & OTOPİLOT</span>
        </div>
        {net === 'down' && <Badge text="API YANIT VERMİYOR — yeniden deneniyor" col={C.danger} />}
        <Badge text={cat?.vault_ready ? 'KASA HAZIR' : 'KASA KİLİTLİ'} col={cat?.vault_ready ? C.neon : C.danger} />
        <Badge text={cat?.logged_in ? `OTURUM: ${cat.email}` : 'OTURUM YOK'} col={cat?.logged_in ? C.neon : C.muted} />
        <Badge text={liveGate?.live ? 'CANLI KAPISI: AÇIK' : 'CANLI KAPISI: KAPALI'}
               col={liveGate?.live ? C.danger : C.muted}
               title={liveGate?.live ? 'Sunucu operatörü canlı emri açmış' : (liveGate?.missing || []).join(' · ')} />
        {cat?.exchanges?.filter((e: any) => e.runner).map((e: any) => (
          <Badge key={e.id} text={`${e.name}: ${MODE_TR[e.runner.mode]}${e.runner.halted ? ' HALT' : e.runner.running ? ' ▶' : ' ■'}`}
                 col={e.runner.halted ? C.danger : e.runner.running ? MODE_COL[e.runner.mode] : C.muted} />
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          {cat?.logged_in && (
            <button className="cm-btn" style={{ background: C.surface, color: C.muted }}
              onClick={async () => { await api('/logout', 'POST'); setSt(null); await pullCatalog() }}>
              <LogOut size={10} style={{ verticalAlign: -1 }} /> ÇIKIŞ
            </button>
          )}
          <button className="cm-btn" style={{ background: C.surface, color: C.muted }} onClick={() => setOpen(o => !o)}>
            {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
        </div>
      </div>

      {open && (
        <div style={{ padding: '10px 16px 12px' }}>
          {/* ── GİRİŞ ── */}
          {!cat?.logged_in && (
            <div className="cm-card" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <Lock size={13} color={C.muted} />
              <span className="cm-title">{cat?.bootstrap_needed || cat?.user_count === 0 ? 'İLK KULLANICIYI OLUŞTUR' : 'GİRİŞ — borsa anahtarlarınız ve otopilot yalnız oturumunuzda görünür'}</span>
              <input placeholder="e-posta" value={email} onChange={e => setEmail(e.target.value)} style={inp} />
              <input placeholder="parola (min 12 karakter)" type="password" value={pw}
                     onChange={e => setPw(e.target.value)} style={inp}
                     onKeyDown={e => e.key === 'Enter' && doAuth('/login')} />
              <button className="cm-btn" disabled={!!busy} onClick={() => doAuth('/login')}
                      style={{ background: C.neon, color: C.bg }}>GİRİŞ</button>
              <button className="cm-btn" disabled={!!busy} onClick={() => doAuth('/register')}
                      style={{ background: C.surface, color: C.info }}>OLUŞTUR</button>
              {!cat?.vault_ready && <span style={{ fontSize: 9, color: C.danger }}>{cat?.vault_note}</span>}
            </div>
          )}

          {msg && (
            <div className="cm-card" style={{ borderColor: (msg.ok ? C.neon : C.danger) + '66',
                  color: msg.ok ? C.neon : C.danger, fontSize: 11, padding: '6px 10px' }}>{msg.t}</div>
          )}

          {/* ── BORSA KARTLARI — oturumdan bağımsız görünür; giriş yapılmadıysa yalnız liste ── */}
          {cat?.exchanges && (
              <div className="cm-strip" style={{ marginBottom: 8 }}>
                {cat.exchanges.map((e: any) => {
                  const isSel = e.id === sel
                  const sc = e.scope
                  const rn = e.runner
                  return (
                    <div key={e.id} onClick={() => { setSel(e.id); setSt(null); setDraft({}) }}
                      style={{ cursor: 'pointer', minWidth: 150, padding: '7px 10px', borderRadius: 8, flexShrink: 0,
                               background: isSel ? 'rgba(34,211,238,0.08)' : C.panel,
                               border: `1px solid ${isSel ? C.cyan + '99' : e.has_key ? C.neon + '44' : C.border}` }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ width: 18, height: 18, borderRadius: 5, display: 'grid', placeItems: 'center',
                                       background: C.surface, fontSize: 10, fontWeight: 900, color: C.text }}>
                          {e.name[0]}
                        </span>
                        <b style={{ fontSize: 11 }}>{e.name}</b>
                        {e.sandbox && <span title="Testnet/sandbox destekler" style={{ fontSize: 7.5, color: C.warn }}>TESTNET</span>}
                      </div>
                      <div style={{ display: 'flex', gap: 4, marginTop: 5, flexWrap: 'wrap' }}>
                        <Badge small text={!cat.logged_in ? 'GİRİŞ GEREKLİ' : !e.has_key ? 'ANAHTAR YOK' : sc === 'trade' ? 'İŞLEM ANAHTARI' : 'SADECE-OKUMA'}
                               col={!cat.logged_in ? C.muted : !e.has_key ? C.muted : sc === 'trade' ? C.neon : C.info} />
                        {rn && <Badge small text={`${MODE_TR[rn.mode]}${rn.halted ? ' HALT' : rn.running ? ' ▶' : ' ■'}`}
                                      col={rn.halted ? C.danger : rn.running ? MODE_COL[rn.mode] : C.muted} />}
                      </div>
                      {rn && (
                        <div className="mono" style={{ fontSize: 9, marginTop: 4, color: rn.net_pnl >= 0 ? C.neon : C.danger }}>
                          net {usd(rn.net_pnl)} · {rn.closed_trades} işlem
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
          )}

          {cat?.logged_in && form && (
            <>
              {ex && (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) minmax(320px, 1.6fr)', gap: 10 }}
                     className="cm-trading-grid">
                  {/* ── ANAHTAR PANELİ ── */}
                  <div className="cm-card" style={{ margin: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                      <KeyRound size={13} color={C.info} />
                      <span className="cm-title">{ex.name.toUpperCase()} — API ANAHTARI</span>
                      {hasKey && <span style={{ fontSize: 8, color: C.neon }}>● kayıtlı ({exScope === 'trade' ? 'işlem' : 'okuma'})</span>}
                      {ex.signup_url && <a href={ex.signup_url} target="_blank" rel="noreferrer"
                        style={{ marginLeft: 'auto', fontSize: 9, color: C.info }}>anahtar al →</a>}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {ex.fields.map((f: any) => {
                        const s = keys.keys?.find((k: any) => k.provider === ex.id &&
                          (k.field === (f.key.endsWith('_API_KEY') ? 'apiKey' : f.key.endsWith('_SECRET') ? 'secret' : 'password')))
                        return (
                          <input key={f.key} type={f.secret ? 'password' : 'text'} autoComplete="off"
                            placeholder={s ? `${f.label} — kayıtlı ${s.masked}` : f.label}
                            value={draft[f.key] || ''} onChange={e => setDraft(d => ({ ...d, [f.key]: e.target.value }))}
                            style={{ ...inp, width: '100%' }} />
                        )
                      })}
                    </div>
                    <div style={{ display: 'flex', gap: 10, marginTop: 8, fontSize: 10, flexWrap: 'wrap' }}>
                      <label style={{ display: 'flex', gap: 4, alignItems: 'center', cursor: 'pointer' }}>
                        <input type="radio" checked={scope === 'trade'} onChange={() => setScope('trade')} />
                        <b style={{ color: C.neon }}>İŞLEM</b> <span style={{ color: C.muted }}>(otopilot emir açar)</span>
                      </label>
                      <label style={{ display: 'flex', gap: 4, alignItems: 'center', cursor: 'pointer' }}>
                        <input type="radio" checked={scope === 'read'} onChange={() => setScope('read')} />
                        <b style={{ color: C.info }}>SADECE-OKUMA</b> <span style={{ color: C.muted }}>(bakiye görünür)</span>
                      </label>
                    </div>
                    {scope === 'trade' && (
                      <label style={{ display: 'flex', gap: 6, alignItems: 'flex-start', marginTop: 6, fontSize: 9.5, cursor: 'pointer' }}>
                        <input type="checkbox" checked={ack} onChange={e => setAck(e.target.checked)} style={{ marginTop: 2 }} />
                        <span>Bu anahtarın <b style={{ color: C.danger }}>para çekme izni KAPALI</b>; yalnız spot işlem izni açık.
                          IP kısıtlaması ekledim. <span style={{ color: C.muted }}>(Binance'te sunucu bunu ayrıca doğrular; para çekme izinli anahtar her koşulda reddedilir.)</span></span>
                      </label>
                    )}
                    <div style={{ display: 'flex', gap: 5, marginTop: 8 }}>
                      <button className="cm-btn" disabled={busy === 'save'} onClick={saveKey}
                              style={{ background: C.neon, color: C.bg }}>KAYDET & DOĞRULA</button>
                      {hasKey && <>
                        <button className="cm-btn" disabled={busy === 'test'} onClick={testKey}
                                style={{ background: C.surface, color: C.info }}>
                          <Plug size={10} style={{ verticalAlign: -1 }} /> TEST</button>
                        <button className="cm-btn" disabled={busy === 'del'} onClick={delKey}
                                style={{ background: 'transparent', color: C.danger, border: `1px solid ${C.danger}55` }}>
                          <Trash2 size={10} style={{ verticalAlign: -1 }} /> SİL</button>
                      </>}
                    </div>
                    <div style={{ fontSize: 8.5, color: C.muted, marginTop: 8, lineHeight: 1.6 }}>
                      <ShieldCheck size={9} color={C.neon} style={{ verticalAlign: -1 }} /> AES-256-GCM ile şifrelenir; düz metin hiçbir uçtan dönmez.
                      Kayıtta izinler <b>canlı</b> test edilir: para çekme açıksa RET. {ex.note && `· ${ex.note}`}
                    </div>
                  </div>

                  {/* ── OTOPİLOT PANELİ ── */}
                  <div className="cm-card" style={{ margin: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
                      <Activity size={13} color={C.cyan} />
                      <span className="cm-title">OTOPİLOT — {cat.strategy?.name}</span>
                      <a href={cat.strategy?.source} target="_blank" rel="noreferrer" title="Kaynak video"
                         style={{ fontSize: 9, color: C.danger, display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Video size={11} /> video</a>
                      <button className="cm-btn" onClick={() => setShowRules(s => !s)}
                              style={{ background: C.surface, color: C.muted, fontSize: 8.5 }}>
                        {showRules ? 'kuralları gizle' : 'kurallar & dersler'}</button>
                      <button className="cm-btn" onClick={loadVideoPreset}
                              style={{ background: C.surface, color: C.warn, fontSize: 8.5 }}>VİDEO AYARLARI</button>
                      {st?.configured && (
                        <span className="mono" style={{ marginLeft: 'auto', fontSize: 9, color: C.muted }}>
                          döngü #{st.cycle} · {ago(st.last_cycle_ts)} · her {st.loop_sec} sn
                        </span>
                      )}
                    </div>

                    {showRules && (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 9, marginBottom: 8 }}>
                        <div style={{ background: C.surface, borderRadius: 6, padding: 8 }}>
                          <b style={{ color: C.info }}>Videodaki kurulum</b>
                          {(cat.strategy?.rules || []).map((r: string, i: number) => <div key={i} style={{ color: '#8892B0' }}>▸ {r}</div>)}
                          <b style={{ color: C.warn, display: 'block', marginTop: 4 }}>Videonun ölçtüğü</b>
                          {(cat.strategy?.measured_lessons || []).map((r: string, i: number) => <div key={i} style={{ color: '#8892B0' }}>▸ {r}</div>)}
                        </div>
                        <div style={{ background: C.surface, borderRadius: 6, padding: 8 }}>
                          <b style={{ color: C.neon }}>CryptoMind'ın eklediği kapılar</b>
                          {(cat.strategy?.cryptomind_gates || []).map((r: string, i: number) => <div key={i} style={{ color: '#8892B0' }}>▸ {r}</div>)}
                        </div>
                      </div>
                    )}

                    {/* strateji seçimi */}
                    <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6, fontSize: 9.5 }}>
                      <span style={{ color: C.muted }}>strateji:</span>
                      {([['video_dip_scalp', 'DİP-SCALP (video)'], ['committee', 'KOMİTE (12 rol)']] as const).map(([k, l]) => (
                        <button key={k} className="cm-btn" disabled={running} onClick={() => setF('strategy', k)}
                          style={{ background: (form.strategy || 'video_dip_scalp') === k ? C.violet : C.surface,
                                   color: (form.strategy || 'video_dip_scalp') === k ? C.bg : C.muted }}>{l}</button>
                      ))}
                      <label style={{ display: 'flex', gap: 4, alignItems: 'center', cursor: 'pointer', color: C.muted }}>
                        <input type="checkbox" disabled={running} checked={form.symbols_mode === 'auto'}
                               onChange={e => setF('symbols_mode', e.target.checked ? 'auto' : 'fixed')} />
                        pariteleri CryptoMind seçsin (mover sırası)
                      </label>
                    </div>
                    {/* mod seçimi */}
                    <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', marginBottom: 6 }}>
                      {(['paper', 'testnet', 'live'] as const).map(m => {
                        const dis = running || (m !== 'paper' && (!hasKey || exScope !== 'trade')) || (m === 'testnet' && ex.sandbox === false)
                        return (
                          <button key={m} className="cm-btn" disabled={dis} onClick={() => setF('mode', m)}
                            title={m === 'paper' ? 'Sanal sermaye — gerçek emir yok' :
                                   m === 'testnet' ? (ex.sandbox === false ? 'Bu borsa sandbox desteklemiyor' : 'Borsa sandbox — sahte para, gerçek API') :
                                   'Gerçek hesap — tüm kapılar geçilmeli'}
                            style={{ background: form.mode === m ? MODE_COL[m] : C.surface,
                                     color: form.mode === m ? C.bg : (dis ? C.border : C.muted), opacity: dis ? 0.6 : 1 }}>
                            {MODE_TR[m]}</button>
                        )
                      })}
                      {form.mode !== 'paper' && (!hasKey || exScope !== 'trade') &&
                        <span style={{ fontSize: 9, color: C.warn }}>testnet/canlı için İŞLEM kapsamlı anahtar gerekir</span>}
                      <span style={{ marginLeft: 'auto', fontSize: 9, color: C.muted }}>
                        pariteler: <input value={(form.symbols || []).join(', ')} disabled={running}
                          onChange={e => setF('symbols', e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                          style={{ ...inp, width: 220, fontSize: 10 }} /></span>
                    </div>

                    {/* temel ayarlar */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(118px, 1fr))', gap: 5 }}>
                      <Num label="sermaye $" v={form.capital_usdt} set={v => setF('capital_usdt', v)} dis={running} />
                      <Num label="emir tavanı $" v={form.max_order_usdt} set={v => setF('max_order_usdt', v)} dis={running} />
                      <Num label="maks açık poz." v={form.max_open} set={v => setF('max_open', v)} dis={running} />
                      <Num label="işlem riski %" v={form.risk_per_trade_pct} set={v => setF('risk_per_trade_pct', v)} step={0.1} dis={running} />
                      <Num label="günlük zarar limiti" v={Math.round(form.daily_loss_limit_pct * 1000) / 10} unit="%"
                           set={v => setF('daily_loss_limit_pct', v / 100)} step={0.5} dis={running} />
                      <Num label="maks drawdown" v={Math.round(form.max_drawdown_pct * 1000) / 10} unit="%"
                           set={v => setF('max_drawdown_pct', v / 100)} step={0.5} dis={running} />
                      <Num label="TP/SL oranı (R)" v={form.params?.rr} set={v => setP('rr', v)} step={0.1} />
                      <Num label="geri-verme (tepe %)" v={Math.round((form.params?.giveback || 0) * 100)} unit="%"
                           set={v => setP('giveback', v / 100)} step={5} />
                      <Num label="asgari tutma dk" v={Math.round((form.params?.min_hold_sec || 0) / 60)} set={v => setP('min_hold_sec', v * 60)} />
                      <Num label="azami tutma dk" v={Math.round((form.params?.max_hold_sec || 0) / 60)} set={v => setP('max_hold_sec', v * 60)} />
                      <Num label="döngü sn" v={form.params?.loop_sec} set={v => setP('loop_sec', v)} step={5} />
                      <Num label="komisyon bps" v={form.params?.fee_bps} set={v => setP('fee_bps', v)} step={0.5} />
                    </div>
                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 6, fontSize: 9.5 }}>
                      <Chk label="BNB komisyon indirimi" v={!!form.params?.bnb_discount} set={v => setP('bnb_discount', v)} />
                      <Chk label="Konsensüs vetosu" v={form.chain?.use_consensus !== false} set={v => setCh('use_consensus', v)} />
                      <Chk label="Nitelendirme matrisi" v={form.chain?.use_qualification !== false} set={v => setCh('use_qualification', v)} />
                      <Chk label="Katı nitelendirme (NO_EDGE → veto)" v={!!form.chain?.strict_qualification} set={v => setCh('strict_qualification', v)} />
                      <Chk label="Fırsat kapıları" v={form.chain?.use_opportunity_gates !== false} set={v => setCh('use_opportunity_gates', v)} />
                      <Chk label="Rejim çarpanı" v={form.chain?.use_regime !== false} set={v => setCh('use_regime', v)} />
                      <button className="cm-btn" onClick={() => setShowParams(s => !s)}
                              style={{ background: 'transparent', color: C.muted, fontSize: 8.5 }}>
                        {showParams ? 'gelişmiş ▲' : 'gelişmiş ▼'}</button>
                    </div>
                    {showParams && (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(118px, 1fr))', gap: 5, marginTop: 5 }}>
                        <Num label="dip z-eşiği" v={form.params?.dip_z} set={v => setP('dip_z', v)} step={0.1} />
                        <Num label="RSI tavanı" v={form.params?.rsi_max} set={v => setP('rsi_max', v)} />
                        <Num label="stop σ çarpanı" v={form.params?.stop_sigma_mult} set={v => setP('stop_sigma_mult', v)} step={0.1} />
                        <Num label="brüt/maliyet ≥" v={form.params?.min_gross_to_cost} set={v => { setP('min_gross_to_cost', v); setCh('min_gross_to_cost', v) }} step={0.5} />
                        <Num label="net eşik %" v={form.chain?.min_net_return_pct} set={v => setCh('min_net_return_pct', v)} step={0.05} />
                        <Num label="ters konsensüs güven" v={form.chain?.consensus_veto_conf} set={v => setCh('consensus_veto_conf', v)} step={0.05} />
                        <Num label="maks maruziyet %" v={form.max_exposure_pct} set={v => setF('max_exposure_pct', v)} dis={running} />
                        <Num label="günlük maks işlem" v={form.max_trades_per_day} set={v => setF('max_trades_per_day', v)} dis={running} />
                        <Num label="paper kanıt (işlem)" v={form.paper_proof_trades} set={v => setF('paper_proof_trades', v)} dis={running} />
                      </div>
                    )}

                    {/* canlı kapısı */}
                    {form.mode === 'live' && (
                      <div style={{ marginTop: 8, padding: 8, borderRadius: 6, background: 'rgba(255,59,92,0.06)', border: `1px solid ${C.danger}55`, fontSize: 9.5 }}>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                          <AlertTriangle size={12} color={C.danger} />
                          <b style={{ color: C.danger }}>CANLI MOD — gerçek para. Hepsi birden gerekli:</b>
                        </div>
                        <Gate ok={!!liveGate?.live} text={`sunucu operatör kapısı${liveGate?.live ? '' : ': ' + (liveGate?.missing || []).join(', ')}`} />
                        <Gate ok={hasKey && exScope === 'trade'} text="İŞLEM kapsamlı anahtar (para çekme KAPALI doğrulanmış)" />
                        <Gate ok={!!rd?.ok} text={`paper kanıtı: ${rd ? `${rd.paper_trades}/${rd.required_trades} işlem, net ${usd(rd.paper_net)}` : 'önce PAPER modunda çalıştırın'}`} />
                        <Gate ok={confirm.trim() === cat.confirm_phrase} text={`onay cümlesi: "${cat.confirm_phrase}"`} />
                        <input value={confirm} onChange={e => setConfirm(e.target.value)} placeholder={cat.confirm_phrase}
                               style={{ ...inp, width: '100%', marginTop: 4, borderColor: C.danger + '77' }} />
                      </div>
                    )}

                    {/* kontrol düğmeleri */}
                    <div style={{ display: 'flex', gap: 5, marginTop: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                      {!running ? (
                        <button className="cm-btn" disabled={!!busy} onClick={start}
                                style={{ background: MODE_COL[form.mode], color: C.bg }}>
                          <Play size={10} style={{ verticalAlign: -1 }} /> {st?.configured ? 'YENİDEN BAŞLAT' : 'BAŞLAT'} ({MODE_TR[form.mode]})
                        </button>
                      ) : (
                        <button className="cm-btn" disabled={!!busy} onClick={() => act('stop')}
                                style={{ background: C.surface, color: C.danger, border: `1px solid ${C.danger}66` }}>
                          <Square size={10} style={{ verticalAlign: -1 }} /> DURDUR
                        </button>
                      )}
                      {st?.configured && <>
                        {running && <button className="cm-btn" disabled={!!busy} onClick={() => act('params', { params: form.params, chain: form.chain })}
                                style={{ background: C.surface, color: C.info }}>PARAMETRELERİ UYGULA</button>}
                        <button className="cm-btn" disabled={!!busy || !st.positions?.length} onClick={() => act('close_all')}
                                style={{ background: C.surface, color: C.warn }}>TÜMÜNÜ KAPAT</button>
                        {st.halted && <button className="cm-btn" disabled={!!busy} onClick={() => act('resume')}
                                style={{ background: C.danger, color: C.bg }}>
                          <Unlock size={10} style={{ verticalAlign: -1 }} /> HALT'I KALDIR</button>}
                        {!running && st.mode === 'paper' && <button className="cm-btn" disabled={!!busy} onClick={() => act('reset')}
                                style={{ background: 'transparent', color: C.muted, border: `1px solid ${C.border}` }}>
                          <RotateCcw size={10} style={{ verticalAlign: -1 }} /> PAPER SIFIRLA</button>}
                        {!running && !st.positions?.length && <button className="cm-btn" disabled={!!busy} onClick={() => act('remove')}
                                style={{ background: 'transparent', color: C.muted, border: `1px solid ${C.border}` }}>KALDIR</button>}
                        {st.manage_only && <Badge text="YALNIZ POZİSYON YÖNETİMİ — yeni giriş için BAŞLAT" col={C.warn} />}
                        {st.halted && <Badge text={`HALT: ${(st.halt_reasons || []).join(' · ')}`} col={C.danger} />}
                        {st.reconcile_ok === false && <Badge text={`MUTABAKAT: ${st.reconcile_note}`} col={C.danger} />}
                      </>}
                    </div>
                  </div>
                </div>
              )}

              {/* ── CANLI DURUM ── */}
              {st?.configured && stats && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(112px, 1fr))', gap: 6 }}>
                    <Tile k="ÖZSERMAYE" v={usd(stats.equity)} sub={`başlangıç ${usd(stats.capital, 0)}`} />
                    <Tile k="NET KÂR/ZARAR" v={usd(stats.net_pnl)} c={stats.net_pnl >= 0 ? C.neon : C.danger} sub={pct(stats.return_pct)} />
                    <Tile k="ÖDENEN KOMİSYON" v={usd(stats.fees_paid)} c={C.warn}
                          sub={stats.fee_share_of_gross_pct != null ? `brütün %${stats.fee_share_of_gross_pct}'i` : 'videodaki tuzak'} icon={<Coins size={10} color={C.warn} />} />
                    <Tile k="BRÜT" v={usd(stats.gross_pnl)} c={stats.gross_pnl >= 0 ? C.neon : C.danger} />
                    <Tile k="KAZANMA" v={`%${stats.win_rate}`} sub={`${stats.closed_trades} işlem`} />
                    <Tile k="KÂR FAKTÖRÜ" v={String(stats.profit_factor)} c={stats.profit_factor >= 1 ? C.neon : C.danger} />
                    <Tile k="AÇIK / GÜN" v={`${stats.open_positions} / ${stats.day_trades}`} sub={`maks ${st.config?.max_open} / ${st.config?.max_trades_per_day}`} />
                    <Tile k="GÜN GETİRİSİ" v={pct(st.day_return_pct)} c={(st.day_return_pct || 0) >= 0 ? C.neon : C.danger}
                          sub={`limit −%${Math.round((st.config?.daily_loss_limit_pct || 0) * 1000) / 10}`} />
                    <Tile k="DRAWDOWN" v={st.drawdown_pct != null ? `%${st.drawdown_pct}` : '—'} c={C.warn}
                          sub={`maks %${Math.round((st.config?.max_drawdown_pct || 0) * 100)}`} />
                    <Tile k="ORT. TUTMA" v={stats.avg_hold_min != null ? `${stats.avg_hold_min} dk` : '—'} sub="hedef 15-60 dk" />
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1.4fr) minmax(200px, 1fr) minmax(200px, 1fr)', gap: 8, marginTop: 8 }}
                       className="cm-trading-grid3">
                    {/* özsermaye eğrisi — zaman eksenli, sürüklenebilir pencere (kullanıcı koşucusu için ayrı uç YOK: st.equity_curve) */}
                    <div className="cm-card" style={{ margin: 0, padding: 8, minWidth: 0 }}>
                      <EquityChart points={st.equity_curve || []} capital={stats.capital} color={C.cyan} height={150} label="ÖZSERMAYE EĞRİSİ" />
                    </div>
                    {/* tutma kovaları — videonun ölçümü */}
                    <div className="cm-card" style={{ margin: 0, padding: 8 }}>
                      <div className="cm-title" style={{ marginBottom: 4 }}>TUTMA SÜRESİ KOVALARI <span style={{ color: C.muted, fontWeight: 400 }}>(video: 0-15 zarar · 15-60 kâr)</span></div>
                      <table style={{ width: '100%', fontSize: 10, borderCollapse: 'collapse' }}>
                        <tbody>
                          {Object.entries(stats.hold_buckets || {}).map(([k, v]: any) => (
                            <tr key={k} style={{ borderTop: `1px solid ${C.border}` }}>
                              <td style={{ padding: '3px 0', color: k === '15-60 dk' ? C.neon : C.muted }}>{k}</td>
                              <td className="mono" style={{ textAlign: 'right' }}>{v.n}</td>
                              <td className="mono" style={{ textAlign: 'right', color: v.net >= 0 ? C.neon : C.danger }}>{usd(v.net)}</td>
                              <td className="mono" style={{ textAlign: 'right', color: C.muted }}>{v.win_rate != null ? `%${v.win_rate}` : '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div style={{ fontSize: 9, color: C.muted, marginTop: 4 }}>
                        çıkış: {Object.entries(stats.exit_reasons || {}).map(([k, v]: any) => `${k} ${v}`).join(' · ') || '—'}
                      </div>
                    </div>
                    {/* açık pozisyonlar */}
                    <div className="cm-card" style={{ margin: 0, padding: 8 }}>
                      <div className="cm-title" style={{ marginBottom: 4 }}>AÇIK POZİSYONLAR</div>
                      {!st.positions?.length && <div style={{ fontSize: 10, color: C.muted }}>açık pozisyon yok</div>}
                      {st.positions?.map((p: any) => (
                        <div key={p.symbol} style={{ fontSize: 10, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <b style={{ color: p.direction === 'LONG' ? C.neon : C.danger }}>{p.symbol} {p.direction}</b>
                            <span className="mono" style={{ color: p.unrealized >= 0 ? C.neon : C.danger }}>{usd(p.unrealized)} ({pct(p.pnl_pct)})</span>
                          </div>
                          <div className="mono" style={{ fontSize: 9, color: C.muted }}>
                            giriş {p.entry} · stop {Number(p.stop).toPrecision(6)} · hedef {Number(p.target).toPrecision(6)} · tepe {pct(p.peak_pnl_pct)} · {p.age_min} dk ({p.hold_bucket}) · {usd(p.notional)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'minmax(260px, 1.4fr) minmax(240px, 1fr)', gap: 8, marginTop: 8 }}
                       className="cm-trading-grid">
                    {/* karar izi */}
                    <div className="cm-card" style={{ margin: 0, padding: 8 }}>
                      <div className="cm-title" style={{ marginBottom: 4 }}>KARAR ZİNCİRİ — parite başına son karar <span style={{ color: C.muted, fontWeight: 400 }}>(neden işlem yok?)</span></div>
                      {!st.decisions?.length && <div style={{ fontSize: 10, color: C.muted }}>ilk döngü bekleniyor…</div>}
                      {st.decisions?.map((d: any) => (
                        <div key={d.symbol} style={{ fontSize: 9.5, padding: '4px 0', borderTop: `1px solid ${C.border}` }}>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                            <b>{d.symbol}</b>
                            <span className="mono" style={{ color: C.muted }}>z {d.signal?.z ?? '—'} · RSI {d.signal?.rsi ?? '—'}</span>
                            <span style={{ color: String(d.result).startsWith('AÇILDI') ? C.neon : String(d.result).startsWith('VETO') ? C.danger : C.muted }}>{d.result}</span>
                          </div>
                          {d.chain?.steps && (
                            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 3 }}>
                              {d.chain.steps.map((s: any, i: number) => (
                                <span key={i} title={s.note} style={{ fontSize: 8, padding: '1px 5px', borderRadius: 3, cursor: 'help',
                                  color: s.status === 'VETO' ? C.danger : s.status === 'SCALE' ? C.warn : s.status === 'SKIP' ? C.muted : C.neon,
                                  border: `1px solid ${s.status === 'VETO' ? C.danger : s.status === 'SCALE' ? C.warn : s.status === 'SKIP' ? C.border : C.neon}55` }}>
                                  {s.gate} {s.status === 'SCALE' ? `×${s.mult}` : s.status}
                                </span>
                              ))}
                            </div>
                          )}
                          {d.plan && <div className="mono" style={{ fontSize: 8.5, color: C.muted, marginTop: 2 }}>
                            stop %{d.plan.stop_pct} · hedef %{d.plan.target_pct} (1:{d.plan.rr}) · maliyet %{d.cost_pct}</div>}
                        </div>
                      ))}
                    </div>
                    {/* işlemler + olaylar */}
                    <div className="cm-card" style={{ margin: 0, padding: 8, maxHeight: 320, overflowY: 'auto' }}>
                      <div className="cm-title" style={{ marginBottom: 4 }}>SON İŞLEMLER</div>
                      {!st.trades?.length && <div style={{ fontSize: 10, color: C.muted }}>henüz kapanan işlem yok</div>}
                      {st.trades?.slice(0, 12).map((t: any, i: number) => (
                        <div key={i} className="mono" style={{ fontSize: 9, display: 'flex', justifyContent: 'space-between', gap: 6, padding: '2px 0', borderTop: `1px solid ${C.border}` }}>
                          <span>{t.symbol} {t.direction} · {t.reason} · {t.hold_bucket}</span>
                          <span style={{ color: t.net_pnl >= 0 ? C.neon : C.danger }}>{usd(t.net_pnl)} <span style={{ color: C.warn }}>(−{usd(t.fees)})</span></span>
                        </div>
                      ))}
                      <div className="cm-title" style={{ margin: '8px 0 4px' }}>OLAYLAR</div>
                      {st.events?.slice(0, 12).map((e: any, i: number) => (
                        <div key={i} style={{ fontSize: 9, color: e.type === 'error' ? C.danger : e.type === 'entry' ? C.neon : '#8892B0', padding: '1px 0' }}>
                          <span className="mono" style={{ color: C.muted }}>{new Date(e.ts * 1000).toLocaleTimeString('tr-TR')}</span> {e.msg}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          {!cat && <div style={{ fontSize: 10, color: C.muted }}>otopilot katmanı yükleniyor…</div>}
          {cat && !cat.logged_in && (
            <div style={{ fontSize: 9, color: C.muted, marginTop: 4, display: 'flex', gap: 6, alignItems: 'center' }}>
              <ShieldAlert size={10} color={C.warn} /> Anahtar girişi ve otopilot yalnız oturum açınca görünür (yukarıdaki borsa
              kartlarından birini seçip giriş yapın). Halka açık panel salt-okunurdur; hiçbir ziyaretçi başka birinin hesabını göremez.
            </div>
          )}
        </div>
      )}
    </section>
  )
}

/* ── küçük parçalar ── */
function Badge({ text, col, small, title }: { text: string; col: string; small?: boolean; title?: string }) {
  return <span title={title} style={{ fontSize: small ? 7.5 : 8.5, fontWeight: 800, color: col, cursor: title ? 'help' : 'default',
                  border: `1px solid ${col}55`, borderRadius: 4, padding: small ? '0 4px' : '1px 6px', whiteSpace: 'nowrap' }}>{text}</span>
}
function Gate({ ok, text }: { ok: boolean; text: string }) {
  return <div style={{ display: 'flex', gap: 5, alignItems: 'center', color: ok ? C.neon : C.danger }}>
    <span>{ok ? '✓' : '✗'}</span><span>{text}</span></div>
}
function Tile({ k, v, c, sub, icon }: { k: string; v: string; c?: string; sub?: string; icon?: any }) {
  return <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, padding: '6px 8px' }}>
    <div style={{ fontSize: 8, color: C.muted, letterSpacing: 0.6, display: 'flex', gap: 3, alignItems: 'center' }}>{icon}{k}</div>
    <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: c || C.text }}>{v}</div>
    {sub && <div style={{ fontSize: 8, color: C.muted }}>{sub}</div>}
  </div>
}
function Num({ label, v, set, step = 1, dis, unit }: { label: string; v: any; set: (n: number) => void; step?: number; dis?: boolean; unit?: string }) {
  return <label style={{ fontSize: 8.5, color: C.muted, display: 'flex', flexDirection: 'column', gap: 2 }}>
    {label}{unit ? ` (${unit})` : ''}
    <input type="number" value={v ?? ''} step={step} disabled={dis}
      onChange={e => set(Number(e.target.value))} style={{ ...inp, width: '100%', padding: '4px 6px', opacity: dis ? 0.6 : 1 }} />
  </label>
}
function Chk({ label, v, set }: { label: string; v: boolean; set: (b: boolean) => void }) {
  return <label style={{ display: 'flex', gap: 4, alignItems: 'center', cursor: 'pointer', color: v ? C.text : C.muted }}>
    <input type="checkbox" checked={v} onChange={e => set(e.target.checked)} />{label}</label>
}
const inp: React.CSSProperties = {
  background: '#0A0E1A', border: '1px solid #1E2A45', borderRadius: 6,
  color: '#E2E8F0', fontSize: 11, padding: '6px 9px', outline: 'none',
}

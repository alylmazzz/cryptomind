import { useState, useEffect, useCallback } from 'react'
import { KeyRound, Lock, LogOut, Trash2, Plug, ShieldCheck, ShieldAlert } from 'lucide-react'

const API = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')
const C = { neon:'#00FF88', danger:'#FF3B5C', warn:'#FFB627', info:'#0099FF',
            muted:'#6B7394', text:'#E2E8F0', border:'#1E2A45', panel:'#131A2E', surface:'#1A2240' }

const csrf = () =>
  document.cookie.split('; ').find(c => c.startsWith('cm_csrf='))?.split('=')[1] || ''

async function api(path: string, method = 'GET', body?: any) {
  const r = await fetch(`${API}/account${path}`, {
    method,
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(method !== 'GET' ? { 'X-CSRF-Token': csrf() } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  })
  const d = await r.json().catch(() => ({}))
  return { ok: r.ok, status: r.status, data: d }
}

type Field = { key: string; label: string; secret: boolean
               provider_id: string; provider_name?: string; provider?: string
               group?: string; category?: string
               free?: boolean; signup_url?: string; note?: string; ccxt_id?: string }

export default function Account() {
  const [status, setStatus] = useState<any>(null)
  const [fields, setFields] = useState<Field[]>([])
  const [keys, setKeys] = useState<any[]>([])
  const [email, setEmail] = useState(''); const [pw, setPw] = useState('')
  const [msg, setMsg] = useState<{t:string;ok:boolean}|null>(null)
  const [draft, setDraft] = useState<Record<string,string>>({})
  const [busy, setBusy] = useState('')
  const [filter, setFilter] = useState('')

  const refresh = useCallback(async () => {
    const s = await api('/status'); setStatus(s.data)
    const p = await api('/providers'); setFields(p.data?.providers || [])
    if (s.data?.logged_in) { const k = await api('/keys'); setKeys(k.data?.keys || []) }
    else setKeys([])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const doAuth = async (path: string) => {
    setBusy(path); setMsg(null)
    const r = await api(path, 'POST', { email, password: pw })
    setBusy('')
    setMsg({ t: r.ok ? 'Giriş başarılı.' : (r.data?.error || 'hata'), ok: r.ok })
    if (r.ok) { setPw(''); await refresh() }
  }

  // sağlayıcıya göre grupla
  // KARARLI slug ile grupla (görünen ad değişse bile kayıtlar bozulmaz)
  const byProvider = fields.reduce((m: Record<string, Field[]>, f) => {
    (m[f.provider_id] ||= []).push(f); return m
  }, {})
  const saved = new Set(keys.map(k => `${k.provider}|${k.field}`))

  const save = async (prov: string, fs: Field[]) => {
    const payload: Record<string,string> = {}
    fs.forEach(f => { const v = draft[f.key]; if (v && v.trim()) payload[f.key] = v.trim() })
    if (!Object.keys(payload).length) { setMsg({t:'boş alan', ok:false}); return }
    setBusy(prov); setMsg(null)
    const ex = fs[0]?.ccxt_id
    // borsa anahtarında sunucu sadece-okuma doğrulaması yapar
    const body: any = { provider: prov, fields: payload }
    if (ex) {
      body.exchange_id = ex
      body.fields = { apiKey: payload[fs[0].key], secret: payload[fs[1]?.key] || '',
                      ...(fs[2] ? { password: payload[fs[2].key] } : {}) }
    }
    const r = await api('/keys', 'POST', body)
    setBusy('')
    setMsg({ t: r.ok ? `${prov}: kaydedildi` : (r.data?.error || 'hata'), ok: r.ok })
    if (r.ok) { fs.forEach(f => setDraft(d => ({ ...d, [f.key]: '' }))); await refresh() }
  }

  const del = async (prov: string) => {
    setBusy(prov)
    const r = await api('/keys/delete', 'POST', { provider: prov })
    setBusy('')
    setMsg({ t: r.ok ? `${prov}: silindi` : (r.data?.error || 'hata'), ok: r.ok })
    await refresh()
  }

  const test = async (prov: string, ex?: string) => {
    setBusy(prov)
    const r = await api('/keys/test', 'POST', { provider: prov, exchange_id: ex || '' })
    setBusy('')
    const p = r.data?.permissions
    setMsg({ t: r.ok ? `${prov}: ${p?.reason || 'anahtar kayıtlı'}`
                     : (r.data?.error || 'test başarısız'), ok: !!r.ok })
  }

  const vaultOk = status?.vault_ready

  return (
    <div style={{maxWidth:900,margin:'0 auto',padding:'16px 14px 40px'}}>
      <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:12}}>
        <KeyRound size={18} color={C.info}/>
        <h1 style={{fontSize:16,fontWeight:800,letterSpacing:0.5}}>API Anahtarları</h1>
        <a href="#/" style={{marginLeft:'auto',fontSize:11,color:C.info}}>← panele dön</a>
      </div>

      {/* kasa durumu */}
      <div className="cm-card" style={{borderColor: vaultOk ? C.border : C.danger+'66'}}>
        <div style={{display:'flex',alignItems:'center',gap:6,fontSize:11}}>
          {vaultOk ? <ShieldCheck size={14} color={C.neon}/> : <ShieldAlert size={14} color={C.danger}/>}
          <b style={{color: vaultOk ? C.neon : C.danger}}>
            Kasa {vaultOk ? 'hazır' : 'KİLİTLİ'}
          </b>
          <span style={{color:C.muted}}>{status?.vault_note}</span>
        </div>
        <div style={{fontSize:9,color:C.muted,marginTop:6,lineHeight:1.6}}>
          Anahtarlar <b style={{color:C.text}}>AES-256-GCM</b> ile şifrelenir; şifreleme
          anahtarı sunucuda ortam değişkenindedir, veritabanında değildir. Kayıtlı anahtar
          hiçbir uçtan düz metin dönmez — yalnız son 4 hane gösterilir.
          <b style={{color:C.warn}}> Borsa anahtarı yalnız SADECE-OKUMA kabul edilir</b>;
          para çekme veya emir izni olan anahtar reddedilir.
        </div>
      </div>

      {/* giriş */}
      {!status?.logged_in && (
        <div className="cm-card">
          <div style={{display:'flex',alignItems:'center',gap:6,marginBottom:8}}>
            <Lock size={13} color={C.muted}/>
            <span className="cm-title">
              {status?.bootstrap_needed ? 'İLK KULLANICIYI OLUŞTUR' : 'GİRİŞ'}
            </span>
          </div>
          <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
            <input placeholder="e-posta" value={email} onChange={e=>setEmail(e.target.value)}
              style={inp}/>
            <input placeholder="parola (min 12 karakter)" type="password" value={pw}
              onChange={e=>setPw(e.target.value)} style={inp}
              onKeyDown={e => e.key === 'Enter' && doAuth('/login')}/>
            <button className="cm-btn" disabled={!!busy} onClick={()=>doAuth('/login')}
              style={{background:C.neon,color:'#0A0E1A'}}>GİRİŞ</button>
            {status?.bootstrap_needed && (
              <button className="cm-btn" disabled={!!busy} onClick={()=>doAuth('/register')}
                style={{background:C.surface,color:C.info}}>OLUŞTUR</button>
            )}
          </div>
        </div>
      )}

      {status?.logged_in && (
        <div className="cm-card" style={{display:'flex',alignItems:'center',gap:8}}>
          <span style={{fontSize:11}}>Oturum: <b>{status.email}</b></span>
          <span style={{fontSize:10,color:C.muted}}>{keys.length} anahtar kayıtlı</span>
          <button className="cm-btn" style={{marginLeft:'auto',background:C.surface,color:C.muted}}
            onClick={async()=>{ await api('/logout','POST'); await refresh() }}>
            <LogOut size={10} style={{verticalAlign:-1}}/> ÇIKIŞ
          </button>
        </div>
      )}

      {msg && (
        <div className="cm-card" style={{borderColor:(msg.ok?C.neon:C.danger)+'66',
              color: msg.ok?C.neon:C.danger, fontSize:11}}>{msg.t}</div>
      )}

      {/* sağlayıcılar */}
      {status?.logged_in && (
        <>
          <input placeholder="sağlayıcı ara (binance, glassnode, fred…)" value={filter}
            onChange={e=>setFilter(e.target.value.toLowerCase())}
            style={{...inp, width:'100%', marginBottom:8}}/>

          {Object.entries(byProvider)
            .filter(([prov, fs]) => !filter || prov.includes(filter) ||
              (fs[0].provider_name||'').toLowerCase().includes(filter) ||
              (fs[0].group||'').toLowerCase().includes(filter))
            .map(([prov, fs]) => {
              const has = fs.some(f => saved.has(`${prov}|${f.key}`))
              const meta = fs[0]
              return (
                <div key={prov} className="cm-card"
                  style={{borderColor: has ? C.neon+'44' : C.border}}>
                  <div style={{display:'flex',alignItems:'center',gap:6,flexWrap:'wrap'}}>
                    <b style={{fontSize:11}}>{meta.provider_name || prov}</b>
                    <span style={{fontSize:8,padding:'1px 5px',borderRadius:3,
                      background: meta.free ? 'rgba(0,255,136,.12)' : 'rgba(255,182,39,.12)',
                      color: meta.free ? C.neon : C.warn}}>
                      {meta.free ? 'ÜCRETSİZ KATMAN' : 'ÜCRETLİ'}
                    </span>
                    {has && <span style={{fontSize:8,color:C.neon}}>● kayıtlı</span>}
                    {meta.signup_url && (
                      <a href={meta.signup_url} target="_blank" rel="noreferrer"
                        style={{fontSize:9,color:C.info,marginLeft:'auto'}}>anahtar al →</a>
                    )}
                  </div>
                  {meta.note && <div style={{fontSize:9,color:C.muted,marginTop:3}}>{meta.note}</div>}

                  <div style={{display:'flex',gap:5,flexWrap:'wrap',marginTop:7}}>
                    {fs.map(f => {
                      const s = keys.find(k => k.provider === prov && k.field === f.key)
                      return (
                        <input key={f.key} type={f.secret ? 'password' : 'text'}
                          placeholder={s ? `${f.label} — kayıtlı ${s.masked}` : f.label}
                          value={draft[f.key] || ''}
                          onChange={e=>setDraft(d=>({...d,[f.key]:e.target.value}))}
                          style={{...inp, minWidth:200, flex:1}}/>
                      )
                    })}
                  </div>

                  <div style={{display:'flex',gap:5,marginTop:7}}>
                    <button className="cm-btn" disabled={busy===prov}
                      onClick={()=>save(prov, fs)}
                      style={{background:C.neon,color:'#0A0E1A'}}>KAYDET</button>
                    {has && <>
                      <button className="cm-btn" disabled={busy===prov}
                        onClick={()=>test(prov, meta.ccxt_id)}
                        style={{background:C.surface,color:C.info}}>
                        <Plug size={10} style={{verticalAlign:-1}}/> TEST
                      </button>
                      <button className="cm-btn" disabled={busy===prov}
                        onClick={()=>del(prov)}
                        style={{background:'transparent',color:C.danger,
                                border:`1px solid ${C.danger}55`}}>
                        <Trash2 size={10} style={{verticalAlign:-1}}/> SİL
                      </button>
                    </>}
                  </div>
                </div>
              )
            })}
        </>
      )}
    </div>
  )
}

const inp: React.CSSProperties = {
  background:'#0A0E1A', border:'1px solid #1E2A45', borderRadius:6,
  color:'#E2E8F0', fontSize:11, padding:'6px 9px', outline:'none',
}

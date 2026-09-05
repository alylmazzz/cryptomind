import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'
import Account from './Account'

/** Hash yönlendirme: `#/hesap` anahtar yönetimi, diğer her şey panel.
 *  Yol tabanlı yönlendirme yerine hash seçildi — nginx statik `alias` altında
 *  `/cryptomind/hesap` 404 verirdi ve try_files+alias kombinasyonu bilinen bir
 *  nginx tuzağıdır. Hash hiçbir sunucu yapılandırması gerektirmez. */
function Root() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash.startsWith('#/hesap') ? <Account /> : <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode><Root /></StrictMode>
)

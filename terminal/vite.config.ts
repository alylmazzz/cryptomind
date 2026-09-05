import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Üretimde panel mindcorplab.com/cryptomind/ altında yayımlanır; varlık yolları
// ve API taban yolu bu base'den türer (App.tsx: import.meta.env.BASE_URL).
// Yerel geliştirmede `npm run dev` base'i '/' yapar ve /api'yi API sunucusuna proxy'ler.
// `vite preview` de derlenmiş paketi sunar → base üretimdeki gibi /cryptomind/ olmalı;
// aksi hâlde index yüklenir ama /cryptomind/assets/* 404 verir (boş sayfa).
const isPreview = process.argv.some(a => a === 'preview')

export default defineConfig(({ command }) => ({
  base: (command === 'build' || isPreview) ? '/cryptomind/' : '/',
  plugins: [react(), tailwindcss()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8210', changeOrigin: true },
      '/account': { target: 'http://127.0.0.1:8210', changeOrigin: true },
      // `npm run preview` üretim base'iyle (/cryptomind/) çalışır — nginx'in yaptığı
      // ön ek kırpmayı burada taklit ederiz ki yerelde birebir prod akışı denensin.
      '/cryptomind/api': { target: 'http://127.0.0.1:8210', changeOrigin: true,
                           rewrite: (p: string) => p.replace(/^\/cryptomind/, '') },
      '/cryptomind/account': { target: 'http://127.0.0.1:8210', changeOrigin: true,
                               rewrite: (p: string) => p.replace(/^\/cryptomind/, '') },
    },
  },
}))

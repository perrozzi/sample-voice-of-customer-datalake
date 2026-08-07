import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    global: 'globalThis',
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes('@tanstack/react-query')) return 'vendor-query'
          if (id.includes('react-markdown') || id.includes('remark-gfm')) return 'vendor-markdown'
          if (id.includes('lucide-react')) return 'vendor-icons'
          if (id.includes('recharts')) return 'vendor-recharts'
          if (id.includes('react-router-dom') || id.includes('react-dom') || id.includes('/react/')) return 'vendor-react'
        },
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})

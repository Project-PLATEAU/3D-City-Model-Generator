import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.obj', '**/*.svg'], // Include .obj and .svg files as assets
  server: {
    // Expose frontend to network (accessible by other users)
    host: '0.0.0.0',
    port: 8080,
    strictPort: true,
    allowedHosts: true, // Allow all hosts (Cloudflare tunnel, etc.)
    // Configure HMR for network access
    // If still having issues, you can disable HMR by setting to false
    hmr: {
      host: '163.220.176.205', // Replace with your actual IP address
      port: 8080,
      protocol: 'ws',
      // Uncomment below to disable HMR if WebSocket connections are problematic
      // overlay: false,
    },
    proxy: {
      // Proxy API requests to backend (backend runs on localhost only)
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      // Proxy output files to backend
      '/outputs': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      // Proxy all backend endpoints
      '/upload': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/jobs': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/generate': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/upload-lod3': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      },
      '/cleanup': {
        target: 'http://localhost:5001',
        changeOrigin: true,
      }
    }
  }
})

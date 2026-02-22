import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const buildSourceMaps = process.env.VITE_BUILD_SOURCEMAP !== 'false'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    sourcemap: buildSourceMaps,
  },
})

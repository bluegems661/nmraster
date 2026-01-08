/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // NMR-specific colors
        spectrum: {
          trace: '#00ff88',
          peak: '#ffaa00',
          selected: '#ff4444',
          background: '#1a1a2e',
        },
      },
    },
  },
  plugins: [],
}

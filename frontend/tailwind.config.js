/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        vera: {
          dark: '#0B0F19', // Deep dark blue/black background
          darker: '#06080D',
          panel: '#151C2C', // Lighter panels
          border: '#232E48',
          accent: '#3B82F6', // Blue accent
          success: '#10B981',
          warning: '#F59E0B',
          danger: '#EF4444',
          text: '#F3F4F6',
          textMuted: '#9CA3AF'
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      }
    },
  },
  plugins: [],
}

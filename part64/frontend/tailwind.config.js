/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          0: '#272822',
          1: '#1f201d',
        },
        ink: '#f8f8f2',
        muted: '#a59f85',
        card: 'rgba(45, 46, 39, 0.86)',
        line: 'rgba(248, 248, 242, 0.16)',
      },
      fontFamily: {
        serif: ['ui-serif', 'Georgia', 'Cambria', '"Times New Roman"', 'Times', 'serif'],
      }
    },
  },
  plugins: [],
}

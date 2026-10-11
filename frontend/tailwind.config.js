/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef6ff',
          100: '#d9eaff',
          200: '#bcdbff',
          300: '#8fc3ff',
          400: '#5ba2ff',
          500: '#347eff',
          600: '#1b5bf5',
          700: '#1446e1',
          800: '#1739b6',
          900: '#18348f',
          950: '#142157',
        },
      },
    },
  },
  plugins: [],
}

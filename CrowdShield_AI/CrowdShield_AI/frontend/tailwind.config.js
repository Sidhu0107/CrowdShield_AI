/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        crowdRed: "#E24B4A",
        crowdAmber: "#BA7517",
        crowdGreen: "#3B6D11",
        background: "#111111",
      },
    },
  },
  plugins: [],
}

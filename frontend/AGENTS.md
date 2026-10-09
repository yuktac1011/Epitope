# fedshield-app

React + Next.js App Router + Tailwind CSS project.

## Development Server

Start the development server with:

```bash
npm run dev
```

- Preview URL: `http://localhost:3000`
- Hot reload: Changes to source files are reflected immediately

## Project Structure

- `src/app/layout.tsx` - Root layout with global styling and metadata
- `src/app/page.tsx` - App entry page rendering the `App` component
- `src/app/globals.css` - Global CSS entrypoint and Tailwind CSS v4 styling
- `src/App.tsx` - Primary application component containing the research console screens
- `next.config.ts` - Next.js configuration
- `postcss.config.mjs` - PostCSS configuration for Tailwind CSS v4
- `package.json` - Project dependencies and scripts (`dev`, `build`, `start`, `format`)
- `tsconfig.json` - TypeScript configuration with Next.js path aliases (`@/*`)

## Dependencies

- Runtime: Next.js 15, React 19, and React DOM 19
- Styling: Tailwind CSS v4 with `@tailwindcss/postcss` and PostCSS
- Build tooling: TypeScript 5.7

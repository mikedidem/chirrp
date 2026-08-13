# HydroProject UI

An Angular frontend for groundwater scenario management, MODFLOW execution, scenario comparison, natural-language autofill, and the RAG assistant.

## Features

- Run MODFLOW scenarios from the browser
- Autofill WEL/RCH percentage fields from natural-language instructions
- Compare stored hydraulic-head scenarios
- Launch Bayesian optimization for hydraulic-head tuning
- Ask groundwater policy questions in the RAG Assistant
- Compare RAG answers with general Gemini answers

## Prerequisites

- Node.js 20 or newer
- npm 10.8.2 or newer
- Backend API running at the configured `environment.apiUrl`

## Installation

```bash
cd hydroproject-ui
npm install
```

## Development

```bash
npm start
```

or

```bash
ng serve
```

The app is served at `http://localhost:4200`.

## Available pages

- Home
- Run MODFLOW
- Compare Hydraulic Head
- Optimize Hydraulic Head
- RAG Assistant
- LLM Config Interaction

## Scripts

- `npm start` — start the dev server
- `npm run build` — build the Angular app
- `npm test` — run the unit test suite
- `npm run watch` — watch mode build
- `npm run serve:ssr:ui-app` — serve the SSR output

## Project structure

```text
src/
├── app/
│   ├── components/
│   ├── pages/
│   │   ├── compare-hydraulic-head/
│   │   ├── home/
│   │   ├── llm-config-interaction/
│   │   ├── optimize-hydraulic-head/
│   │   ├── rag-assistant/
│   │   └── run-modflow/
│   ├── services/
│   └── environments/
└── assets/
```

## Notes

- The UI talks to the FastAPI backend on port 8000.
- The RAG Assistant route is `/rag-assistant`.
- If API calls fail, confirm the backend and database are running first.

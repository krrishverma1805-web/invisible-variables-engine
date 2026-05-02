# Phase D — React/Next.js Frontend Scoping

**Effective date:** 2026-04-29
**Owner:** Frontend lead (TBD) + ML lead
**Plan reference:** §94 + §144

This document is the forcing function for a Phase D decision: ship a real
SaaS-grade frontend, accept Streamlit's structural ceiling permanently,
or build a hybrid. It lists every choice with a recommended default and
the trade-off so the team can sign off in one meeting.

## Decision required

**Build now / build later / Streamlit forever.**

| Option | Cost | Risk | Recommendation |
|---|---|---|---|
| Build now | 6–8 wk solo, 4–5 wk pair | Drag on Phase B/C closeout | Defer until Phase B ships |
| Build later (Phase D, ~2026-Q4) | Same | Lowest — Streamlit gaps documented; gaps inform requirements | **Recommended** |
| Streamlit + custom components forever | 2–3 wk per major polish push | Permanent ceiling on power-user features | Acceptable if SaaS polish is not a goal |

## Streamlit gaps that drove this scoping

(documented during Phase A walkthrough — see `docs/competitive_baseline.md`
Axis 2):

1. **No native keyboard hooks** — command palette / `⌘K` requires
   `streamlit-shortcuts` (third-party) or custom React component.
2. **No real side-sheet primitive** — `st.dialog` is modal, blocks
   side-by-side comparison.
3. **`st.dataframe` ceiling** — no per-column filter, no virtualization,
   no bulk actions; AgGrid pin is a workaround with license posture
   (RC §22).
4. **Polling-only for progress** — Streamlit can't subscribe natively to
   the FastAPI WebSocket (`src/ive/api/websocket/progress.py`).
5. **Streamlit owns the chrome** — top bar / sidebar gradient overrides
   are CSS-fragile.
6. **No first-class background fetch / no client-side cache** — every
   interaction triggers a full rerun.

## Stack recommendations (forcing-function defaults)

| Decision | Recommendation | Rationale |
|---|---|---|
| **Framework** | Next.js 14 App Router | Server components for static pages, built-in routing, mature SSR/streaming |
| **Component library** | shadcn/ui + Tailwind | Carbon-parity achievable, no licensing hazard (MIT), copy-paste components mean no version-lock pain |
| **Auth** | Auth0 | Reuses our scope model; oauth2-proxy in front of Streamlit handed off cleanly |
| **Data fetching** | TanStack Query + server components | Cache + retry semantics out of the box; SSR for static landing |
| **Charts** | Plotly.js (existing palettes survive port) + recharts for simpler bars | Keep `_apply_carbon_layout` logic; recharts for non-statistical chart types |
| **Tables** | TanStack Table | Headless, virtualized, zero license risk |
| **Deployment** | Vercel for FE, current Docker stack for BE | Vercel handles SSR + edge caching; BE unchanged |
| **State** | Zustand for client state, TanStack Query for server state | Lightweight, no Redux ceremony |
| **Form** | React Hook Form + Zod | Schema-first validation matches our Pydantic discipline |
| **Testing** | Playwright (E2E), Vitest (unit), Storybook (component) | Established, well-tooled |
| **Type checking** | TypeScript strict mode | No exceptions |
| **Linting** | ESLint + Prettier + import-sort | Standard |

## Migration scope

Phase D **does not** rewrite the FastAPI surface — it consumes it. The
contract in `docs/RESPONSE_CONTRACT.md` already lists every field a
Phase D client needs.

### In-scope for Phase D

- All pages currently in `streamlit_app/pages/`:
  - 01_upload (drag-drop, multi-file, sensitivity tagging UI)
  - 02_configure (presets, live profile preview)
  - 03_monitor (true WebSocket subscription, Gantt timeline)
  - 04_results (filterable table + side-sheet, DML + share-token UI)
  - 05_compare (N-way Venn, side-by-side YAML diff)
- Sidebar with command palette (`⌘K`), breadcrumb, live health.
- Auth (login, key management, audit log view).

### Out-of-scope for Phase D

- Server-side feature work (everything stays in FastAPI).
- Pipeline logic (no duplication; FE is a thin client).
- Streamlit removal (kept available for power-user dev workflows; QA
  page `_99_visual_qa.py` stays as a regression target).

## Effort estimate

| Sub-area | Solo | Pair |
|---|---|---|
| Project scaffolding (Next.js + Auth0 + Vercel) | 0.5 wk | 0.3 wk |
| Component library + theme port | 1 wk | 0.5 wk |
| Pages 01-02 (upload + configure) | 1 wk | 0.5 wk |
| Page 03 (monitor — WebSocket subscription) | 1 wk | 0.5 wk |
| Page 04 (results — most complex; filterable + side-sheet + DML + share) | 1.5 wk | 1 wk |
| Page 05 (compare) | 0.5 wk | 0.3 wk |
| Sidebar + command palette + auth flows | 1 wk | 0.5 wk |
| E2E tests, polish, deployment | 1 wk | 0.5 wk |
| **Total** | **7.5 wk** | **4.1 wk** |

## Acceptance criteria for Phase D close

- [ ] Every page in `streamlit_app/pages/` has a Next.js equivalent at
      feature parity.
- [ ] Playwright E2E suite covers all of `tests/e2e/test_ui_smoke.py`.
- [ ] Lighthouse Performance ≥85 on the results page (largest, most JS).
- [ ] Bundle size ≤500 kB gzipped on the home route.
- [ ] WCAG AA contrast verified by `axe-core` against every page.
- [ ] No regression in any RC §-labeled gate.
- [ ] Streamlit remains runnable; Phase D does not delete it (one-quarter
      observation window before sunset).

## Open questions

- **Multi-tenancy** (RC §20): Phase D is single-tenant per deployment.
  Multi-tenancy is its own follow-up — track separately.
- **Self-hosted LLM endpoint UX** (`docs/self_hosted_llm.md`): does
  Phase D surface the choice, or is it operator-only? Recommendation:
  operator-only env-flag in v1; surface in admin console in v1.1.
- **Mobile**: explicitly out-of-scope. Tabular data + side-sheet UX is
  desktop-first by design.

## Recommendation

**Defer build to 2026-Q4** (after Phase B ships). The scoping doc is
what matters today: it lets Phase B/C engineers stop reasoning about
what Streamlit can/can't deliver. When Phase D starts, this doc is the
brief.

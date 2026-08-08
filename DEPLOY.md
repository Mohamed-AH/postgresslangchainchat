# Deploying ragchat (free tier)

The whole demo runs for free on **Render** (compute: web UI + API + a cleanup cron) with
**Neon** as the PostgreSQL + pgvector database. Phase 1 runs on *your* Cohere and Gemini
keys, protected by upload caps, per-session rate limits, a global daily budget, and
auto-expiring session data.

```
Browser ─► Render web service (FastAPI + built-in UI)  ─►  Neon (Postgres + pgvector)
                                                              ▲
          GitHub Actions (hourly) ─ ragchat cleanup ─────────┘  purges expired sessions
```

Render's free tier has no cron jobs, so scheduled cleanup runs as a **GitHub Actions**
workflow that connects straight to Neon (step 3).

## 1. Create the database (Neon)

1. Create a project at <https://neon.com>. pgvector is available on the free plan.
2. Enable the extension once (Neon SQL editor): `CREATE EXTENSION IF NOT EXISTS vector;`
   (The app also attempts this on first use, but doing it here avoids a first-request
   permission surprise.)
3. Copy the **pooled** connection string. It looks like:
   `postgresql://USER:PASSWORD@ep-xxx-pooler.REGION.aws.neon.tech/DB?sslmode=require`
   — the app normalizes the driver and passes `sslmode` straight through to psycopg3.

Notes: the free plan gives ~0.5 GB storage and scales to zero after 5 minutes idle (the
first request after idle takes ~1 s). Session data auto-expires, so storage stays bounded.

## 2. Deploy the app (Render)

1. Push this repo to GitHub (already done for this branch).
2. In Render, **New → Blueprint** and point it at the repo. It reads `render.yaml` and
   creates the `ragchat` web service.
3. Set these **secret** environment variables on the service (marked `sync:false`, so
   Render prompts for them — they are never committed):
   - `DATABASE_URL` — the Neon pooled string from step 1
   - `COHERE_API_KEY`
   - `GOOGLE_API_KEY`
4. Deploy. When it's live, open the service URL: the web UI is at `/`, interactive API
   docs at `/docs`, readiness at `/health`.

The free web service spins down after ~15 minutes idle and takes ~1 minute to wake; the
UI shows a "waking up" banner during that window.

## 3. Schedule cleanup (GitHub Actions)

Render's free tier has no cron, so expired sessions are reclaimed by the
`.github/workflows/cleanup.yml` workflow (hourly + a manual "Run workflow" button). It
connects directly to Neon and needs **only the database URL** — dropping expired
collections computes no embeddings, so your provider keys aren't required.

1. In the GitHub repo: **Settings → Secrets and variables → Actions → New repository
   secret** → add `DATABASE_URL` with the same Neon pooled string.
2. That's it. The workflow runs hourly; you can also trigger it from the **Actions** tab
   (**Cleanup expired sessions → Run workflow**) to verify it.

> Prefer daily instead of hourly? Change the `cron:` line in the workflow to `0 3 * * *`.
> On a **private** repo, Actions minutes are quota-limited — daily keeps usage minimal;
> public repos get unlimited minutes.

## 4. Tuning (no redeploy needed)

All limits are environment variables you can change in the Render dashboard:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Gemini model. Flash-Lite has the most generous free quota (~15 RPM / 1,000 per day). A `404 NOT_FOUND` means the name isn't available to your key; a `429 RESOURCE_EXHAUSTED` with `limit: 0` means that model has no free-tier allocation for your key — switch models or enable billing. |
| `SESSION_TTL_HOURS` | `24` | How long uploaded data is retained |
| `RATE_LIMIT_ASKS_PER_MINUTE` | `10` | Per-session ask limit, kept under the free-tier RPM |
| `MAX_UPLOAD_BYTES` | `2097152` | Max upload size (2 MiB) |
| `MAX_SECTIONS_PER_UPLOAD` | `150` | Max chunks per upload |
| `RATE_LIMIT_INGESTS_PER_HOUR` | `20` | Per-session upload limit |
| `DAILY_REQUEST_BUDGET` | `1000` | Instance-wide daily op cap protecting your keys (0 = off) |

## Caveats (by design, for a free demo)

- **Cold starts** on both Render and Neon after idle — expected; surfaced in the UI.
- **Rate limits/budget are per-instance** (in-memory). The free tier is a single
  instance, so this is fine; a multi-instance deployment would back them with Redis.
- **Schema changes** need a fresh database or a migration — `create_all` does not alter
  existing tables. Alembic is the intended follow-up.

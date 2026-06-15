# Love Island Episode Generator

A "fun chaos generator" — friends submit secret crushes/drama, you hit
**Generate Episode**, and an AI writers' room turns it into a structured
Love Island script.

## How it works (the pipeline)

Five focused LLM calls (gpt-4o-mini), each doing ONE job, plus deterministic
Python glue that actually applies the changes — the LLM proposes, plain code
decides ground truth:

```
inputs
  → 1. State Builder        → "what's true right now" + this round's "wants"
  → 2a. Outcome Decider      → for each want: granted / denied / complicated
  → 2b. System Event Maker   → 1-2 declarative twist events
  → [deterministic updater]  → applies outcomes + events to character memory,
                                 appends to want_history — no LLM involved
  → 3. Scene Layer           → storylines + scene beats + per-scene
                                 emotional deltas
  → [deterministic snapshots] → applies scene deltas in order, producing
                                 escalating emotional snapshots — no LLM
  → 4. Script Writer          → ONLY expands the given beats, using the
                                 snapshots to show escalation
```

**Validation:** every JSON call checks for required fields. If a response is
invalid or incomplete, it gets one retry with a stricter follow-up message; if
that also fails, the pipeline falls back to safe empty defaults instead of
crashing.

**1. State Builder** updates persistent character memory: current partner,
this round's "wants" (desires/claims — separate from reality), interest
scores, and emotional momentum (`trust`, `jealousy`, `confusion`,
`confidence`, 0.0-1.0, shifting gradually rather than resetting).

**2a. Outcome Decider** is the core mechanic: *wants are inputs, not
guarantees.* For each want it decides `granted | denied | complicated` and
why — optionally nudged by a lightweight deterministic "public opinion" score
computed from each character's current emotional state (popular = more likely
granted, unpopular = more likely denied/complicated — a soft signal, not a
rule).

**2b. System Event Maker** proposes 1-2 declarative twist events (forced
recoupling, new arrival, secret vote, bombshell reveal, etc.) with mechanical
`state_effects` — described, not yet applied.

**[Deterministic updater]** — plain Python, no LLM — applies the outcome
decisions and system event effects to character memory: updates
`current_partner`, nudges `emotional_state` by fixed amounts depending on
outcome (denied → jealousy up/trust down, granted → trust/confidence up,
complicated → confusion up), and appends each want's result to
`want_history` (`{episode, target, status}`, last 10 kept) — so resentment,
repeated rejection, and villain arcs build across episodes.

**3. Scene Layer** turns everything into an episode blueprint with a fixed
scene order (morning → confessionals → gossip → challenge → twist →
recoupling → fallout → night), 2-4 escalating beats per scene, 2-3
storylines, and per-scene `emotional_deltas`.

**[Deterministic snapshots]** — plain Python — applies each scene's
emotional deltas in order, producing an escalating snapshot per scene
("pressure building" scene-by-scene). The final snapshot becomes the new
persisted baseline for next episode.

**4. Script Writer** is constrained to ONLY expand the given beats — no new
events, characters, or outcomes — using the scene snapshots so a character
visibly gets more on-edge as jealousy climbs through the episode, and showing
honest reactions when a want gets denied.

Submissions are cleared after each generation (new round of secrets), but
character memory — including `want_history` and emotional state — persists,
so episode 2 remembers what happened in episode 1.

## Run locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY="sk-..."
   ```
   (Windows: `set OPENAI_API_KEY=sk-...`)

3. Run the server (it serves both the API and the frontend):
   ```
   uvicorn main:app --reload
   ```

4. Open **http://localhost:8000** — that's it, frontend + backend on one URL.

## Deploy on Render

1. Push this folder to a GitHub repo (must include `main.py`, `index.html`,
   `requirements.txt`).
2. Render → **New Web Service** → connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command:
   ```
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```
5. In Render's environment variables, add `OPENAI_API_KEY` with your key.
6. Render gives you a URL like `https://your-app.onrender.com` — send that
   to friends. No installs, no setup.

## Endpoints

- `GET /` — serves the frontend
- `POST /submit` — `{ "user": "optional name", "input": "who you like / what you did" }`
- `GET /generate` — runs the full pipeline, clears submissions, returns
  `{ "episode": "..." }`
- `GET /generate?debug=true` — also returns `episode_number`,
  `public_opinion`, `outcomes`, `events`, `scenes`, `scene_snapshots`, and
  `characters` (with `want_history`) so you can see every stage of the
  writers' room (there's a checkbox for this in the UI)
- `GET /submissions` — peek at pending submissions
- `GET /characters` — peek at persistent character memory
- `POST /reset` — clears submissions AND character memory (start a brand
  new season)

## Troubleshooting

- **"It works locally but not online"** → make sure the start command
  includes `--host 0.0.0.0`.
- **Frontend doesn't load** → `index.html` must be in the same directory as
  `main.py`.
- **OpenAI errors on Render** → check that `OPENAI_API_KEY` is set in
  Render's environment variables.
- **Drama state looks empty/garbled** → each LLM call validates required
  fields and retries once before falling back to safe empty defaults (e.g. no
  outcomes, no events). Check `/generate?debug=true` to see which stage came
  back empty — that's the one that failed validation twice.
- **Generation feels slow** → this is now 5 sequential LLM calls per episode
  (gpt-4o-mini, so still cheap, but it adds up). If you want it faster, the
  Outcome Decider and System Event Maker (stages 2a/2b) could be merged back
  into one call at the cost of the LLM having more "authority" at once.

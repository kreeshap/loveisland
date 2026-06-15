import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Gemini exposes an OpenAI-compatible endpoint, so we can keep using the
# `openai` Python library — just point it at Google's base_url and use
# the GEMINI_API_KEY environment variable (set this in Render's dashboard).
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = "gemini-2.5-flash"

# ----------------------------------------------------------------------
# In-memory "database" — resets when the server restarts
# ----------------------------------------------------------------------
submissions: list[dict] = []
characters: dict = {}
episode_number: int = 0

DEFAULT_EMOTIONAL_STATE = {"trust": 0.5, "jealousy": 0.3, "confusion": 0.3, "confidence": 0.5}
SCENE_ORDER = ["morning", "confessionals", "gossip", "challenge", "twist", "recoupling", "fallout", "night"]


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------
# Serve the frontend
# ----------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse("index.html")


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class Submission(BaseModel):
    user: str | None = None
    input: str


# ----------------------------------------------------------------------
# Submission endpoints
# ----------------------------------------------------------------------
@app.post("/submit")
def submit(data: Submission):
    submissions.append(data.model_dump())
    return {"status": "ok", "total": len(submissions)}


@app.get("/submissions")
def get_submissions():
    return {"submissions": submissions, "total": len(submissions)}


@app.get("/characters")
def get_characters():
    return {"characters": characters, "episode_number": episode_number}


@app.post("/reset")
def reset():
    """Clear submissions AND character memory — start a brand new season."""
    global episode_number
    submissions.clear()
    characters.clear()
    episode_number = 0
    return {"status": "cleared"}


# ----------------------------------------------------------------------
# LLM JSON helper — validates required keys, retries once with a stricter
# follow-up if invalid, and falls back to a safe default rather than crashing.
# ----------------------------------------------------------------------
def call_llm_json(system_prompt: str, user_prompt: str, required_keys: list[str],
                   fallback: dict, temperature: float = 0.7) -> tuple[dict, bool]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(2):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw = response.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None

        if data is not None and all(k in data for k in required_keys):
            return data, True

        messages.append({"role": "assistant", "content": raw or ""})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous response was invalid JSON or missing required "
                f"fields {required_keys}. Respond again with ONLY valid JSON "
                f"containing all required fields."
            ),
        })

    return fallback, False


# ----------------------------------------------------------------------
# THE PIPELINE (5 calls, each focused on ONE job, plus deterministic glue)
#
#   1. State Builder        -> "what's true right now" (relationships,
#                                wants, emotional momentum)
#   2a. Outcome Decider      -> for each want, granted / denied / complicated
#   2b. System Event Maker   -> 1-2 declarative twist events
#   [deterministic updater]  -> applies outcomes + events to character memory,
#                                appends to want_history (no LLM involved)
#   3. Scene Layer           -> storylines + scene-by-scene beats +
#                                per-scene emotional deltas
#   [deterministic snapshots] -> applies scene deltas in order to compute
#                                escalating emotional snapshots per scene
#   4. Script Writer          -> ONLY expands the given beats, using the
#                                snapshots to show escalation
# ----------------------------------------------------------------------

# ---- Stage 1: State Builder -------------------------------------------------

STATE_BUILDER_SYSTEM = """You are the continuity tracker for a reality dating show.
Your ONLY job is to figure out what is currently true. You do NOT invent drama,
twists, or conflict — that's a different team's job.

Rules:
- Do NOT invent contestants who weren't mentioned in inputs or existing memory.
- Use existing character memory as ground truth; update it incrementally, don't
  discard history.
- "emotional_state" values are 0.0-1.0 and represent MOMENTUM — shift gradually
  from previous values based on new inputs, never reset to a default.
- Submitted desires (who someone says they like, what they say they did) are
  just "wants" for this round — separate from "current_partner" (which only
  changes via recoupling events from a previous episode).
- Each want should be a short string naming a target person, e.g. "wants to get
  closer to Ava" or "wants Ben to notice them".

Respond with ONLY valid JSON, no markdown, matching this shape:

{
  "characters": {
    "Name": {
      "current_partner": "Name or null",
      "wants": ["short want strings naming a target person"],
      "interest_scores": {"OtherName": 0-10},
      "emotional_state": {"trust": 0.0-1.0, "jealousy": 0.0-1.0, "confusion": 0.0-1.0, "confidence": 0.0-1.0},
      "history": ["short bullet of past events, keep last ~5"]
    }
  },
  "public_statements": ["things said openly"],
  "secret_feelings": ["things only submitted privately"]
}
"""


def run_state_builder() -> dict:
    prompt = f"""
Existing character memory (may be empty for a brand new season):
{json.dumps(characters, indent=2)}

New anonymous submissions for this round:
{json.dumps(submissions, indent=2)}
"""
    fallback = {"characters": characters, "public_statements": [], "secret_feelings": []}
    data, _ = call_llm_json(STATE_BUILDER_SYSTEM, prompt,
                             required_keys=["characters", "public_statements", "secret_feelings"],
                             fallback=fallback, temperature=0.5)
    return data


def merge_state(new_chars: dict) -> None:
    """Merge State Builder output into persistent character memory."""
    for name, data in new_chars.items():
        existing = characters.setdefault(name, {})
        existing["current_partner"] = data.get("current_partner", existing.get("current_partner"))
        existing["wants"] = data.get("wants", [])
        existing["interest_scores"] = data.get("interest_scores", existing.get("interest_scores", {}))

        emo = existing.setdefault("emotional_state", dict(DEFAULT_EMOTIONAL_STATE))
        for k, v in data.get("emotional_state", {}).items():
            try:
                emo[k] = clamp(float(v))
            except (TypeError, ValueError):
                pass

        existing["history"] = (data.get("history", existing.get("history", [])) or [])[-5:]
        existing.setdefault("want_history", existing.get("want_history", []))


def compute_public_opinion() -> dict:
    """Lightweight 'audience sentiment' — deterministic, no LLM.
    Positive = audience likes them right now, negative = audience is souring."""
    opinions = {}
    for name, data in characters.items():
        emo = data.get("emotional_state", DEFAULT_EMOTIONAL_STATE)
        score = (emo.get("trust", 0.5) + emo.get("confidence", 0.5)
                 - emo.get("jealousy", 0.3) - emo.get("confusion", 0.3)) / 2
        opinions[name] = round(score, 2)
    return opinions


# ---- Stage 2a: Outcome Decider ----------------------------------------------

OUTCOME_DECIDER_SYSTEM = """You are the outcomes desk for a reality dating show.
You take each character's "wants" and decide what actually happens — NOT what
they asked for. CORE PRINCIPLE: wants are inputs, not guarantees. The show
doesn't owe anyone their crush. You may use "public_opinion" (audience
sentiment, roughly -1 to 1) as a soft signal — unpopular contestants are more
likely to get denied or complicated outcomes, popular ones more likely granted,
but it's not a hard rule.

You do NOT generate twists or system events — that's a separate step. Just
decide outcomes for the wants you're given.

Respond with ONLY valid JSON, no markdown, matching this shape:

{
  "wants_vs_outcomes": [
    {"person": "Name", "target": "Name", "wanted": "original want string", "outcome": "granted | denied | complicated", "why": "short reason"}
  ]
}
"""


def run_outcome_decider(public_opinion: dict) -> dict:
    prompt = f"""
Characters and their current wants:
{json.dumps({n: {"wants": d.get("wants", []), "current_partner": d.get("current_partner"), "interest_scores": d.get("interest_scores", {})} for n, d in characters.items()}, indent=2)}

Public opinion (audience sentiment, -1 to 1):
{json.dumps(public_opinion, indent=2)}
"""
    fallback = {"wants_vs_outcomes": []}
    data, _ = call_llm_json(OUTCOME_DECIDER_SYSTEM, prompt,
                             required_keys=["wants_vs_outcomes"],
                             fallback=fallback, temperature=0.8)
    return data


# ---- Stage 2b: System Event Maker -------------------------------------------

SYSTEM_EVENT_SYSTEM = """You are the twist desk for a reality dating show. Given
the current state and the outcome decisions already made, propose 1-2
"system events" — production-driven announcements (forced recoupling, new
arrival, secret vote, surprise dumping, bombshell reveal, public challenge
twist) that shake up the villa.

These are DECLARATIVE state changes only — you describe the announcement and
its mechanical effects. You do NOT write dialogue or scenes.

Respond with ONLY valid JSON, no markdown, matching this shape:

{
  "system_events": [
    {
      "type": "forced_recoupling | new_arrival | secret_vote | surprise_dumping | bombshell_reveal | public_challenge_twist",
      "description": "what happens, framed as an in-villa announcement",
      "forces_outcome_for": ["Names affected"],
      "state_effects": {
        "Name": {
          "current_partner": "Name or null (only set if this event changes it)",
          "emotional_state_delta": {"trust": -0.2, "jealousy": 0.3}
        }
      }
    }
  ]
}
"""


def run_system_event_maker(outcomes: list[dict], public_opinion: dict) -> dict:
    prompt = f"""
Current characters:
{json.dumps(characters, indent=2)}

Outcome decisions already made this round:
{json.dumps(outcomes, indent=2)}

Public opinion (audience sentiment, -1 to 1):
{json.dumps(public_opinion, indent=2)}
"""
    fallback = {"system_events": []}
    data, _ = call_llm_json(SYSTEM_EVENT_SYSTEM, prompt,
                             required_keys=["system_events"],
                             fallback=fallback, temperature=0.8)
    return data


# ---- Deterministic state updater (no LLM) -----------------------------------

def apply_outcomes_and_events(outcomes: list[dict], events: list[dict]) -> None:
    """Apply outcome decisions + system events to persistent character memory.
    This is plain Python — the LLM only proposes, this function decides ground truth."""

    for o in outcomes:
        person = o.get("person")
        if person not in characters:
            continue

        wh = characters[person].setdefault("want_history", [])
        wh.append({"episode": episode_number, "target": o.get("target"), "status": o.get("outcome")})
        characters[person]["want_history"] = wh[-10:]

        emo = characters[person].setdefault("emotional_state", dict(DEFAULT_EMOTIONAL_STATE))
        status = o.get("outcome")
        if status == "denied":
            emo["jealousy"] = clamp(emo.get("jealousy", 0.3) + 0.15)
            emo["trust"] = clamp(emo.get("trust", 0.5) - 0.1)
            emo["confidence"] = clamp(emo.get("confidence", 0.5) - 0.05)
        elif status == "granted":
            emo["trust"] = clamp(emo.get("trust", 0.5) + 0.1)
            emo["confidence"] = clamp(emo.get("confidence", 0.5) + 0.1)
        elif status == "complicated":
            emo["confusion"] = clamp(emo.get("confusion", 0.3) + 0.15)

    for ev in events:
        for name, delta in ev.get("state_effects", {}).items():
            if name not in characters:
                continue
            if delta.get("current_partner") is not None:
                characters[name]["current_partner"] = delta["current_partner"]

            emo = characters[name].setdefault("emotional_state", dict(DEFAULT_EMOTIONAL_STATE))
            for k, v in delta.get("emotional_state_delta", {}).items():
                try:
                    emo[k] = clamp(emo.get(k, 0.5) + float(v))
                except (TypeError, ValueError):
                    pass


# ---- Stage 3: Scene Layer -----------------------------------------------------

SCENE_LAYER_SYSTEM = """You are the episode producer for a reality dating show.
You take the current state, the outcome decisions, and the system events, and
turn them into a concrete episode blueprint: storylines + scene-by-scene beats
+ per-scene emotional deltas. You do NOT write dialogue.

The episode must use this fixed scene order:
morning, confessionals, gossip, challenge, twist, recoupling, fallout, night

For each scene, list 2-4 short beats that ESCALATE — early scenes plant
tension, later scenes pay it off. Place both system_events in the most natural
scene ("twist" for big announcements, but a "bombshell_reveal" might land
during "gossip" or "fallout" — use judgment).

ALSO for each scene, provide "emotional_deltas": small per-character nudges
(e.g. -0.1 to +0.2) to trust/jealousy/confusion/confidence that reflect what
just happened in that scene. These accumulate scene-by-scene to create
escalating pressure — don't put all the change in one scene.

Respond with ONLY valid JSON, no markdown, matching this shape:

{
  "storylines": ["2-3 main arcs for this episode, one sentence each"],
  "scenes": {
    "morning": {"beats": ["...", "..."], "emotional_deltas": {"Name": {"trust": -0.05}}},
    "confessionals": {"beats": ["..."], "emotional_deltas": {}},
    "gossip": {"beats": ["..."], "emotional_deltas": {}},
    "challenge": {"beats": ["..."], "emotional_deltas": {}},
    "twist": {"beats": ["..."], "emotional_deltas": {}},
    "recoupling": {"beats": ["..."], "emotional_deltas": {}},
    "fallout": {"beats": ["..."], "emotional_deltas": {}},
    "night": {"beats": ["..."], "emotional_deltas": {}}
  }
}
"""


def run_scene_layer(outcomes: list[dict], events: list[dict]) -> dict:
    prompt = f"""
Current characters:
{json.dumps(characters, indent=2)}

Outcome decisions:
{json.dumps(outcomes, indent=2)}

System events:
{json.dumps(events, indent=2)}
"""
    fallback_scenes = {scene: {"beats": [], "emotional_deltas": {}} for scene in SCENE_ORDER}
    fallback = {"storylines": [], "scenes": fallback_scenes}
    data, _ = call_llm_json(SCENE_LAYER_SYSTEM, prompt,
                             required_keys=["storylines", "scenes"],
                             fallback=fallback, temperature=0.8)

    # Make sure every scene exists even if the model skipped one
    for scene in SCENE_ORDER:
        data.setdefault("scenes", {})
        data["scenes"].setdefault(scene, {"beats": [], "emotional_deltas": {}})

    return data


# ---- Deterministic scene snapshot calculator (no LLM) ------------------------

def compute_scene_snapshots(scenes_data: dict) -> dict:
    """Apply each scene's emotional_deltas in order, producing an escalating
    snapshot per scene. The final snapshot becomes the new persisted baseline."""
    snapshot = {name: dict(data.get("emotional_state", DEFAULT_EMOTIONAL_STATE))
                 for name, data in characters.items()}

    scene_snapshots = {}
    for scene_name in SCENE_ORDER:
        scene = scenes_data.get("scenes", {}).get(scene_name, {})
        for name, delta in scene.get("emotional_deltas", {}).items():
            if name not in snapshot:
                snapshot[name] = dict(DEFAULT_EMOTIONAL_STATE)
            for k, v in delta.items():
                try:
                    snapshot[name][k] = clamp(snapshot[name].get(k, 0.5) + float(v))
                except (TypeError, ValueError):
                    pass
        scene_snapshots[scene_name] = {n: dict(s) for n, s in snapshot.items()}

    # Persist the final snapshot as the new baseline emotional state
    for name, emo in snapshot.items():
        if name in characters:
            characters[name]["emotional_state"] = emo

    return scene_snapshots


# ---- Stage 4: Script Writer ---------------------------------------------------

SCRIPT_WRITER_SYSTEM = """You are a reality TV scriptwriter. You ONLY expand the
beats you're given, in the exact order provided. You do NOT invent new events,
new characters, new outcomes, or new system events beyond what's in the
blueprint — that would break continuity for future episodes. If a beat is
vague, you may add small color (a glance, a tone of voice) but the SUBSTANCE
of what happens must come only from the blueprint.

For each scene, write a heading and then prose + dialogue + confessional-style
asides ("CONFESSIONAL: Name - ..."). Use "scene_snapshots" to show escalating
emotional pressure — someone whose jealousy is climbing scene-to-scene should
visibly become more on edge as the episode progresses. When a contestant's want
was "denied" or "complicated" (see wants_vs_outcomes), show their reaction
honestly — don't soften it into a win. Make system_events feel like real
production announcements (text messages read aloud, or a host announcement).

Scene order: morning, confessionals, gossip, challenge, twist, recoupling,
fallout, night.

BLUEPRINT:
{blueprint}
"""


def run_script_writer(outcomes: list[dict], events: list[dict], scenes_data: dict,
                       scene_snapshots: dict) -> str:
    blueprint = {
        "wants_vs_outcomes": outcomes,
        "system_events": events,
        "storylines": scenes_data.get("storylines", []),
        "scenes": scenes_data.get("scenes", {}),
        "scene_snapshots": scene_snapshots,
    }
    prompt = SCRIPT_WRITER_SYSTEM.format(blueprint=json.dumps(blueprint, indent=2))

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        messages=[
            {"role": "system", "content": "You are a reality TV writer."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------
@app.get("/generate")
def generate(debug: bool = False):
    global episode_number

    if not submissions:
        return {"episode": "No submissions yet — add some inputs first!"}

    episode_number += 1

    # Stage 1: State Builder
    state_data = run_state_builder()
    merge_state(state_data.get("characters", {}))

    public_opinion = compute_public_opinion()

    # Stage 2a: Outcome Decider
    outcomes_data = run_outcome_decider(public_opinion)
    outcomes = outcomes_data.get("wants_vs_outcomes", [])

    # Stage 2b: System Event Maker
    events_data = run_system_event_maker(outcomes, public_opinion)
    events = events_data.get("system_events", [])

    # Deterministic: apply outcomes + events to persistent memory
    apply_outcomes_and_events(outcomes, events)

    # Stage 3: Scene Layer
    scenes_data = run_scene_layer(outcomes, events)

    # Deterministic: escalating emotional snapshots, persisted as new baseline
    scene_snapshots = compute_scene_snapshots(scenes_data)

    # Stage 4: Script Writer (expands beats only)
    episode = run_script_writer(outcomes, events, scenes_data, scene_snapshots)

    # New episode = fresh round of submissions (character memory persists)
    submissions.clear()

    result = {"episode": episode}
    if debug:
        result["episode_number"] = episode_number
        result["public_opinion"] = public_opinion
        result["outcomes"] = outcomes
        result["events"] = events
        result["scenes"] = scenes_data
        result["scene_snapshots"] = scene_snapshots
        result["characters"] = characters

    return result

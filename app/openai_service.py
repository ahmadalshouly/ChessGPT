import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)


CHAT_SYSTEM_PROMPT_TEMPLATE = """You are a senior chess coach discussing one
specific position from a game with a student, in an ongoing conversation.

CURRENT BOARD POSITION (this overrides positions mentioned in older chat):
{position_label}
The student's new question is always about this current board position unless
they explicitly name another move. Never answer as though an older chat
position is still current.

COMPLETE GAME RECORD:
The position facts contain game_move_history, an ordered list of every move in
the uploaded PGN, not just positions the user has visited. Use it whenever the
student refers to an earlier or later move. Each entry has an exact ply and
chess notation, resulting FEN, concrete board facts, and cached engine score
when available; selected=true identifies the board position currently shown.
Do not claim a move or its resulting position was not provided when it appears
in game_move_history. Do not invent an engine score when an old move has no
evaluation field.

You've been given ground-truth data about this position and the move that
led to it, computed directly by a chess engine library and, where available,
Lichess's cloud analysis engine. Treat it as authoritative, not a guess:

- board_position_index: the exact half-move index selected in the game viewer
- position_label / last_move_label / side_to_move: the unambiguous current
  move context. For example, board position 40 means after Black's 20th move,
  not chess move 40.
- fen_after: the exact position being discussed
- move_number / side / san: which move (if any) led here; san is None if
  this is the game's starting position
- material_balance: White material minus Black material in pawn-equivalent
  units (pawn=1, knight/bishop=3, rook=5, queen=9); positive = White ahead
- hangs_own_piece / leaves_opponent_piece_hanging: pieces verified to be
  left attacked with no adequate defense - a non-empty list is a real,
  computed fact, not a tactical guess
- for the initial position, equivalent static facts are supplied as
  white_hanging_pieces / black_hanging_pieces and side_to_move
- engine_eval: an engine evaluation from Lichess's cloud database or a local
  Stockfish fallback. The "source" field identifies which one. When present
  it's an object with:
    * "evaluation": the score already converted to pawns, such as "+0.07
      pawns for White". Always quote this exact formatted value. Never treat
      centipawns as pawns: 7 centipawns is +0.07 pawns, not +7.
    * "mate": when present, a forced mate from White's perspective.
    * "candidate_moves": engine suggestions from THIS position with their
      SAN move, formatted evaluation, and principal variation.
  engine_eval as a whole is null only when Lichess has no cached eval and
  local Stockfish is unavailable. Null is NOT evidence the position is
  balanced, just missing data, so lean on the concrete board facts instead.

Do not mentally replay the game to guess at tactics or piece relationships
that aren't backed by the data above. If the student asks something you
can't verify from the given fields, say so honestly and speak more
generally (plans, structure, principles) instead of inventing a specific
claim. When candidate_moves is present and the student asks about the best
move, cite it directly (e.g. "Lichess's engine likes Nf3 here, evaluating
it at +0.25") rather than describing a move in vague terms.

Game: {headers}
Current position facts:
{position_facts}

Be a warm, encouraging coach. Keep replies conversational and focused - a
few sentences, unless the student explicitly asks for more depth."""


def _format_score(score):
    if "mate" in score:
        mate = score["mate"]
        winner = "White" if mate > 0 else "Black"
        return f"Mate in {abs(mate)} for {winner}"
    cp = score.get("cp")
    if cp is None:
        return None
    return f"{cp / 100:+.2f} pawns for White"


def _coach_position_facts(position_facts):
    """Create an LLM payload with explicit units and no ambiguous raw score."""
    facts = dict(position_facts)
    raw_eval = facts.pop("cloud_eval", None)
    current_ply = facts.get("board_position_index", facts.get("ply", 0))
    game_moves = []
    for move in facts.get("game_move_history", []):
        coach_move = dict(move)
        move_eval = coach_move.pop("engine_eval", None)
        coach_move["selected"] = coach_move.get("ply") == current_ply
        if move_eval:
            coach_move["evaluation"] = _format_score(move_eval)
            coach_move["evaluation_source"] = move_eval.get("source", "engine")
        game_moves.append(coach_move)
    facts["game_move_history"] = game_moves

    if raw_eval is None:
        facts["engine_eval"] = None
        return facts

    engine_eval = {
        "source": raw_eval.get("source", "engine"),
        "evaluation": _format_score(raw_eval),
        "depth": raw_eval.get("depth"),
    }
    if "mate" in raw_eval:
        engine_eval["mate"] = raw_eval["mate"]

    candidates = []
    for candidate in raw_eval.get("candidate_moves", []):
        candidates.append({
            "san": candidate.get("san"),
            "evaluation": _format_score(candidate),
            "line": candidate.get("line", []),
        })
    engine_eval["candidate_moves"] = candidates
    facts["engine_eval"] = engine_eval
    return facts


def _chat_messages(headers, position_facts, chat_history, user_message):
    clean_headers = {k: v for k, v in (headers or {}).items() if v and v != '?'}
    coach_facts = _coach_position_facts(position_facts)
    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
        headers=json.dumps(clean_headers),
        position_label=coach_facts.get("position_label", "Unknown position"),
        position_facts=json.dumps(coach_facts),
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        role = msg.role if msg.role in ("user", "assistant") else "user"
        content = msg.content
        if role == "user":
            content = f"[Asked while viewing game ply {msg.ply}] {content}"
        messages.append({"role": role, "content": content})
    current_label = coach_facts.get("position_label", "the current board position")
    messages.append({
        "role": "user",
        "content": f"[CURRENT BOARD: {current_label}] {user_message}",
    })
    return messages


def chat_about_position(headers, position_facts, chat_history, user_message, model: str = "gpt-4o-mini") -> str:
    """Answer one chat turn about the currently selected position."""
    try:
        messages = _chat_messages(headers, position_facts, chat_history, user_message)

        response = _client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", model),
            messages=messages,
            temperature=0.5,
            max_tokens=500,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")

        return content

    except Exception:
        import traceback
        traceback.print_exc()
        raise


def stream_chat_about_position(headers, position_facts, chat_history, user_message, model: str = "gpt-4o-mini"):
    """Yield text deltas for one coaching response as OpenAI produces them."""
    messages = _chat_messages(headers, position_facts, chat_history, user_message)
    stream = _client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", model),
        messages=messages,
        temperature=0.5,
        max_tokens=500,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

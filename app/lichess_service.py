"""Lichess Cloud Evaluation client.

Looks up engine evaluations for positions from Lichess's cloud eval
database (https://lichess.org/api#tag/Analysis/operation/apiCloudEval).

Important: Lichess only serves evals for positions it has already
analyzed - mostly opening theory and positions many users have looked at
via their site. There are only a few million positions in that database,
so most middlegame/endgame positions from an arbitrary game will simply
have no eval. A miss (404) is the expected, normal case, not an error -
callers should treat `None` as "no data available."

Results are cached in CloudEvalCache (see app/models.py) so re-analyzing
the same game, or a different game that transposes into the same
position, doesn't repeat requests to Lichess.
"""
import time
import json
import os
import shutil
import glob
import requests
import chess
import chess.engine

from app import db
from app.models import CloudEvalCache

LICHESS_CLOUD_EVAL_URL = "https://lichess.org/api/cloud-eval"

# Lichess asks unauthenticated clients to stay well under their rate limit.
# When we're looking up many new positions from one game in a row, pace the
# requests instead of firing them as fast as possible.
_REQUEST_DELAY_SECONDS = 0.3

_HEADERS = {
    # Identify the app per Lichess's API etiquette; update the contact URL
    # for your own deployment.
    "User-Agent": "chessgpt-app/1.0"
}


def _fetch_from_lichess(fen, multi_pv=3, timeout=5):
    """Raw HTTP call to Lichess's cloud-eval endpoint.

    Returns the parsed JSON response, or None if Lichess has no cached
    eval for this position (HTTP 404 - the normal/expected case) or the
    request otherwise fails.
    """
    try:
        response = requests.get(
            LICHESS_CLOUD_EVAL_URL,
            params={"fen": fen, "multiPv": multi_pv},
            headers=_HEADERS,
            timeout=timeout,
        )

        if response.status_code == 404:
            return None

        if response.status_code == 429:
            # Back off once and retry a single time rather than continuing
            # to hammer an API that just told us to slow down.
            time.sleep(2)
            response = requests.get(
                LICHESS_CLOUD_EVAL_URL,
                params={"fen": fen, "multiPv": multi_pv},
                headers=_HEADERS,
                timeout=timeout,
            )

        if response.status_code != 200:
            return None

        return response.json()

    except requests.RequestException as e:
        print(f"Lichess cloud eval request failed for fen={fen!r}: {e}")
        return None


def _san_line(board, uci_moves, max_moves=6):
    """Convert a Lichess UCI principal variation to SAN."""
    san_moves = []
    for uci in (uci_moves or '').split()[:max_moves]:
        try:
            move = board.parse_uci(uci)
            san_moves.append(board.san(move))
            board.push(move)
        except ValueError:
            break
    return san_moves


def _position_eval(fen, eval_json):
    """Reduce a cloud-eval response to a score and useful engine lines.

    Lichess scores are relative to the side to move. They are normalized to
    White's perspective here, and principal variations are converted to SAN.
    """
    if not eval_json:
        return None
    pvs = eval_json.get("pvs") or []
    if not pvs:
        return None
    board = chess.Board(fen)
    score_sign = 1 if board.turn == chess.WHITE else -1
    result = {"perspective": "white", "source": "lichess"}
    top = pvs[0]
    if "mate" in top:
        result["mate"] = score_sign * top["mate"]
    elif "cp" in top:
        result["cp"] = score_sign * top["cp"]
    else:
        return None

    candidates = []
    for pv in pvs:
        line = _san_line(board.copy(), pv.get("moves"))
        if not line:
            continue
        candidate = {"san": line[0], "line": line}
        if "mate" in pv:
            candidate["mate"] = score_sign * pv["mate"]
        elif "cp" in pv:
            candidate["cp"] = score_sign * pv["cp"]
        candidates.append(candidate)

    if candidates:
        result["candidate_moves"] = candidates
    if eval_json.get("depth") is not None:
        result["depth"] = eval_json["depth"]
    return result


def _stockfish_path():
    configured_path = os.getenv("STOCKFISH_PATH")
    if configured_path and os.path.isfile(configured_path):
        return configured_path
    path = shutil.which("stockfish")
    if path:
        return path

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        winget_pattern = os.path.join(
            local_app_data,
            "Microsoft",
            "WinGet",
            "Packages",
            "Stockfish.Stockfish_*",
            "stockfish",
            "stockfish*.exe",
        )
        matches = glob.glob(winget_pattern)
        if matches:
            return matches[0]
    return None


def _score_to_white(score, board):
    white_score = score.pov(chess.WHITE)
    mate = white_score.mate()
    if mate is not None:
        return {"mate": mate}
    cp = white_score.score()
    return {"cp": cp} if cp is not None else None


def _stockfish_eval(engine, fen, depth=16, multi_pv=3):
    """Analyze one FEN locally and return the same shape as cloud eval."""
    board = chess.Board(fen)
    analyses = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=multi_pv,
    )
    if not isinstance(analyses, list):
        analyses = [analyses]
    if not analyses:
        return None

    result = {
        "perspective": "white",
        "source": "stockfish",
        "depth": analyses[0].get("depth", depth),
    }
    top_score = _score_to_white(analyses[0]["score"], board)
    if not top_score:
        return None
    result.update(top_score)

    candidates = []
    for analysis in analyses:
        pv = analysis.get("pv") or []
        if not pv:
            continue
        line_board = board.copy()
        san_line = []
        for move in pv[:6]:
            san_line.append(line_board.san(move))
            line_board.push(move)
        candidate = {"san": san_line[0], "line": san_line}
        candidate_score = _score_to_white(analysis["score"], board)
        if candidate_score:
            candidate.update(candidate_score)
        candidates.append(candidate)
    if candidates:
        result["candidate_moves"] = candidates
    return result


def _read_cache(fen):
    """Returns (score, cache_hit). score is None either way if the cached
    entry recorded a miss (found=False)."""
    entry = db.session.get(CloudEvalCache, fen)
    if entry is None:
        return None, False
    score = json.loads(entry.eval_json) if entry.found else None
    # Entries produced before scores were normalized can have the wrong sign
    # when Black is to move. Treat them as misses and refresh them once.
    if score is not None and score.get("perspective") != "white":
        return None, False
    return score, True


def _write_cache(fen, score):
    entry = CloudEvalCache(
        fen=fen,
        eval_json=json.dumps(score) if score else None,
        found=score is not None,
    )
    db.session.merge(entry)
    db.session.commit()


def get_cloud_eval(fen):
    """Get the cloud eval for a single FEN, checking the cache first.

    Returns {"cp": int} or {"mate": int} (White's perspective), or None.
    """
    score, hit = _read_cache(fen)
    if hit:
        return score

    raw = _fetch_from_lichess(fen)
    score = _position_eval(fen, raw)
    _write_cache(fen, score)
    return score


def get_position_eval(fen):
    """Use Lichess first, then local Stockfish when the cloud has no data."""
    score, hit = _read_cache(fen)
    if hit and score is not None:
        return score

    raw = _fetch_from_lichess(fen)
    score = _position_eval(fen, raw)
    if score is not None:
        _write_cache(fen, score)
        return score

    path = _stockfish_path()
    if not path:
        _write_cache(fen, None)
        return None

    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        score = _stockfish_eval(engine, fen)
    _write_cache(fen, score)
    return score


def get_cached_eval_only(fen):
    """Read-only cache lookup - never hits the network.

    Use this on pages that render on every load (like the game viewer) so
    an un-evaluated game doesn't make the page slow. Returns the cached
    score, or None if there's no cache entry yet (which is indistinguishable
    here from "Lichess has no eval" - both just mean "nothing to show").
    """
    score, hit = _read_cache(fen)
    return score if hit else None


def annotate_move_facts_with_cloud_eval(move_facts, delay=_REQUEST_DELAY_SECONDS):
    """Attach engine evaluations to every supplied position.

    Lichess cloud analysis is preferred. Cloud misses are analyzed by one
    reusable local Stockfish process when STOCKFISH_PATH is configured.
    """
    stockfish = None
    stockfish_path = _stockfish_path()
    try:
        for fact in move_facts:
            fen = fact.get("fen_after")
            if not fen:
                fact["cloud_eval"] = None
                continue

            score, hit = _read_cache(fen)
            if not hit or score is None:
                raw = _fetch_from_lichess(fen)
                score = _position_eval(fen, raw)
                if score is None and stockfish_path:
                    if stockfish is None:
                        stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    score = _stockfish_eval(stockfish, fen)
                _write_cache(fen, score)
                time.sleep(delay)

            fact["cloud_eval"] = score
    finally:
        if stockfish is not None:
            stockfish.quit()

    return move_facts

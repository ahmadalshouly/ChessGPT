"""PGN parsing and analysis-grounding helpers.

Kept separate from routes.py (same pattern as openai_service.py) so route
handlers stay focused on request/response handling, not chess/PGN mechanics.
"""
import io
import chess
import chess.pgn


_QUOTE_REPLACEMENTS = {
    '\u201c': '"',  # “
    '\u201d': '"',  # ”
    '\u2018': "'",  # ‘
    '\u2019': "'",  # ’
    '\u2033': '"',  # ″
    '\u2032': "'",  # ′
}


def normalize_pgn_text(pgn_text):
    """Some PGN sources (copy-pasted from word processors, chat apps, etc.)
    use curly/smart quotes around header values, e.g. [White "name"].
    python-chess's tag-pair parser only recognizes straight double quotes,
    so headers silently fail to parse (falling back to '?') when this
    happens. Normalize common smart-quote variants to straight quotes."""
    if not pgn_text:
        return pgn_text
    for bad, good in _QUOTE_REPLACEMENTS.items():
        pgn_text = pgn_text.replace(bad, good)
    return pgn_text


def clean_header(value):
    """python-chess fills missing standard tags with '?' rather than leaving
    them absent, so treat '?' (and blank) the same as 'not provided'."""
    if value is None:
        return None
    value = value.strip()
    if value in ('', '?'):
        return None
    return value


def parse_pgn(pgn_text):
    """Parse PGN text and return (headers, moves, fens).

    headers: dict of PGN header tags (raw, not cleaned)
    moves: list of SAN strings, one per ply
    fens: list of FEN strings, fens[0] is the start position, fens[i] is the
          position after moves[i-1] was played
    """
    headers = {}
    moves = []
    fens = [chess.STARTING_FEN]

    if not pgn_text:
        return headers, moves, fens

    pgn_text = normalize_pgn_text(pgn_text)

    try:
        pgn_io = io.StringIO(pgn_text)
        pgn_game = chess.pgn.read_game(pgn_io)

        if pgn_game:
            headers = dict(pgn_game.headers)
            board = pgn_game.board()
            fens = [board.fen()]

            for move in pgn_game.mainline_moves():
                san = board.san(move)
                board.push(move)
                moves.append(san)
                fens.append(board.fen())

    except Exception as e:
        print(f"PGN parse error: {e}")

    return headers, moves, fens


_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _material_balance(board):
    """White material minus Black material, in pawn-equivalent units."""
    balance = 0
    for piece in board.piece_map().values():
        value = _PIECE_VALUES.get(piece.piece_type, 0)
        balance += value if piece.color == chess.WHITE else -value
    return balance


def _hanging_pieces(board):
    """Pieces currently left en prise: attacked with no defenders, or
    attacked by a cheaper piece with no way to recapture favorably.

    This is a simplified heuristic (not a full static-exchange evaluation,
    and it won't catch multi-move combinations), but it reliably catches
    real one-move blunders and missed captures — the exact kind of claim
    an LLM cannot reliably verify on its own from a FEN string alone.

    Returns a list of {"color": "White"/"Black", "piece": "knight", "square": "c4"}.
    """
    hanging = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        color = piece.color
        attackers = board.attackers(not color, square)
        if not attackers:
            continue

        defenders = board.attackers(color, square)
        min_attacker_value = min(
            _PIECE_VALUES.get(board.piece_type_at(sq), 0) for sq in attackers
        )
        piece_value = _PIECE_VALUES.get(piece.piece_type, 0)

        if len(defenders) == 0 or min_attacker_value < piece_value:
            hanging.append({
                "color": "White" if color == chess.WHITE else "Black",
                "piece": chess.piece_name(piece.piece_type),
                "square": chess.square_name(square),
            })

    return hanging


def build_position_facts(fen):
    """Compute static facts for a position that was not reached by a move."""
    board = chess.Board(fen)
    hanging = _hanging_pieces(board)
    return {
        "material_balance": _material_balance(board),
        "white_hanging_pieces": [h for h in hanging if h["color"] == "White"],
        "black_hanging_pieces": [h for h in hanging if h["color"] == "Black"],
        "side_to_move": "White" if board.turn == chess.WHITE else "Black",
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
    }


def build_move_facts(pgn_text):
    """Compute concrete, verifiable facts for every move using python-chess
    itself — not the AI's mental simulation of the game.

    LLMs asked to "play through" a PGN in their head reliably drift after
    a couple dozen half-moves and start inventing piece relationships that
    aren't real (e.g. claiming two rooks are connected when they aren't),
    and they can't reliably judge whether a move was a blunder just by
    reading move text. python-chess already computes the exact board after
    every move and can check what's attacked/defended, so we hand that
    ground truth to the AI instead of making it guess.

    Returns (headers, move_facts) where move_facts is a list of dicts:
    {
        "ply": 1-based half-move number,
        "move_number": full move number (1. e4 e5 -> both are move_number 1),
        "side": "White" | "Black",
        "san": "<SAN>",
        "fen_after": "<FEN after this move>",
        "is_capture": bool,
        "is_check": bool,
        "is_checkmate": bool,
        "is_castle": bool,
        "material_balance": White material minus Black material (pawn=1,
            knight/bishop=3, rook=5, queen=9), computed after this move,
        "hangs_own_piece": pieces belonging to the side who just moved that
            are now left attacked with no adequate defense (a real,
            computed blunder signal, not a guess),
        "leaves_opponent_piece_hanging": opponent pieces left attacked with
            no adequate defense after this move (a real missed-tactic
            signal for the mover),
    }
    """
    headers = {}
    move_facts = []

    if not pgn_text:
        return headers, move_facts

    pgn_text = normalize_pgn_text(pgn_text)

    try:
        pgn_io = io.StringIO(pgn_text)
        pgn_game = chess.pgn.read_game(pgn_io)

        if pgn_game:
            headers = dict(pgn_game.headers)
            board = pgn_game.board()

            for i, move in enumerate(pgn_game.mainline_moves()):
                is_capture = board.is_capture(move)
                is_castle = board.is_castling(move)
                san = board.san(move)
                side = "White" if board.turn == chess.WHITE else "Black"
                move_number = board.fullmove_number

                board.push(move)

                hanging = _hanging_pieces(board)
                own_hanging = [h for h in hanging if h["color"] == side]
                opponent_hanging = [h for h in hanging if h["color"] != side]

                move_facts.append({
                    "ply": i + 1,
                    "move_number": move_number,
                    "side": side,
                    "san": san,
                    "fen_after": board.fen(),
                    "is_capture": is_capture,
                    "is_check": board.is_check(),
                    "is_checkmate": board.is_checkmate(),
                    "is_castle": is_castle,
                    "material_balance": _material_balance(board),
                    "hangs_own_piece": own_hanging,
                    "leaves_opponent_piece_hanging": opponent_hanging,
                })

    except Exception as e:
        print(f"PGN parse error: {e}")

    return headers, move_facts


def _normalize_san(san):
    """Strip check/mate/annotation glyphs so 'Qh5+' and 'Qh5' compare equal."""
    return (san or '').strip().rstrip('+#!?')


def align_move_analysis(real_moves, ai_moves):
    """Line up the AI's per-move comments with the actual moves of the game.

    Returns a list the same length as real_moves, each entry
    {"move": <real SAN>, "comment": <AI comment or None>}. Even if the AI's
    list is missing entries, has extras, or drifts, this searches a small
    window around the expected position so later moves don't end up matched
    to the wrong comment.
    """
    aligned = []
    ai_idx = 0
    n_ai = len(ai_moves)

    for real_san in real_moves:
        target = _normalize_san(real_san)
        comment = None

        window = range(max(0, ai_idx - 2), min(n_ai, ai_idx + 6))
        for j in window:
            entry = ai_moves[j] if isinstance(ai_moves[j], dict) else {}
            if _normalize_san(entry.get('move')) == target:
                comment = entry.get('comment')
                ai_idx = j + 1
                break

        aligned.append({"move": real_san, "comment": comment})

    return aligned

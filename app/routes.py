from flask import Blueprint, Response, render_template, redirect, url_for, flash, request, session, abort, jsonify, stream_with_context
from app import db
from app.models import User, Game, ChatMessage
from app.openai_service import chat_about_position, stream_chat_about_position
from app.pgn_service import (
    normalize_pgn_text,
    clean_header,
    parse_pgn,
    build_move_facts,
    build_position_facts,
)
from app.lichess_service import annotate_move_facts_with_cloud_eval, get_position_eval, get_cached_eval_only
import chess.pgn
import io
import json

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already taken')
            return redirect(url_for('main.register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registered successfully! Please log in.')
        return redirect(url_for('main.login'))

    return render_template('register.html')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Logged in successfully!')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@bp.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('main.index'))


@bp.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    user = db.session.get(User, session['user_id'])
    if user is None:
        session.pop('user_id', None)
        return redirect(url_for('main.login'))
    games = Game.query.filter_by(user_id=user.id).order_by(Game.created_at.desc()).all()

    return render_template('dashboard.html', user=user, games=games)


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    if request.method == 'POST':
        title = request.form.get('title', 'Untitled Game')
        pgn_text = normalize_pgn_text(request.form.get('pgn_text', '').strip())

        if not pgn_text:
            flash('Please paste a PGN')
            return redirect(url_for('main.upload'))

        try:
            pgn_io = io.StringIO(pgn_text)
            parsed_game = chess.pgn.read_game(pgn_io)
            if parsed_game is None:
                raise ValueError("Invalid PGN")
            board = parsed_game.board()
            for move in parsed_game.mainline_moves():
                board.push(move)
            final_fen = board.fen()
        except Exception as e:
            flash(f'Invalid PGN: {str(e)}')
            return redirect(url_for('main.upload'))

        # Save to database
        new_game = Game(
            title=title,
            pgn_text=pgn_text,
            fen_position=final_fen,
            user_id=session['user_id']
        )
        db.session.add(new_game)
        db.session.commit()

        flash('Game uploaded successfully!')
        return redirect(url_for('main.dashboard'))

    return render_template('upload.html')


@bp.route('/game/<int:game_id>')
def game(game_id):
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    game = Game.query.get_or_404(game_id)

    if game.user_id != session['user_id']:
        abort(403)

    headers, moves, fens = parse_pgn(game.pgn_text)

    white_name = clean_header(headers.get('White')) or 'Unknown'
    black_name = clean_header(headers.get('Black')) or 'Unknown'
    white_elo = clean_header(headers.get('WhiteElo'))
    black_elo = clean_header(headers.get('BlackElo'))

    # Cache-only lookup - never hits the network, so viewing the page is
    # always fast. Positions not yet fetched (via /evaluate or chat) just
    # show as None; the frontend already handles that.
    move_evals = [get_cached_eval_only(fen) for fen in fens]

    return render_template('game.html',
                           game=game,
                           moves=moves,
                           fens=fens,
                           move_evals=move_evals,
                           white_name=white_name,
                           black_name=black_name,
                           white_elo=white_elo,
                           black_elo=black_elo)


@bp.route('/game/<int:game_id>/evaluate', methods=['POST'])
def evaluate_with_lichess(game_id):
    """Fetch and cache Lichess cloud eval (score + candidate moves) for
    every position in the game. This is the only "analysis" step now -
    OpenAI no longer generates a batch game review; it only chats,
    grounded in this Lichess data (see /game/<id>/chat below).
    """
    if 'user_id' not in session:
        return redirect(url_for('main.login'))

    game = Game.query.get_or_404(game_id)
    if game.user_id != session['user_id']:
        abort(403)

    _, move_facts = build_move_facts(game.pgn_text)
    _, _, fens = parse_pgn(game.pgn_text)

    try:
        facts_to_evaluate = [{"fen_after": fens[0]}] + move_facts
        annotate_move_facts_with_cloud_eval(facts_to_evaluate)
        evaluated = sum(fact.get("cloud_eval") is not None for fact in facts_to_evaluate)
        total = len(facts_to_evaluate)
        if evaluated == total:
            flash(f'Engine evaluation complete: all {total} positions analyzed!')
        else:
            flash(
                f'Analyzed {evaluated} of {total} positions. Configure '
                'STOCKFISH_PATH to evaluate positions missing from Lichess.'
            )
    except Exception as e:
        print(f"Lichess evaluation failed: {e}")
        flash('Could not reach Lichess right now, please try again.')

    return redirect(url_for('main.game', game_id=game.id))


def _get_position_facts(game, ply):
    """Build the position_facts dict for a given ply of a game.

    ply=0 is the starting position (no move played yet); ply=N is the
    position after move_facts[N-1]. Returns None if ply is out of range.
    """
    headers, move_facts = build_move_facts(game.pgn_text)
    _, _, fens = parse_pgn(game.pgn_text)

    if ply == 0:
        static_facts = build_position_facts(fens[0])
        facts = {
            "ply": 0,
            "move_number": 0,
            "side": None,
            "san": None,
            "fen_after": fens[0],
            **static_facts,
        }
    elif 1 <= ply <= len(move_facts):
        facts = dict(move_facts[ply - 1])
    else:
        return headers, None

    game_move_history = []
    for move_fact in move_facts:
        move_prefix = (
            f'{move_fact["move_number"]}.'
            if move_fact["side"] == "White"
            else f'{move_fact["move_number"]}...'
        )
        game_move_history.append({
            "ply": move_fact["ply"],
            "notation": f'{move_prefix}{move_fact["san"]}',
            "side": move_fact["side"],
            "san": move_fact["san"],
            "fen_after": move_fact["fen_after"],
            "material_balance": move_fact["material_balance"],
            "is_capture": move_fact["is_capture"],
            "is_check": move_fact["is_check"],
            "is_checkmate": move_fact["is_checkmate"],
            "hangs_own_piece": move_fact["hangs_own_piece"],
            "leaves_opponent_piece_hanging": move_fact["leaves_opponent_piece_hanging"],
            "engine_eval": get_cached_eval_only(move_fact["fen_after"]),
        })

    facts["game_move_history"] = game_move_history
    facts["game_move_count"] = len(game_move_history)

    board = chess.Board(facts["fen_after"])
    facts["board_position_index"] = ply
    facts["side_to_move"] = "White" if board.turn == chess.WHITE else "Black"
    if ply == 0:
        facts["position_label"] = "Starting position, before the first move"
        facts["last_move_label"] = None
    else:
        move_prefix = f'{facts["move_number"]}.' if facts["side"] == "White" else f'{facts["move_number"]}...'
        facts["last_move_label"] = f'{move_prefix}{facts["san"]}'
        facts["position_label"] = (
            f'Board position {ply} of the game, after {facts["last_move_label"]}; '
            f'{facts["side_to_move"]} to move'
        )

    # Ground this specific position with Lichess's cloud eval. Cache-first,
    # so this is free if /evaluate (or an earlier chat/message) already
    # fetched this exact position.
    try:
        facts["cloud_eval"] = get_position_eval(facts["fen_after"])
    except Exception as e:
        print(f"Cloud eval lookup failed during chat: {e}")
        facts["cloud_eval"] = None

    return headers, facts


@bp.route('/game/<int:game_id>/chat', methods=['GET', 'POST'])
def chat(game_id):
    if 'user_id' not in session:
        return jsonify({'error': 'not authenticated'}), 401

    game = Game.query.get_or_404(game_id)
    if game.user_id != session['user_id']:
        abort(403)

    if request.method == 'GET':
        messages = (ChatMessage.query
                    .filter_by(game_id=game.id)
                    .order_by(ChatMessage.created_at, ChatMessage.id)
                    .all())
        return jsonify([
            {
                'role': message.role,
                'content': message.content,
                'ply': message.ply,
                'created_at': message.created_at.isoformat(),
            }
            for message in messages
        ])

    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': 'message is required'}), 400

    ply = data.get('ply')
    try:
        ply = int(ply)
    except (TypeError, ValueError):
        return jsonify({'error': 'ply is required and must be an integer'}), 400

    headers, position_facts = _get_position_facts(game, ply)
    if position_facts is None:
        return jsonify({'error': 'invalid ply for this game'}), 400

    history = (ChatMessage.query
               .filter_by(game_id=game.id)
               .order_by(ChatMessage.created_at, ChatMessage.id)
               .all())

    try:
        reply = chat_about_position(headers, position_facts, history, user_message)
    except Exception:
        return jsonify({'error': 'chat failed, please try again'}), 502

    db.session.add(ChatMessage(game_id=game.id, ply=ply, role='user', content=user_message))
    db.session.add(ChatMessage(game_id=game.id, ply=ply, role='assistant', content=reply))
    db.session.commit()

    return jsonify({'reply': reply})


@bp.route('/game/<int:game_id>/chat/stream', methods=['POST'])
def chat_stream(game_id):
    if 'user_id' not in session:
        return jsonify({'error': 'not authenticated'}), 401

    game = Game.query.get_or_404(game_id)
    if game.user_id != session['user_id']:
        abort(403)

    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': 'message is required'}), 400
    try:
        ply = int(data.get('ply'))
    except (TypeError, ValueError):
        return jsonify({'error': 'ply is required and must be an integer'}), 400

    headers, position_facts = _get_position_facts(game, ply)
    if position_facts is None:
        return jsonify({'error': 'invalid ply for this game'}), 400
    history = (ChatMessage.query
               .filter_by(game_id=game.id)
               .order_by(ChatMessage.created_at, ChatMessage.id)
               .all())

    @stream_with_context
    def generate():
        reply_parts = []
        try:
            for delta in stream_chat_about_position(
                headers, position_facts, history, user_message
            ):
                reply_parts.append(delta)
                yield json.dumps({'delta': delta}) + '\n'

            reply = ''.join(reply_parts).strip()
            if not reply:
                raise ValueError('Empty response from OpenAI')
            db.session.add(ChatMessage(
                game_id=game.id, ply=ply, role='user', content=user_message
            ))
            db.session.add(ChatMessage(
                game_id=game.id, ply=ply, role='assistant', content=reply
            ))
            db.session.commit()
            yield json.dumps({'done': True}) + '\n'
        except Exception:
            import traceback
            traceback.print_exc()
            db.session.rollback()
            yield json.dumps({'error': 'chat failed, please try again'}) + '\n'

    return Response(
        generate(),
        content_type='application/x-ndjson',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@bp.route('/game/<int:game_id>/chat/<int:ply>', methods=['GET'])
def chat_history(game_id, ply):
    if 'user_id' not in session:
        return jsonify({'error': 'not authenticated'}), 401

    game = Game.query.get_or_404(game_id)
    if game.user_id != session['user_id']:
        abort(403)

    messages = (ChatMessage.query
                .filter_by(game_id=game.id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
                .all())

    return jsonify([
        {
            'role': m.role,
            'content': m.content,
            'ply': m.ply,
            'created_at': m.created_at.isoformat(),
        }
        for m in messages
    ])

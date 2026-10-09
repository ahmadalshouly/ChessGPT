from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship: one user has many games
    games = db.relationship('Game', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    pgn_text = db.Column(db.Text)
    fen_position = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Game {self.title}>'

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'), nullable=False)
    summary = db.Column(db.Text)
    opening = db.Column(db.String(200))
    moves_analysis = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    game = db.relationship('Game', backref='analysis')


class CloudEvalCache(db.Model):
    """Cache of Lichess cloud-eval lookups, keyed by FEN.

    Positions repeat heavily across games (openings especially), so this
    avoids re-hitting the Lichess API for a FEN we've already looked up -
    whether it came from this game or someone else's. We cache misses too
    (found=False) so we don't keep re-asking about positions Lichess simply
    doesn't have cached.
    """
    __tablename__ = 'cloud_eval_cache'

    fen = db.Column(db.String(100), primary_key=True)
    eval_json = db.Column(db.Text)  # JSON {"cp": int} or {"mate": int}; NULL if found=False
    found = db.Column(db.Boolean, default=True, nullable=False)
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<CloudEvalCache {self.fen} found={self.found}>'


class ChatMessage(db.Model):
    """One message in a continuous chat about a game.

    ``ply`` records which position was displayed when the message was sent;
    it provides position context without splitting the conversation.
    """
    __tablename__ = 'chat_message'

    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey('game.id'), nullable=False)
    ply = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    game = db.relationship('Game', backref='chat_messages')

    def __repr__(self):
        return f'<ChatMessage game={self.game_id} ply={self.ply} role={self.role}>'

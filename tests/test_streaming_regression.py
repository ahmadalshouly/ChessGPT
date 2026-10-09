import json
import unittest
from unittest.mock import patch

from app import create_app, db
from app.models import ChatMessage, Game, User


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class StreamingRegressionTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.context = self.app.app_context()
        self.context.push()
        db.create_all()
        user = User(username="streamer", email="streamer@example.com")
        user.set_password("password")
        db.session.add(user)
        db.session.commit()
        game = Game(title="Stream game", pgn_text="1. e4 e5", user_id=user.id)
        db.session.add(game)
        db.session.commit()
        self.game_id = game.id
        self.client = self.app.test_client()
        self.client.post("/login", data={"username": "streamer", "password": "password"})

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def test_stream_route_persists_deltas(self):
        with patch("app.routes.get_position_eval", return_value=None), patch(
            "app.routes.stream_chat_about_position",
            return_value=iter(["Use ", "the center."]),
        ):
            response = self.client.post(
                f"/game/{self.game_id}/chat/stream",
                json={"ply": 2, "message": "What now?"},
            )

        self.assertEqual(response.status_code, 200)
        events = [json.loads(line) for line in response.data.splitlines()]
        self.assertEqual(events[-1], {"done": True})
        self.assertEqual(ChatMessage.query.count(), 2)
        self.assertEqual(ChatMessage.query.filter_by(role="assistant").one().content, "Use the center.")


if __name__ == "__main__":
    unittest.main()

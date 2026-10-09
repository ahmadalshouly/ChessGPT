# ChessGPT

ChessGPT lets a user upload a PGN, step through every position, retrieve
Lichess Cloud Evaluation scores and principal variations, and chat with an
OpenAI-powered coach grounded in the selected position and engine data.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `SECRET_KEY`.
4. Start the app with `python run.py`.
5. Install Stockfish and set `STOCKFISH_PATH` to its executable.
6. Register, upload a PGN, open the game, and select **Analyze All Positions**.

The database is created automatically on local startup. Each game has one
continuous chat. Messages retain the board position that was visible when
they were sent, so the coach can follow the discussion while the user moves
through the game.

## Engine Availability

The app checks Lichess's public Cloud Evaluation API first. Since Lichess only
returns positions already in its cloud database, uncommon positions are sent
to the local Stockfish executable configured by `STOCKFISH_PATH`. This makes
it possible to evaluate every position in a game. Without local Stockfish,
the app reports how many positions Lichess could evaluate instead of claiming
that partial analysis is complete.

Please set an identifying `User-Agent` with deployment contact information in
`app/lichess_service.py` before deploying publicly, and respect Lichess API
rate limits.

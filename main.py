import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from othello import Othello
from SizedOthello import SizedOthello

app = FastAPI()

# ゲームセッション管理 (game_id -> dict)
games: dict[str, dict] = {}


# ---------- リクエストモデル ----------

class NewGameRequest(BaseModel):
    board_size: int = 8      # 6 or 8
    player_color: int = 1   # 1=黒, 2=白, 0=両方(人間同士)
    ai_type: str = "mcts"   # "random" | "mcts" | "alphazero" | "human"


class MoveRequest(BaseModel):
    x: int
    y: int


# ---------- ヘルパー ----------

def build_state(game_id: str) -> dict:
    g = games[game_id]
    ot = g["othello"]
    winner = ot.winner()
    _, black, white = ot.count()

    # 合法手: ゲーム中のみ
    possible = ot.possible_puts(ot.color) if winner is None else []

    # 最後の着手
    last_move = None
    if ot.history:
        lm = ot.history[-1]
        if lm is not None:
            last_move = list(lm)

    return {
        "game_id": game_id,
        "board": ot.board,
        "current_color": ot.color,
        "player_color": g["player_color"],
        "possible_puts": [list(p) for p in possible],
        "winner": winner,
        "black_count": black,
        "white_count": white,
        "board_size": g["board_size"],
        "skipped": g.get("skipped", False),
        "last_move": last_move,
    }


def do_ai_move(game_id: str) -> None:
    """AIが手番の間ずっと指し続ける。プレイヤーのパスも処理する。"""
    g = games[game_id]
    ot = g["othello"]
    agent = g["ai_agent"]
    player_color = g["player_color"]
    g["skipped"] = False

    while ot.winner() is None and ot.color != player_color:
        # AIがパスの場合
        if not ot.is_possible_to_put_anywhere(ot.color):
            ot.color = 3 - ot.color
            continue

        x, y = agent(ot, ot.color)
        ot.put(x, y, ot.color)
        ot.history.append((x, y))
        ot.color = 3 - ot.color

        # プレイヤーがパスしなければならない場合
        if ot.winner() is None and not ot.is_possible_to_put_anywhere(ot.color):
            g["skipped"] = True
            ot.color = 3 - ot.color  # プレイヤーをスキップ → 再びAIの番へ


# ---------- エンドポイント ----------

@app.post("/api/new_game")
def new_game(req: NewGameRequest):
    game_id = str(uuid.uuid4())

    ot = SizedOthello(6) if req.board_size == 6 else Othello()

    ai_agent = None
    if req.ai_type == "random":
        from play.random_play import random_play
        ai_agent = random_play

    elif req.ai_type == "mcts":
        from play.mcts_agent import mcts_agent
        ai_agent = mcts_agent

    elif req.ai_type == "alphazero":
        if req.board_size != 6:
            raise HTTPException(
                status_code=400,
                detail="AlphaZero は 6×6 ボードのみ対応しています。"
            )
        ckpt = "checkpoints/model_final.pt"
        if not os.path.exists(ckpt):
            raise HTTPException(
                status_code=400,
                detail="AlphaZero のチェックポイントが見つかりません。"
            )
        from play.alphazero_agent import load_alphazero_agent
        ai_agent = load_alphazero_agent(ckpt)

    # ai_type == "human" の場合は ai_agent = None のまま

    games[game_id] = {
        "othello": ot,
        "player_color": req.player_color,
        "ai_agent": ai_agent,
        "board_size": req.board_size,
        "skipped": False,
    }

    # プレイヤーが白(後手)の場合 → AIが先に指す
    if req.player_color == 2 and ai_agent is not None:
        do_ai_move(game_id)

    return build_state(game_id)


@app.get("/api/state/{game_id}")
def get_state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません。")
    return build_state(game_id)


@app.post("/api/move/{game_id}")
def make_move(game_id: str, req: MoveRequest):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="ゲームが見つかりません。")

    g = games[game_id]
    ot = g["othello"]
    player_color = g["player_color"]
    g["skipped"] = False

    if ot.winner() is not None:
        raise HTTPException(status_code=400, detail="ゲームはすでに終了しています。")

    # 人間同士(player_color==0)以外は手番チェック
    if player_color != 0 and ot.color != player_color:
        raise HTTPException(status_code=400, detail="あなたの手番ではありません。")

    if not ot.is_possible_to_put(req.x, req.y, ot.color):
        raise HTTPException(status_code=400, detail="そこには置けません。")

    ot.put(req.x, req.y, ot.color)
    ot.history.append((req.x, req.y))
    ot.color = 3 - ot.color

    # プレイヤーがパスしなければならない場合
    if ot.winner() is None and not ot.is_possible_to_put_anywhere(ot.color):
        ot.color = 3 - ot.color

    # AIが応答
    if (
        ot.winner() is None
        and g["ai_agent"] is not None
        and (player_color == 0 or ot.color != player_color)
    ):
        do_ai_move(game_id)

    return build_state(game_id)


# ---------- フロントエンド配信 ----------

@app.get("/")
def root():
    return FileResponse("static/index.html")

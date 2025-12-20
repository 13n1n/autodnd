"""Flask API application."""

import uuid
from datetime import datetime
from typing import Optional

import os

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from autodnd.agents.game_master import GameMasterAgent
from autodnd.agents.tools import (
    create_get_inventory_tool,
    create_get_map_state_tool,
    create_get_player_stats_tool,
    create_roll_dice_tool,
)
from autodnd.api.llm_config import LLMConfig, LLMConfigManager
from autodnd.engine.game_engine import GameEngine
from autodnd.models.actions import Action, ActionType, TimeCost
from autodnd.models.messages import MessageSource, MessageType
from autodnd.models.player import Player
from autodnd.models.state import GameState
from autodnd.models.stats import PlayerStats
from autodnd.models.world import HexCoordinate, HexMap, TimeState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")


# Error handlers for API routes
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    """Return JSON instead of HTML for HTTP errors in API routes."""
    if request.path.startswith("/api/"):
        response = e.get_response()
        response.data = jsonify(
            {
                "error": e.name,
                "code": e.code,
                "description": e.description,
            }
        ).data
        response.content_type = "application/json"
        return response
    return e


@app.errorhandler(500)
def handle_internal_error(e: Exception):
    """Handle 500 errors."""
    if request.path.startswith("/api/"):
        app.logger.error(f"Internal server error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred"}), 500
    # For non-API routes, return simple error message
    # Flask's default handler will take over if this doesn't return properly
    return f"Internal Server Error", 500


# Global game engine instance
_game_engine: Optional[GameEngine] = None
_game_master: Optional[GameMasterAgent] = None
_llm_config_manager = LLMConfigManager()


def _get_game_engine() -> GameEngine:
    """Get or create game engine."""
    global _game_engine
    if _game_engine is None:
        _game_engine = _create_initial_game()
        _setup_game_master()
    return _game_engine


def _setup_game_master() -> None:
    """Setup game master agent."""
    global _game_master, _game_engine
    if _game_engine is None:
        return

    def state_getter():
        return _game_engine.state

    tools = [
        create_roll_dice_tool(),
        create_get_player_stats_tool(state_getter),
        create_get_inventory_tool(state_getter),
        create_get_map_state_tool(state_getter),
    ]

    llm = _llm_config_manager.get_llm()
    _game_master = GameMasterAgent(llm=llm, tools=tools, engine_getter=lambda: _game_engine)


def _create_initial_game() -> GameEngine:
    """Create initial game state."""
    # Create a default player
    player_id = str(uuid.uuid4())
    player = Player(
        player_id=player_id,
        name="Player 1",
        base_stats=PlayerStats(health=100, max_health=100, strength=10, dexterity=10, intelligence=10, charisma=10),
        current_stats=PlayerStats(health=100, max_health=100, strength=10, dexterity=10, intelligence=10, charisma=10),
        level=1,
        experience=0,
        position=HexCoordinate(q=0, r=0),
    )

    initial_state = GameState(
        game_id=str(uuid.uuid4()),
        state_version=0,
        created_at=datetime.now(),
        players=[player],
        world_map=HexMap(),
        current_time=TimeState(),
    )

    engine = GameEngine(initial_state=initial_state)
    return engine


@app.route("/api/game/start", methods=["POST"])
def start_game():
    """Initialize new game."""
    global _game_engine, _game_master
    _game_engine = _create_initial_game()
    _setup_game_master()
    return jsonify({"success": True, "game_id": _game_engine.state.game_id})


@app.route("/api/game/action", methods=["POST"])
def submit_action():
    """Submit player action."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    action_text = data.get("action", "")
    player_id = data.get("player_id")

    if not action_text:
        return jsonify({"error": "Action text is required"}), 400

    engine = _get_game_engine()
    if not engine.state.players:
        return jsonify({"error": "No players in game"}), 400

    # Use first player if no player_id specified
    if not player_id:
        player_id = engine.state.players[0].player_id

    # Create action from text (simplified - in real implementation, parse action text)
    action = Action(
        action_id=str(uuid.uuid4()),
        action_type=ActionType.TALK,  # Default to TALK for text actions
        parameters={"text": action_text},
        time_cost=TimeCost.NO_TIME,
        player_id=player_id,
    )

    # Apply action
    new_state, success, error_msg = engine.apply_action(action)

    # Get master response using game master agent
    master_response = ""
    global _game_master
    if success and _game_master:
        # Get recent message history for context
        recent_messages = [
            {
                "source": msg.source.value,
                "content": msg.content,
                "message_type": msg.message_type.value,
            }
            for msg in engine.state.message_history.messages[-10:]
        ]

        try:
            master_response = _game_master.process_action(action_text, recent_messages)
            # Add master response to state
            engine.add_message(
                content=master_response,
                source=MessageSource.MASTER,
                message_type=MessageType.RESPONSE,
            )
            new_state = engine.state
        except Exception as e:
            app.logger.error(f"Error generating master response: {e}", exc_info=True)
            # Add error message to state
            engine.add_message(
                content=f"Master encountered an error: {str(e)}. Please try again.",
                source=MessageSource.SYSTEM,
                message_type=MessageType.SYSTEM,
                metadata={"error": str(e)},
            )
            master_response = "Master encountered an error processing your action."
            new_state = engine.state

    return jsonify(
        {
            "success": success,
            "error": error_msg if not success else None,
            "state": _serialize_state_for_api(new_state),
            "master_response": master_response,
        }
    )


@app.route("/api/game/state", methods=["GET"])
def get_state():
    """Get current game state."""
    engine = _get_game_engine()
    return jsonify({"state": _serialize_state_for_api(engine.state)})


@app.route("/api/game/revert", methods=["POST"])
def revert_to_snapshot():
    """Revert to snapshot."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    snapshot_index = data.get("snapshot_index")
    if snapshot_index is None:
        return jsonify({"error": "snapshot_index is required"}), 400

    engine = _get_game_engine()
    success, error_msg = engine.revert_to(snapshot_index)

    if success:
        return jsonify({"success": True, "state": _serialize_state_for_api(engine.state)})
    else:
        return jsonify({"success": False, "error": error_msg}), 400


@app.route("/api/game/history", methods=["GET"])
def list_snapshots():
    """List available snapshots."""
    engine = _get_game_engine()
    snapshots = engine.history.list_snapshots()
    return jsonify(
        {
            "snapshots": [
                {
                    "index": snap.index,
                    "timestamp": snap.timestamp.isoformat(),
                    "metadata": snap.metadata,
                }
                for snap in snapshots
            ]
        }
    )


@app.route("/api/config/llm", methods=["GET"])
def get_llm_config():
    """Get current LLM configuration."""
    return jsonify({"config": _llm_config_manager.config.model_dump()})


@app.route("/api/config/llm", methods=["POST"])
def update_llm_config():
    """Update LLM configuration (hot-reload)."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        new_config = LLMConfig(**data)
        _llm_config_manager.update_config(new_config)
        # Update game master with new LLM
        global _game_master
        if _game_master:
            _game_master.update_llm(_llm_config_manager.get_llm())
        else:
            # Recreate game master if it doesn't exist
            _setup_game_master()
        return jsonify({"success": True, "config": _llm_config_manager.config.model_dump()})
    except Exception as e:
        app.logger.error(f"Error updating LLM config: {e}", exc_info=True)
        return jsonify({"error": "Invalid configuration", "message": str(e)}), 400


@app.route("/")
def index():
    """Serve main page."""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "Frontend not found"}), 404


def _serialize_state_for_api(state: GameState) -> dict:
    """Serialize state for API (exclude internal data)."""
    return {
        "game_id": state.game_id,
        "state_version": state.state_version,
        "created_at": state.created_at.isoformat(),
        "players": [player.model_dump() for player in state.players],
        "current_time": state.current_time.model_dump(),
        "message_history": {
            "messages": [
                {
                    "message_id": msg.message_id,
                    "timestamp": msg.timestamp.isoformat(),
                    "sequence_number": msg.sequence_number,
                    "source": msg.source.value,
                    "source_id": msg.source_id,
                    "content": msg.content,
                    "message_type": msg.message_type.value,
                    "tool_name": msg.tool_name,
                    "npc_id": msg.npc_id,
                    "metadata": msg.metadata,
                }
                for msg in state.message_history.messages
            ]
        },
        "combat_state": state.combat_state.model_dump() if state.combat_state else None,
        "metadata": state.metadata.model_dump(),
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)


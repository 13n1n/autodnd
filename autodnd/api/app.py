"""Flask API application."""

import uuid
from datetime import datetime
from typing import Optional

import os
import logging
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
from autodnd.api.security_config import SecurityConfig, SecurityConfigManager
from autodnd.engine.game_engine import GameEngine
from autodnd.security.input_sanitizer import InputSanitizer
from autodnd.security.security_agent import SecurityAgent
from autodnd.models.actions import Action, ActionType, TimeCost
from autodnd.models.metadata import Difficulty, GameMetadata, RulesVariant
from autodnd.models.messages import MessageSource, MessageType
from autodnd.models.player import Player
from autodnd.models.state import GameState
from autodnd.models.stats import PlayerStats
from autodnd.models.world import HexCoordinate, HexMap, TimeState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s - %(name)-15s - %(levelname)-8s] %(message)s')

@app.before_request
def log_request_info():
    app.logger.info('Access to: %s', request.url)


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


# Global game storage: game_id -> (GameEngine, GameMasterAgent)
_games: dict[str, tuple[GameEngine, GameMasterAgent]] = {}
_llm_config_manager = LLMConfigManager()
_security_config_manager = SecurityConfigManager()
_security_agent: SecurityAgent | None = None
_input_sanitizer = InputSanitizer()


def _get_security_agent() -> SecurityAgent | None:
    """Get or create security agent."""
    global _security_agent
    if _security_config_manager.config.enabled and _security_config_manager.config.use_llm_validation:
        if _security_agent is None:
            security_config = _security_config_manager.config.get_security_llm_config()
            _security_agent = SecurityAgent(config=security_config)
        return _security_agent
    return None


def _get_game_engine(game_id: Optional[str] = None) -> Optional[GameEngine]:
    """Get game engine by game_id, or return None if not found."""
    if game_id is None:
        return None
    if game_id in _games:
        return _games[game_id][0]
    return None


def _get_game_master(game_id: Optional[str] = None) -> Optional[GameMasterAgent]:
    """Get game master by game_id, or return None if not found."""
    if game_id is None:
        return None
    if game_id in _games:
        return _games[game_id][1]
    return None


def _setup_game_master(engine: GameEngine, custom_prompt: Optional[str] = None) -> GameMasterAgent:
    """Setup game master agent."""
    def state_getter():
        return engine.state

    tools = [
        create_roll_dice_tool(),
        create_get_player_stats_tool(state_getter),
        create_get_inventory_tool(state_getter),
        create_get_map_state_tool(state_getter),
    ]

    llm = _llm_config_manager.get_llm()
    return GameMasterAgent(llm=llm, tools=tools, engine_getter=lambda: engine, system_prompt=custom_prompt)


def _create_initial_game(
    difficulty: Optional[str] = None,
    rules_variant: Optional[str] = None,
    game_master_prompt: Optional[str] = None,
) -> tuple[GameEngine, GameMasterAgent]:
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

    # Parse difficulty and rules_variant
    diff = Difficulty(difficulty) if difficulty else Difficulty.NORMAL
    rules = RulesVariant(rules_variant) if rules_variant else RulesVariant.STANDARD
    
    settings = {}
    if game_master_prompt:
        settings["game_master_prompt"] = game_master_prompt
    
    metadata = GameMetadata(difficulty=diff, rules_variant=rules, settings=settings)

    initial_state = GameState(
        game_id=str(uuid.uuid4()),
        state_version=0,
        created_at=datetime.now(),
        players=[player],
        world_map=HexMap(),
        current_time=TimeState(),
        metadata=metadata,
    )

    engine = GameEngine(initial_state=initial_state)
    game_master = _setup_game_master(engine, game_master_prompt)
    return engine, game_master


@app.route("/api/games", methods=["GET"])
def list_games():
    """List all games."""
    games = []
    for game_id, (engine, _) in _games.items():
        games.append({
            "game_id": game_id,
            "created_at": engine.state.created_at.isoformat(),
            "state_version": engine.state.state_version,
            "players": [{"player_id": p.player_id, "name": p.name} for p in engine.state.players],
            "metadata": engine.state.metadata.model_dump(),
        })
    return jsonify({"games": games})


@app.route("/api/games", methods=["POST"])
def create_game():
    """Create a new game with settings."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    
    difficulty = data.get("difficulty")
    rules_variant = data.get("rules_variant")
    game_master_prompt = data.get("game_master_prompt")
    
    try:
        engine, game_master = _create_initial_game(
            difficulty=difficulty,
            rules_variant=rules_variant,
            game_master_prompt=game_master_prompt,
        )
        game_id = engine.state.game_id
        _games[game_id] = (engine, game_master)
        return jsonify({"success": True, "game_id": game_id})
    except Exception as e:
        app.logger.error(f"Error creating game: {e}", exc_info=True)
        return jsonify({"error": "Failed to create game", "message": str(e)}), 500


@app.route("/api/game/start", methods=["POST"])
def start_game():
    """Initialize new game (deprecated, use /api/games POST instead)."""
    # For backwards compatibility
    return create_game()


@app.route("/api/games/<game_id>/action", methods=["POST"])
def submit_action(game_id: str):
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

    # Security layer: sanitize and validate input
    security_config = _security_config_manager.config
    if security_config.enabled:
        # Sanitize input
        sanitized_text = _input_sanitizer.sanitize(action_text)
        
        # Check if input is safe
        is_safe, error_msg = _input_sanitizer.is_safe(sanitized_text)
        if not is_safe:
            return jsonify({"error": f"Input validation failed: {error_msg}"}), 400
        
        # Use sanitized text
        action_text = sanitized_text
        
        # LLM-based validation (if enabled)
        if security_config.use_llm_validation:
            security_agent = _get_security_agent()
            if security_agent:
                validation_result = security_agent.validate_input(action_text)
                if not validation_result.is_safe:
                    return jsonify({
                        "error": f"Security validation failed: {validation_result.reason}",
                        "risk_level": validation_result.risk_level,
                    }), 400

    engine = _get_game_engine(game_id)
    if not engine:
        return jsonify({"error": "Game not found"}), 404
    
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
    game_master = _get_game_master(game_id)
    if success and game_master:
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
            master_response = game_master.process_action(action_text, recent_messages)
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


@app.route("/api/game/action", methods=["POST"])
def submit_action_legacy():
    """Submit player action (legacy route without game_id)."""
    # For backwards compatibility - use first available game
    if not _games:
        return jsonify({"error": "No games available. Please create a game first."}), 400
    game_id = list(_games.keys())[0]
    return submit_action(game_id)


@app.route("/api/games/<game_id>/state", methods=["GET"])
def get_state(game_id: str):
    """Get current game state."""
    engine = _get_game_engine(game_id)
    if not engine:
        return jsonify({"error": "Game not found"}), 404
    return jsonify({"state": _serialize_state_for_api(engine.state)})


@app.route("/api/game/state", methods=["GET"])
def get_state_legacy():
    """Get current game state (legacy route without game_id)."""
    # For backwards compatibility - use first available game
    if not _games:
        return jsonify({"error": "No games available. Please create a game first."}), 400
    game_id = list(_games.keys())[0]
    return get_state(game_id)


@app.route("/api/games/<game_id>/revert", methods=["POST"])
def revert_to_snapshot(game_id: str):
    """Revert to snapshot."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    snapshot_index = data.get("snapshot_index")
    if snapshot_index is None:
        return jsonify({"error": "snapshot_index is required"}), 400

    engine = _get_game_engine(game_id)
    if not engine:
        return jsonify({"error": "Game not found"}), 404

    success, error_msg = engine.revert_to(snapshot_index)

    if success:
        return jsonify({"success": True, "state": _serialize_state_for_api(engine.state)})
    else:
        return jsonify({"success": False, "error": error_msg}), 400


@app.route("/api/games/<game_id>/history", methods=["GET"])
def list_snapshots(game_id: str):
    """List available snapshots."""
    engine = _get_game_engine(game_id)
    if not engine:
        return jsonify({"error": "Game not found"}), 404
    
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
        # Update game masters with new LLM
        new_llm = _llm_config_manager.get_llm()
        for game_id, (_, game_master) in _games.items():
            game_master.update_llm(new_llm)
        return jsonify({"success": True, "config": _llm_config_manager.config.model_dump()})
    except Exception as e:
        app.logger.error(f"Error updating LLM config: {e}", exc_info=True)
        return jsonify({"error": "Invalid configuration", "message": str(e)}), 400


@app.route("/api/config/security", methods=["GET"])
def get_security_config():
    """Get current security configuration."""
    config = _security_config_manager.config
    return jsonify({"config": config.model_dump()})


@app.route("/api/config/security", methods=["POST"])
def update_security_config():
    """Update security configuration (hot-reload)."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        new_config = SecurityConfig(**data)
        _security_config_manager.update_config(new_config)
        
        # Update security agent if LLM config changed
        global _security_agent
        if new_config.enabled and new_config.use_llm_validation:
            security_llm_config = new_config.get_security_llm_config()
            if _security_agent:
                _security_agent.update_config(security_llm_config)
            else:
                _security_agent = SecurityAgent(config=security_llm_config)
        else:
            _security_agent = None
            
        return jsonify({"success": True, "config": _security_config_manager.config.model_dump()})
    except Exception as e:
        app.logger.error(f"Error updating security config: {e}", exc_info=True)
        return jsonify({"error": "Invalid configuration", "message": str(e)}), 400


@app.route("/")
def index():
    """Serve main page."""
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "Frontend not found"}), 404


@app.route("/game/<game_id>")
def game_view(game_id: str):
    """Serve game view page."""
    # Check if game exists
    if game_id not in _games:
        return jsonify({"error": "Game not found"}), 404
    
    # Serve the game.html page if it exists, otherwise serve index.html
    game_html_path = os.path.join(app.static_folder, "game.html") if app.static_folder else None
    if game_html_path and os.path.exists(game_html_path):
        return send_from_directory(app.static_folder, "game.html")
    elif app.static_folder and os.path.exists(os.path.join(app.static_folder, "index.html")):
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


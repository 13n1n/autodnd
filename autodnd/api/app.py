"""Flask API application."""

import uuid
from datetime import datetime
from typing import Optional

import os
import shutil
import logging
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from autodnd.agents.game_master import GameMasterAgent
from autodnd.agents.orchestrator import AgentOrchestrator
from autodnd.agents.tools import (
    create_get_data_tool,
    create_get_inventory_tool,
    create_get_map_state_tool,
    create_get_player_stats_tool,
    create_roll_dice_tool,
    create_store_data_tool,
    create_take_item_tool,
)
from ..api.llm_config import LLMConfig, LLMConfigManager
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from autodnd.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_API_KEY,
)
from ..api.security_config import SecurityConfig, SecurityConfigManager
from ..engine.game_engine import GameEngine
from ..persistence.state_dumper import StateDumper
from ..security.input_sanitizer import InputSanitizer
from ..security.security_agent import SecurityAgent
from ..models.actions import Action, ActionType, TimeCost
from ..models.metadata import Difficulty, GameMetadata, RulesVariant
from ..models.messages import MessageHistory, MessageSource, MessageType
from ..models.player import Player
from ..models.state import GameState
from ..models.stats import PlayerStats
from ..models.world import HexCoordinate, HexMap, TimeState



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(BASE_DIR, "static")

logging.basicConfig(level=logging.DEBUG, format='[%(name)-19s - %(levelname)5s] %(message)s')

app = Flask("flask.autodnd", static_folder=STATIC_DIR, static_url_path="/static")


@app.before_request
def log_request_info():
    app.logger.info('Access to: %s from %s (%s)',
        request.url,
        request.headers.get('X-Forwarded-For', request.remote_addr),
        request.headers.get('User-Agent'))


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

# State dumper for persisting game state to disk
_dump_directory = os.getenv("AUTODND_DUMP_DIR", "/var/autodnd")
_state_dumper = StateDumper(dump_directory=_dump_directory)


def _get_security_agent() -> SecurityAgent | None:
    """Get or create security agent."""
    global _security_agent
    if _security_config_manager.config.enabled and _security_config_manager.config.use_llm_validation:
        if _security_agent is None:
            security_config = _security_config_manager.config.get_security_llm_config()
            _security_agent = SecurityAgent(config=security_config)
        return _security_agent
    return None


def _dump_game_state(state: GameState) -> None:
    """Dump game state to disk."""
    try:
        # Dump state using new structure: {game_id}/v{version}.json
        _state_dumper.dump_state(state)
    except Exception as e:
        app.logger.error(f"Failed to dump game state {state.game_id}: {e}", exc_info=True)


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


def _create_llm_from_config(config: LLMConfig) -> BaseChatModel:
    """Create an LLM instance from an LLMConfig."""
    kwargs = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
    }

    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens

    try:
        match config.provider:
            case "ollama":
                kwargs["base_url"] = config.base_url or DEFAULT_OLLAMA_BASE_URL
                kwargs["api_key"] = DEFAULT_OLLAMA_API_KEY
                kwargs["num_ctx"] = DEFAULT_LLM_NUM_CTX
                return ChatOllama(**kwargs)
            case "openai":
                kwargs["base_url"] = config.base_url or DEFAULT_OPENAI_BASE_URL
                kwargs["api_key"] = config.api_key or DEFAULT_OPENAI_API_KEY
                return ChatOpenAI(**kwargs)
            case _:
                raise ValueError(f"Invalid provider: {config.provider}")
    except Exception as e:
        app.logger.error(f"Failed to create LLM from config: {e}", exc_info=True)
        # Fallback to default LLM
        return _llm_config_manager.get_llm()


def _setup_game_master(engine: GameEngine, custom_prompt: Optional[str] = None, llm: Optional[BaseChatModel] = None) -> GameMasterAgent:
    """Setup game master agent."""
    def state_getter():
        return engine.state

    tools = [
        create_roll_dice_tool(),
        create_get_player_stats_tool(state_getter),
        create_get_inventory_tool(state_getter),
        create_get_map_state_tool(state_getter),
        create_store_data_tool(state_getter, engine.update_storage),
        create_get_data_tool(state_getter),
        create_take_item_tool(state_getter, engine.take_item),
    ]

    # Use provided LLM or fall back to global config
    if llm is None:
        llm = _llm_config_manager.get_llm()
    return GameMasterAgent(llm=llm, tools=tools, engine_getter=lambda: engine, system_prompt=custom_prompt)


def _load_game_from_state(state: GameState) -> tuple[GameEngine, GameMasterAgent]:
    """
    Load a game from a saved state.

    Args:
        state: GameState to load

    Returns:
        Tuple of (GameEngine, GameMasterAgent)
    """
    # Extract game master prompt from metadata if available
    game_master_prompt = None
    if state.metadata.settings and "game_master_prompt" in state.metadata.settings:
        game_master_prompt = state.metadata.settings["game_master_prompt"]

    # Extract LLM config from metadata if available
    game_llm = None
    if state.metadata.settings and "llm_config" in state.metadata.settings:
        try:
            llm_config_dict = state.metadata.settings["llm_config"]
            llm_config = LLMConfig(**llm_config_dict)
            game_llm = _create_llm_from_config(llm_config)
        except Exception as e:
            app.logger.warning(f"Failed to load LLM config from game state: {e}. Using default LLM.")
            # Fall back to default LLM

    # Create engine with loaded state
    # Title will be computed automatically via computed_field
    engine = GameEngine(initial_state=state)
    
    # Setup game master with game-specific LLM if available
    game_master = _setup_game_master(engine, game_master_prompt, llm=game_llm)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        game_master=game_master,
        npc_agent=None,  # Can be added later if needed
        rag_agent=None,  # Can be added later if needed
        engine_getter=lambda: engine,
    )
    engine.set_orchestrator(orchestrator)
    
    return engine, game_master


def _load_all_games_from_disk() -> None:
    """Load all games from disk at startup."""
    try:
        app.logger.info("Loading games from disk...")
        loaded_games = _state_dumper.load_all_games()
        
        for game_id, state in loaded_games.items():
            try:
                engine, game_master = _load_game_from_state(state)
                _games[game_id] = (engine, game_master)
                app.logger.info(f"Successfully loaded game {game_id} (version {state.state_version})")
            except Exception as e:
                app.logger.error(f"Failed to restore game {game_id}: {e}", exc_info=True)
                # Continue loading other games on exception
        
        app.logger.info(f"Loaded {len(loaded_games)} game(s) from disk")
    except Exception as e:
        app.logger.error(f"Error loading games from disk: {e}", exc_info=True)
        # Don't raise - allow app to start even if loading fails


# Load all games from disk at startup
_load_all_games_from_disk()

def _create_initial_game(
    difficulty: Optional[str] = None,
    rules_variant: Optional[str] = None,
    game_master_prompt: Optional[str] = None,
    llm_config: Optional[LLMConfig] = None,
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
    
    # Use provided LLM config or fall back to global config
    game_llm_config = llm_config if llm_config is not None else _llm_config_manager.config
    
    # Create game-specific LLM instance
    game_llm = _create_llm_from_config(game_llm_config)
    
    # Store LLM config in game settings
    settings = {
        "llm_config": game_llm_config.model_dump(),  # Store LLM config for this game
    }
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
    game_master = _setup_game_master(engine, game_master_prompt, llm=game_llm)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        game_master=game_master,
        npc_agent=None,  # Can be added later if needed
        rag_agent=None,  # Can be added later if needed
        engine_getter=lambda: engine,
    )
    engine.set_orchestrator(orchestrator)
    
    # Generate initial intro message from game master
    try:
        intro_message, tool_messages = game_master.generate_initial_intro(
            difficulty=difficulty or "NORMAL",
            custom_prompt=game_master_prompt,
        )
        
        # Add tool messages to game state first
        for tool_msg in tool_messages:
            engine.add_message(
                content=tool_msg.content,
                source=tool_msg.source,
                message_type=tool_msg.message_type,
                source_id=tool_msg.source_id,
                tool_name=tool_msg.tool_name,
                metadata=tool_msg.metadata,
            )
        
        # Add intro message to game state
        engine.add_message(
            content=intro_message,
            source=MessageSource.MASTER,
            message_type=MessageType.RESPONSE,
            metadata={"is_initial_intro": True},
        )
    except Exception as e:
        app.logger.error(f"Error generating initial intro: {e}", exc_info=True)
        # Add fallback intro message
        engine.add_message(
            content="Welcome, adventurer! Your journey begins here. What would you like to do?",
            source=MessageSource.MASTER,
            message_type=MessageType.RESPONSE,
            metadata={"is_initial_intro": True, "error": str(e)},
        )

    # Dump initial game state
    _dump_game_state(engine.state)
    
    return engine, game_master


@app.route("/api/games", methods=["GET"])
def list_games():
    """List all games."""
    games = []
    for game_id, (engine, *_) in _games.items():
        games.append({
            "game_id": game_id,
            "created_at": engine.state.created_at.isoformat(),
            "state_version": engine.state.state_version,
            "title": engine.state.title,
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
    llm_config_data = data.get("llm_config")

    # Parse LLM config if provided
    llm_config = None
    if llm_config_data:
        try:
            llm_config = LLMConfig(**llm_config_data)
        except Exception as e:
            app.logger.error(f"Invalid LLM config: {e}", exc_info=True)
            return jsonify({"error": "Invalid LLM configuration", "message": str(e)}), 400

    try:
        engine, game_master = _create_initial_game(
            difficulty=difficulty,
            rules_variant=rules_variant,
            game_master_prompt=game_master_prompt,
            llm_config=llm_config,
        )
        game_id = engine.state.game_id
        _games[game_id] = (engine, game_master)
        return jsonify({"success": True, "game_id": game_id})
    except Exception as e:
        app.logger.error(f"Error creating game: {e}", exc_info=True)
        return jsonify({"error": "Failed to create game", "message": str(e)}), 500


@app.route("/api/games/<game_id>", methods=["DELETE"])
def delete_game(game_id: str):
    """Delete a game and all its state files."""
    # Check if game exists in memory
    if game_id not in _games:
        return jsonify({"error": "Game not found"}), 404

    try:
        # Remove game from memory
        del _games[game_id]
        app.logger.info(f"Removed game {game_id} from memory")

        # Delete game directory and all state files
        game_dir = _state_dumper._get_game_directory(game_id)
        if game_dir.exists():
            shutil.rmtree(game_dir)
            app.logger.info(f"Deleted game directory and all state files for {game_id}")
        else:
            app.logger.warning(f"Game directory not found: {game_dir}")

        return jsonify({"success": True, "message": f"Game {game_id} and all its state files have been deleted"})
    except Exception as e:
        app.logger.error(f"Error deleting game {game_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to delete game", "message": str(e)}), 500


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

    # Parse action text to determine action type
    import re
    action_type = ActionType.TALK
    action_parameters = {"text": action_text}
    
    # Check for move commands: "move to hex (q, r)" or "move to (q, r)" or similar patterns
    move_patterns = [
        r"move\s+to\s+hex\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
        r"move\s+to\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
        r"go\s+to\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
    ]
    
    for pattern in move_patterns:
        match = re.search(pattern, action_text, re.IGNORECASE)
        if match:
            try:
                q = int(match.group(1))
                r = int(match.group(2))
                action_type = ActionType.MOVE
                action_parameters = {
                    "target_coordinate": {"q": q, "r": r}
                }
                break
            except (ValueError, IndexError):
                pass

    # Create action
    action = Action(
        action_id=str(uuid.uuid4()),
        action_type=action_type,
        parameters=action_parameters,
        time_cost=TimeCost.NO_TIME if action_type == ActionType.TALK else TimeCost.HALF_DAY,
        player_id=player_id,
    )

    # Apply action
    new_state, success, error_msg = engine.apply_action(action)

    _dump_game_state(new_state)

    return jsonify(
        {
            "success": success,
            "error": error_msg if not success else None,
            "state": _serialize_state_for_api(new_state)
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
        # Dump reverted game state to disk
        _dump_game_state(engine.state)
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


@app.route("/api/games/<game_id>/recover", methods=["POST"])
def recover_to_message(game_id: str):
    """Recover game state to a specific message, removing all states after it."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    sequence_number = data.get("sequence_number")
    if sequence_number is None:
        return jsonify({"error": "sequence_number is required"}), 400

    # Check if game exists
    if game_id not in _games:
        return jsonify({"error": "Game not found"}), 404

    try:
        # Find the state version that contains this message
        target_version = _state_dumper.find_state_with_message(game_id, sequence_number)
        if target_version is None:
            return jsonify({"error": f"Message with sequence_number {sequence_number} not found in any saved state"}), 404

        app.logger.info(f"Recovering game {game_id} to state version {target_version} (message sequence {sequence_number})")

        # Load the target state from disk
        target_state = _state_dumper.load_state(game_id, target_version)
        if not target_state:
            return jsonify({"error": f"Failed to load state version {target_version}"}), 500

        # Verify the message exists in this state
        message_found = False
        for msg in target_state.message_history.messages:
            if msg.sequence_number == sequence_number:
                message_found = True
                break

        if not message_found:
            return jsonify({"error": f"Message with sequence_number {sequence_number} not found in state version {target_version}"}), 404

        # Reload the game from the target state
        engine, game_master = _load_game_from_state(target_state)
        _games[game_id] = (engine, game_master)

        # Delete all state files with version numbers greater than target_version
        deleted_count = _state_dumper.delete_versions_after(game_id, target_version)
        app.logger.info(f"Deleted {deleted_count} state file(s) after version {target_version}")

        # Dump the recovered state (this will overwrite/create v{target_version}.json if needed)
        _dump_game_state(engine.state)

        return jsonify({
            "success": True,
            "state": _serialize_state_for_api(engine.state),
            "recovered_to_version": target_version,
            "deleted_files": deleted_count,
        })
    except Exception as e:
        app.logger.error(f"Error recovering game {game_id} to message {sequence_number}: {e}", exc_info=True)
        return jsonify({"error": "Failed to recover game state", "message": str(e)}), 500


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
    # Serialize world_map cells
    world_map_serialized = {
        "cells": {
            f"({q},{r})": {
                "coordinates": {"q": cell.coordinates.q, "r": cell.coordinates.r},
                "terrain": cell.terrain.value,
                "contents": cell.contents,
                "discovered": cell.discovered,
                "description": cell.description,
            }
            for (q, r), cell in state.world_map.cells.items()
        }
    }
    
    serialized = {
        "game_id": state.game_id,
        "state_version": state.state_version,
        "created_at": state.created_at.isoformat(),
        "title": state.title,
        "players": [player.model_dump() for player in state.players],
        "current_time": state.current_time.model_dump(),
        "world_map": world_map_serialized,
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
    return serialized


if __name__ == "__main__":
    app.run(debug=True, port=5000)


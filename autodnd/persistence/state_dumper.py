"""State dumper for saving game state to disk."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from autodnd.models.state import GameState

logger = logging.getLogger(__name__.split(".")[-1])


class StateDumper:
    """Dumps game state to disk and loads them back."""

    def __init__(self, dump_directory: str = "/var/autodnd"):
        """
        Initialize state dumper.

        Args:
            dump_directory: Directory where game states will be saved
        """
        self.dump_directory = Path(dump_directory)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure dump directory exists, create if it doesn't."""
        try:
            self.dump_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"State dump directory ready: {self.dump_directory}")
        except PermissionError:
            logger.error(f"Permission denied creating directory: {self.dump_directory}")
            raise
        except OSError as e:
            logger.error(f"Error creating directory {self.dump_directory}: {e}")
            raise

    def _get_game_directory(self, game_id: str) -> Path:
        """Get directory for a specific game."""
        return self.dump_directory / game_id

    def _ensure_game_directory_exists(self, game_id: str) -> None:
        """Ensure game directory exists, create if it doesn't."""
        game_dir = self._get_game_directory(game_id)
        try:
            game_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating game directory {game_dir}: {e}", exc_info=True)
            raise

    def dump_state(self, state: GameState) -> str:
        """
        Dump game state to disk using structure: {game_id}/v{version}.json

        Args:
            state: GameState to dump

        Returns:
            Path to the dumped file
        """
        # Ensure game directory exists
        self._ensure_game_directory_exists(state.game_id)

        # Create filename with version suffix
        filename = f"v{state.state_version}.json"
        game_dir = self._get_game_directory(state.game_id)
        file_path = game_dir / filename

        try:
            # Serialize state to JSON
            state_json = state.model_dump_json(indent=2)

            # Write to file atomically
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(state_json)

            # Atomic rename
            temp_path.replace(file_path)

            logger.debug(f"Dumped game state {state.game_id} v{state.state_version} to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error dumping state {state.game_id} v{state.state_version} to {file_path}: {e}", exc_info=True)
            raise

    def load_state(self, game_id: str, version: Optional[int] = None) -> Optional[GameState]:
        """
        Load game state from disk.

        Args:
            game_id: Game ID to load
            version: Optional version number. If None, loads the latest version.

        Returns:
            GameState if found, None otherwise
        """
        game_dir = self._get_game_directory(game_id)
        
        if not game_dir.exists():
            logger.warning(f"Game directory not found: {game_dir}")
            return None

        if version is not None:
            # Load specific version
            file_path = game_dir / f"v{version}.json"
            if not file_path.exists():
                logger.warning(f"State file not found: {file_path}")
                return None
            return self._load_state_from_file(file_path)

        # Find latest version
        latest_file = self._find_latest_version_file(game_dir)
        if latest_file is None:
            logger.warning(f"No state files found in {game_dir}")
            return None

        return self._load_state_from_file(latest_file)

    def _find_latest_version_file(self, game_dir: Path) -> Optional[Path]:
        """
        Find the latest version file in a game directory.

        Args:
            game_dir: Game directory to search

        Returns:
            Path to latest version file, or None if not found
        """
        pattern = re.compile(r"^v(\d+)\.json$")
        latest_version = -1
        latest_file = None

        try:
            for file_path in game_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                match = pattern.match(file_path.name)
                if match:
                    version = int(match.group(1))
                    if version > latest_version:
                        latest_version = version
                        latest_file = file_path
        except Exception as e:
            logger.error(f"Error scanning game directory {game_dir}: {e}", exc_info=True)
            return None

        return latest_file

    def _load_state_from_file(self, file_path: Path) -> Optional[GameState]:
        """
        Load state from a specific file.

        Args:
            file_path: Path to state file

        Returns:
            GameState if loaded successfully, None otherwise
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            state = GameState.model_validate(state_data)
            logger.debug(f"Loaded game state from {file_path}")
            return state
        except Exception as e:
            logger.error(f"Error loading state from {file_path}: {e}", exc_info=True)
            return None

    def list_games(self) -> list[str]:
        """
        List all game IDs that have saved states.

        Returns:
            List of game IDs
        """
        game_ids = []
        
        if not self.dump_directory.exists():
            return game_ids

        try:
            for item in self.dump_directory.iterdir():
                if item.is_dir():
                    # Check if directory contains any state files
                    if any(item.glob("v*.json")):
                        game_ids.append(item.name)
        except Exception as e:
            logger.error(f"Error listing games in {self.dump_directory}: {e}", exc_info=True)

        return game_ids

    def load_all_games(self) -> dict[str, GameState]:
        """
        Load all games from disk (latest state for each game).

        Returns:
            Dictionary mapping game_id to GameState
        """
        games = {}
        game_ids = self.list_games()

        for game_id in game_ids:
            try:
                state = self.load_state(game_id)
                if state:
                    games[game_id] = state
                    logger.info(f"Loaded game {game_id} (version {state.state_version})")
                else:
                    logger.warning(f"Failed to load game {game_id}: state is None")
            except Exception as e:
                logger.error(f"Error loading game {game_id}: {e}", exc_info=True)
                # Continue loading other games on exception

        return games


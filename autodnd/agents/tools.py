"""LangChain tools for game agents."""

import logging
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from autodnd.engine.dice import DiceRoller
from autodnd.models.state import GameState

logger = logging.getLogger(__name__)


class RollDiceInput(BaseModel):
    """Input for roll dice tool."""

    dice_type: int = Field(description="Type of dice (e.g., 20 for d20, 6 for d6)")
    modifier: int = Field(default=0, description="Modifier to add to result")
    count: int = Field(default=1, description="Number of dice to roll")
    reason: Optional[str] = Field(default=None, description="Reason for rolling the dice")


class GetPlayerStatsInput(BaseModel):
    """Input for get player stats tool."""

    player_id: Optional[str] = Field(default=None, description="Player ID to get stats for. If not provided, returns stats for any/default player.")


class GetInventoryInput(BaseModel):
    """Input for get inventory tool."""

    player_id: Optional[str] = Field(default=None, description="Player ID to get inventory for. If not provided, returns inventory for any/default player.")


class GetMapStateInput(BaseModel):
    """Input for get map state tool."""

    coordinate_q: Optional[int] = Field(default=None, description="Q coordinate (optional)")
    coordinate_r: Optional[int] = Field(default=None, description="R coordinate (optional)")


class GetNPCInfoInput(BaseModel):
    """Input for get NPC info tool."""

    npc_id: str = Field(description="NPC ID to get information for")
    coordinate_q: Optional[int] = Field(default=None, description="Q coordinate where NPC is located")
    coordinate_r: Optional[int] = Field(default=None, description="R coordinate where NPC is located")


class StoreDataInput(BaseModel):
    """Input for store data tool."""

    key: str = Field(description="Key to store the value under")
    value: str = Field(description="Value to store")
    reason: Optional[str] = Field(default=None, description="Reason for storing this data")


class GetDataInput(BaseModel):
    """Input for get data tool."""

    key: Optional[str] = Field(default=None, description="Key to retrieve the value for. If not provided, returns a list of all available keys.")


class TakeItemInput(BaseModel):
    """Input for take item tool."""

    item_id: str = Field(description="Item ID to take (must exist in current cell contents or be provided as new item)")
    item_name: Optional[str] = Field(default=None, description="Item name (required if creating new item)")
    item_description: Optional[str] = Field(default=None, description="Item description (required if creating new item)")
    tags: Optional[list[str]] = Field(default=None, description="Item tags (e.g., ['weapon', 'heavy', 'quest', 'potion', 'book/scroll' and etc])")
    slot_size: int = Field(default=1, ge=1, le=2, description="Slot size (1 for normal, 2 for heavy)")
    stat_modifiers: Optional[dict[str, int]] = Field(default=None, description="Stat modifiers when equipped (e.g., {'strength': 2} or {'bag_size': 12})")
    player_id: Optional[str] = Field(default=None, description="Player ID to take item for. If not provided, uses first player.")
    wear: bool = Field(default=False, description="Whether to equip/wear the item immediately after taking it")


def create_roll_dice_tool() -> StructuredTool:
    """Create roll dice tool."""

    def roll_dice(dice_type: int, modifier: int = 0, count: int = 1, reason: Optional[str] = None) -> dict:
        """Roll DnD dice with optional modifier and count."""
        logger.info(f"Tool called: roll_dice(dice_type={dice_type}, modifier={modifier}, count={count}, reason={reason})")
        result = DiceRoller.roll(dice_type, modifier, count)
        if reason:
            logger.debug(f"roll_dice result: {result} (reason: {reason})")
        else:
            logger.debug(f"roll_dice result: {result}")
        return result

    return StructuredTool.from_function(
        func=roll_dice,
        name="roll_dice",
        description="Roll DnD dice. Use dice_type=20 for d20, dice_type=6 for d6, etc.",
        args_schema=RollDiceInput,
    )


def create_get_player_stats_tool(state_getter) -> StructuredTool:
    """Create get player stats tool."""

    def get_player_stats(player_id: Optional[str] = None) -> dict:
        """Get player stats from current game state."""
        logger.info(f"Tool called: get_player_stats(player_id={player_id})")
        state: GameState = state_getter()
        player = next((p for p in state.players if player_id is None or p.player_id == player_id), None)
        if not player:
            logger.warning(f"Player {player_id} not found")
            return {"error": f"Player {player_id} not found"}

        result = {
            "player_id": player.player_id,
            "name": player.name,
            "level": player.level,
            "experience": player.experience,
            "base_stats": player.base_stats.model_dump(),
            "current_stats": player.current_stats.model_dump(),
            "is_alive": player.is_alive,
            "status_conditions": player.status_conditions,
            "inventory_size": len(player.inventory.all_items),
        }
        logger.debug(f"get_player_stats result for {player_id}: level={player.level}, stats={player.current_stats.model_dump()}")
        return result

    return StructuredTool.from_function(
        func=get_player_stats,
        name="get_player_stats",
        description="Get current stats for a player by player_id",
        args_schema=GetPlayerStatsInput,
    )


def create_get_inventory_tool(state_getter) -> StructuredTool:
    """Create get inventory tool."""

    def get_inventory(player_id: Optional[str] = None) -> dict:
        """Get player inventory from current game state."""
        logger.info(f"Tool called: get_inventory(player_id={player_id})")
        state: GameState = state_getter()
        player = next((p for p in state.players if player_id is None or p.player_id == player_id), None)
        if not player:
            logger.warning(f"Player {player_id} not found")
            return {"error": f"Player {player_id} not found"}

        result = {
            "player_id": player.player_id,
            "inventory": player.inventory.model_dump(),
        }
        logger.debug(f"get_inventory result for {player_id}: {len(player.inventory.bags)} bags")
        return result

    return StructuredTool.from_function(
        func=get_inventory,
        name="get_inventory",
        description="Get inventory for a player by player_id",
        args_schema=GetInventoryInput,
    )


def create_get_map_state_tool(state_getter) -> StructuredTool:
    """Create get map state tool."""

    from autodnd.models.world import HexCoordinate, TerrainType

    def get_map_state(coordinate_q: Optional[int] = None, coordinate_r: Optional[int] = None) -> dict:
        """
        Clean get_map_state:
        - If no (coordinate_q, coordinate_r), default to player's position
        - Always returns nearest cities and dungeons in radius 6 of requested position
        - Always returns player's position
        """
        logger.info(f"Tool called: get_map_state(coordinate_q={coordinate_q}, coordinate_r={coordinate_r})")
        state: GameState = state_getter()
        if not state.players or not hasattr(state.players[0], "position"):
            logger.warning("No players or player position in current state")
            return {"error": "No player position found"}

        player = state.players[0]
        player_position = {"q": player.position.q, "r": player.position.r}

        # Substitute missing arguments with player's current position
        if coordinate_q is None or coordinate_r is None:
            coordinate_q = player.position.q
            coordinate_r = player.position.r

        coord = HexCoordinate(q=coordinate_q, r=coordinate_r)
        cell = state.world_map.get_cell(coord)

        result = {}
        if cell:
            result["cell"] = cell.model_dump()
        else:
            logger.warning(f"Cell at ({coordinate_q}, {coordinate_r}) not found")
            result["error"] = f"Cell at ({coordinate_q}, {coordinate_r}) not found"

        # Always include player's current position
        result["player_position"] = player_position

        # Find cities and dungeons in radius 6 of the requested position
        def hex_distance(a: HexCoordinate, b: HexCoordinate) -> int:
            # Cube coordinates: (q, r, s)
            aq, ar, as_ = a.q, a.r, -a.q - a.r
            bq, br, bs = b.q, b.r, -b.q - b.r
            return max(abs(aq - bq), abs(ar - br), abs(as_ - bs))

        cities_and_dungeons = []
        RADIUS = 6
        for (q, r), c in state.world_map.cells.items():
            check_coord = HexCoordinate(q=q, r=r)
            if hex_distance(coord, check_coord) <= RADIUS:
                if getattr(c, "terrain", None) in [TerrainType.CITY, TerrainType.DUNGEON]:
                    # For clarity, include coordinates, terrain, and description only
                    cities_and_dungeons.append({
                        "q": c.coordinates.q,
                        "r": c.coordinates.r,
                        "terrain": c.terrain.value,
                        "description": c.description,
                    })
        result["nearby_cities_and_dungeons"] = cities_and_dungeons

        logger.debug(
            f"get_map_state: pos=({coordinate_q},{coordinate_r}), "
            f"cell={'present' if cell else 'none'}, "
            f"nearby_cities_and_dungeons={len(cities_and_dungeons)}, "
            f"player_position={player_position}"
        )
        return result

    return StructuredTool.from_function(
        func=get_map_state,
        name="get_map_state",
        description="Get map state at a specific coordinates (coordinate_q, coordinate_r). If omitted, defaults to player's position. Always returns the player's current position and all cities and dungeons within 6 hexes of the requested position.",
        args_schema=GetMapStateInput,
    )



def create_get_npc_info_tool(state_getter) -> StructuredTool:
    """Create get NPC info tool."""

    def get_npc_info(npc_id: str, coordinate_q: Optional[int] = None, coordinate_r: Optional[int] = None) -> dict:
        """Get NPC information from current game state."""
        logger.info(f"Tool called: get_npc_info(npc_id={npc_id}, coordinate_q={coordinate_q}, coordinate_r={coordinate_r})")
        state: GameState = state_getter()

        # Search for NPC in map cells
        npc_found = False
        npc_location = None

        if coordinate_q is not None and coordinate_r is not None:
            from autodnd.models.world import HexCoordinate

            coord = HexCoordinate(q=coordinate_q, r=coordinate_r)
            cell = state.world_map.get_cell(coord)
            if cell and npc_id in cell.contents:
                npc_found = True
                npc_location = {"q": coordinate_q, "r": coordinate_r}
        else:
            # Search all cells
            for (q, r), cell in state.world_map.cells.items():
                if npc_id in cell.contents:
                    npc_found = True
                    npc_location = {"q": q, "r": r}
                    break

        if not npc_found:
            logger.warning(f"NPC {npc_id} not found in current game state")
            return {"error": f"NPC {npc_id} not found in current game state"}

        result = {
            "npc_id": npc_id,
            "location": npc_location,
            "found": True,
        }
        logger.debug(f"get_npc_info result: NPC {npc_id} found at {npc_location}")
        return result

    return StructuredTool.from_function(
        func=get_npc_info,
        name="get_npc_info",
        description="Get information about an NPC by npc_id. Optionally provide coordinates to search specific cell",
        args_schema=GetNPCInfoInput,
    )


def create_store_data_tool(state_getter, engine_updater) -> StructuredTool:
    """Create store data tool for key-value storage."""

    def store_data(key: str, value: str, reason: Optional[str] = None) -> dict:
        """Store a key-value pair in persistent storage."""
        logger.info(f"Tool called: store_data(key={key}, value_length={len(value)}, reason={reason})")
        state = state_getter()
        engine_updater(key, value)
        result = {"success": True, "key": key, "message": f"Stored value for key '{key}'"}
        if reason:
            logger.debug(f"store_data: Stored data for key '{key}' (reason: {reason})")
        else:
            logger.debug(f"store_data: Stored data for key '{key}'")
        return result

    return StructuredTool.from_function(
        func=store_data,
        name="store_data",
        description="Store a key-value pair in persistent storage. Useful for storing hidden objectives, game notes, or other data that should persist across turns.",
        args_schema=StoreDataInput,
    )


def create_get_data_tool(state_getter) -> StructuredTool:
    """Create get data tool for key-value retrieval."""

    def get_data(key: Optional[str] = None) -> dict:
        """Retrieve a value by key from persistent storage. If key is not provided, returns a list of all available keys."""
        logger.info(f"Tool called: get_data(key={key})")
        state = state_getter()
        if key is None:
            keys = list[str](state.storage.keys())
            result = {"success": True, "keys": keys, "count": len(keys)}
            logger.debug(f"get_data: Returning list of {len(keys)} available keys")
            return result
        if key in state.storage:
            result = {"success": True, "key": key, "value": state.storage[key]}
            logger.debug(f"get_data: Found data for key '{key}', value_length={len(str(result['value']))}")
            return result
        logger.warning(f"Key '{key}' not found in storage")
        return {"success": False, "key": key, "error": f"Key '{key}' not found in storage"}

    return StructuredTool.from_function(
        func=get_data,
        name="get_data",
        description="Retrieve a value by key from persistent storage. If key is not provided, returns a list of all available keys. Useful for retrieving hidden objectives, game notes, or other stored data.",
        args_schema=GetDataInput,
    )


def create_take_item_tool(state_getter, engine_updater) -> StructuredTool:
    """Create take item tool."""

    def take_item(
        item_id: str,
        item_name: Optional[str] = None,
        item_description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        slot_size: int = 1,
        stat_modifiers: Optional[dict[str, int]] = None,
        player_id: Optional[str] = None,
        wear: bool = False,
    ) -> dict:
        """Take an item from a location and add it to player inventory."""
        logger.info(
            f"Tool called: take_item(item_id={item_id}, player_id={player_id}, wear={wear}"
        )
        state: GameState = state_getter()
        
        # Get player (same pattern as other tools)
        player = next((p for p in state.players if player_id is None or p.player_id == player_id), None)
        if not player:
            logger.warning(f"Player {player_id} not found")
            return {"error": f"Player {player_id} not found"}

        # If item not in cell and we don't have item details, return error
        if (not item_name or not item_description):
            logger.warning(f"item details not provided")
            return {"error": f"Provide item_name and item_description to create new item."}

        # Update state through engine_updater
        result = engine_updater(
            item_id=item_id,
            item_name=item_name,
            item_description=item_description,
            tags=tags or [],
            slot_size=slot_size,
            stat_modifiers=stat_modifiers or {},
            player_id=player.player_id,
            wear=wear,
        )
        
        if result.get("error"):
            logger.warning(f"take_item failed: {result.get('error')}")
            return result
        
        logger.debug(f"take_item result: {result}")
        return result

    return StructuredTool.from_function(
        func=take_item,
        name="take_item",
        description="Take an item from a location (cell) and add it to player inventory. Stores items to first available slot in bag. If taking a bag, adds it if player has < 7 bags, or replaces first bag with lesser storage capacity. Can optionally equip the item if wear=true.",
        args_schema=TakeItemInput,
    )


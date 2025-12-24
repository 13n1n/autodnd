"""LangChain tools for game agents."""

import logging
from typing import Optional

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

    player_id: str = Field(description="Player ID to get stats for")


class GetInventoryInput(BaseModel):
    """Input for get inventory tool."""

    player_id: str = Field(description="Player ID to get inventory for")


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
    reason: Optional[str] = Field(default=None, description="Reason for rolling the dice")


class GetDataInput(BaseModel):
    """Input for get data tool."""

    key: str = Field(description="Key to retrieve the value for")


def create_roll_dice_tool() -> StructuredTool:
    """Create roll dice tool."""

    def roll_dice(dice_type: int, modifier: int = 0, count: int = 1) -> dict:
        """Roll DnD dice with optional modifier and count."""
        logger.info(f"Tool called: roll_dice(dice_type={dice_type}, modifier={modifier}, count={count})")
        result = DiceRoller.roll(dice_type, modifier, count)
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

    def get_player_stats(player_id: str) -> dict:
        """Get player stats from current game state."""
        logger.info(f"Tool called: get_player_stats(player_id={player_id})")
        state: GameState = state_getter()
        player = next((p for p in state.players if p.player_id == player_id), None)
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

    def get_inventory(player_id: str) -> dict:
        """Get player inventory from current game state."""
        logger.info(f"Tool called: get_inventory(player_id={player_id})")
        state: GameState = state_getter()
        player = next((p for p in state.players if p.player_id == player_id), None)
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

    def get_map_state(coordinate_q: Optional[int] = None, coordinate_r: Optional[int] = None) -> dict:
        """Get map state from current game state."""
        logger.info(f"Tool called: get_map_state(coordinate_q={coordinate_q}, coordinate_r={coordinate_r})")
        state: GameState = state_getter()

        if coordinate_q is not None and coordinate_r is not None:
            from autodnd.models.world import HexCoordinate

            coord = HexCoordinate(q=coordinate_q, r=coordinate_r)
            cell = state.world_map.get_cell(coord)
            if cell:
                logger.debug(f"get_map_state: Found cell at ({coordinate_q}, {coordinate_r})")
                return {"cell": cell.model_dump()}
            logger.warning(f"Cell at ({coordinate_q}, {coordinate_r}) not found")
            return {"error": f"Cell at ({coordinate_q}, {coordinate_r}) not found"}

        # Return overall map info
        result = {
            "total_cells": len(state.world_map.cells),
            "cells": [cell.model_dump() for cell in state.world_map.cells.values()],
        }
        logger.debug(f"get_map_state: Returning {result['total_cells']} cells")
        return result

    return StructuredTool.from_function(
        func=get_map_state,
        name="get_map_state",
        description="Get map state. Provide coordinate_q and coordinate_r for specific cell, or omit for all cells",
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

    def store_data(key: str, value: str) -> dict:
        """Store a key-value pair in persistent storage."""
        logger.info(f"Tool called: store_data(key={key}, value_length={len(value)})")
        state = state_getter()
        engine_updater(key, value)
        result = {"success": True, "key": key, "message": f"Stored value for key '{key}'"}
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

    def get_data(key: str) -> dict:
        """Retrieve a value by key from persistent storage."""
        logger.info(f"Tool called: get_data(key={key})")
        state = state_getter()
        if key in state.storage:
            result = {"success": True, "key": key, "value": state.storage[key]}
            logger.debug(f"get_data: Found data for key '{key}', value_length={len(str(result['value']))}")
            return result
        logger.warning(f"Key '{key}' not found in storage")
        return {"success": False, "key": key, "error": f"Key '{key}' not found in storage"}

    return StructuredTool.from_function(
        func=get_data,
        name="get_data",
        description="Retrieve a value by key from persistent storage. Useful for retrieving hidden objectives, game notes, or other stored data.",
        args_schema=GetDataInput,
    )


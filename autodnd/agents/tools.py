"""LangChain tools for game agents."""

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from autodnd.engine.dice import DiceRoller
from autodnd.models.state import GameState


class RollDiceInput(BaseModel):
    """Input for roll dice tool."""

    dice_type: int = Field(description="Type of dice (e.g., 20 for d20, 6 for d6)")
    modifier: int = Field(default=0, description="Modifier to add to result")
    count: int = Field(default=1, description="Number of dice to roll")


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


def create_roll_dice_tool() -> StructuredTool:
    """Create roll dice tool."""

    def roll_dice(dice_type: int, modifier: int = 0, count: int = 1) -> dict:
        """Roll DnD dice with optional modifier and count."""
        return DiceRoller.roll(dice_type, modifier, count)

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
        state: GameState = state_getter()
        player = next((p for p in state.players if p.player_id == player_id), None)
        if not player:
            return {"error": f"Player {player_id} not found"}

        return {
            "player_id": player.player_id,
            "name": player.name,
            "level": player.level,
            "experience": player.experience,
            "base_stats": player.base_stats.model_dump(),
            "current_stats": player.current_stats.model_dump(),
            "is_alive": player.is_alive,
            "status_conditions": player.status_conditions,
        }

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
        state: GameState = state_getter()
        player = next((p for p in state.players if p.player_id == player_id), None)
        if not player:
            return {"error": f"Player {player_id} not found"}

        return {
            "player_id": player.player_id,
            "inventory": player.inventory.model_dump(),
        }

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
        state: GameState = state_getter()

        if coordinate_q is not None and coordinate_r is not None:
            from autodnd.models.world import HexCoordinate

            coord = HexCoordinate(q=coordinate_q, r=coordinate_r)
            cell = state.world_map.get_cell(coord)
            if cell:
                return {"cell": cell.model_dump()}
            return {"error": f"Cell at ({coordinate_q}, {coordinate_r}) not found"}

        # Return overall map info
        return {
            "total_cells": len(state.world_map.cells),
            "cells": [cell.model_dump() for cell in state.world_map.cells.values()],
        }

    return StructuredTool.from_function(
        func=get_map_state,
        name="get_map_state",
        description="Get map state. Provide coordinate_q and coordinate_r for specific cell, or omit for all cells",
        args_schema=GetMapStateInput,
    )


"""Stat calculation system."""

from autodnd.models.items import Item, PlayerInventory
from autodnd.models.player import Player
from autodnd.models.stats import PlayerStats


class StatCalculator:
    """Computes effective stats (base + items + buffs/debuffs)."""

    @staticmethod
    def calculate_current_stats(player: Player, active_effects: list) -> PlayerStats:
        """
        Calculate current effective stats for a player.

        Args:
            player: Player to calculate stats for
            active_effects: List of active effects (buffs/debuffs)

        Returns:
            New PlayerStats with effective values
        """
        # Start with base stats
        effective_stats = {
            "health": player.base_stats.health,
            "max_health": player.base_stats.max_health,
            "strength": player.base_stats.strength,
            "dexterity": player.base_stats.dexterity,
            "intelligence": player.base_stats.intelligence,
            "charisma": player.base_stats.charisma,
        }

        # Add stat modifiers from equipped items
        equipped_items = StatCalculator._get_equipped_items(player.inventory)
        for item in equipped_items:
            for stat_name, modifier in item.stat_modifiers.items():
                if stat_name in effective_stats:
                    effective_stats[stat_name] += modifier

        # Add stat modifiers from active effects
        for effect in active_effects:
            for stat_name, modifier in effect.stat_modifications.items():
                if stat_name in effective_stats:
                    effective_stats[stat_name] += modifier

        # Ensure stats don't go below 0
        for stat_name in effective_stats:
            if stat_name not in ["health", "max_health"]:
                effective_stats[stat_name] = max(0, effective_stats[stat_name])

        # Health can't exceed max_health
        effective_stats["health"] = min(
            effective_stats["health"], effective_stats["max_health"]
        )
        effective_stats["health"] = max(0, effective_stats["health"])

        return PlayerStats(**effective_stats)

    @staticmethod
    def _get_equipped_items(inventory: PlayerInventory) -> list[Item]:
        """Get all equipped items from inventory."""
        equipped = []
        if inventory.primary_weapon:
            equipped.append(inventory.primary_weapon)
        if inventory.secondary_weapon:
            equipped.append(inventory.secondary_weapon)
        if inventory.cloth:
            equipped.append(inventory.cloth)
        if inventory.hat:
            equipped.append(inventory.hat)
        if inventory.pants:
            equipped.append(inventory.pants)
        if inventory.large_item:
            equipped.append(inventory.large_item)
        return equipped


"""Dice rolling system for DnD mechanics."""

import random
from typing import Optional


class DiceRoller:
    """Handles DnD dice mechanics."""

    @staticmethod
    def roll(dice_type: int, modifier: int = 0, count: int = 1) -> dict[str, int]:
        """
        Roll dice with modifier.

        Args:
            dice_type: Type of dice (e.g., 20 for d20, 6 for d6)
            modifier: Modifier to add to result
            count: Number of dice to roll

        Returns:
            Dictionary with 'total', 'rolls', and 'modifier' keys
        """
        rolls = [random.randint(1, dice_type) for _ in range(count)]
        total = sum(rolls) + modifier
        return {
            "total": total,
            "rolls": rolls,
            "modifier": modifier,
            "dice_type": dice_type,
            "count": count,
        }

    @staticmethod
    def roll_d20(modifier: int = 0) -> dict[str, int]:
        """Roll a d20 with modifier."""
        return DiceRoller.roll(20, modifier, 1)

    @staticmethod
    def roll_d6(modifier: int = 0, count: int = 1) -> dict[str, int]:
        """Roll d6 dice with modifier."""
        return DiceRoller.roll(6, modifier, count)

    @staticmethod
    def roll_damage(dice_type: int, count: int, modifier: int = 0) -> dict[str, int]:
        """Roll damage dice."""
        return DiceRoller.roll(dice_type, modifier, count)

    @staticmethod
    def roll_check(stat_value: int, difficulty_class: Optional[int] = None) -> dict[str, any]:
        """
        Roll a skill check.

        Args:
            stat_value: Stat value to use as modifier
            difficulty_class: Optional DC to check against

        Returns:
            Dictionary with roll result and success status if DC provided
        """
        roll_result = DiceRoller.roll_d20(modifier=stat_value)
        result = {
            "roll": roll_result["rolls"][0],
            "modifier": stat_value,
            "total": roll_result["total"],
        }
        if difficulty_class is not None:
            result["success"] = roll_result["total"] >= difficulty_class
            result["dc"] = difficulty_class
        return result


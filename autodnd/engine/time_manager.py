"""Time management for game state."""

from autodnd.models.actions import TimeCost
from autodnd.models.world import TimeState


class TimeManager:
    """Manages game time progression."""

    @staticmethod
    def advance_time(current_time: TimeState, time_cost: TimeCost) -> TimeState:
        """
        Advance game time based on action time cost.

        Args:
            current_time: Current time state
            time_cost: Time cost of the action

        Returns:
            New time state with advanced time
        """
        if time_cost == TimeCost.NO_TIME:
            return current_time

        half_days = 0
        if time_cost == TimeCost.HALF_DAY:
            half_days = 1
        elif time_cost == TimeCost.WHOLE_DAY:
            half_days = 2

        new_total = current_time.total_time_elapsed + half_days
        new_increment = current_time.half_day_increment + half_days
        new_day = current_time.current_day

        # Handle day rollover
        if new_increment >= 2:
            new_day += new_increment // 2
            new_increment = new_increment % 2

        return TimeState(
            current_day=new_day,
            half_day_increment=new_increment,
            total_time_elapsed=new_total,
        )


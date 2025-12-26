"""State history management system."""

from datetime import datetime
from typing import Optional

from autodnd.models.state import GameState, StateSnapshot


class StateHistory:
    """Manages state snapshots and history."""

    def __init__(self) -> None:
        """Initialize empty state history."""
        self._snapshots: list[StateSnapshot] = []
        self._next_index = 0

    def create_snapshot(
        self, state: GameState, metadata: Optional[dict] = None
    ) -> StateSnapshot:
        """
        Create a snapshot of the current state.

        Args:
            state: Current game state to snapshot
            metadata: Optional metadata for the snapshot

        Returns:
            Created StateSnapshot
        """
        snapshot = StateSnapshot(
            index=self._next_index,
            timestamp=datetime.now(),
            state=state,
            metadata=metadata or {},
        )
        self._snapshots.append(snapshot)
        self._next_index += 1
        return snapshot

    def get_snapshot(self, index: int) -> Optional[StateSnapshot]:
        """
        Get snapshot by index.

        Args:
            index: Snapshot index

        Returns:
            StateSnapshot if found, None otherwise
        """
        if 0 <= index < len(self._snapshots):
            return self._snapshots[index]
        return None

    def list_snapshots(self) -> list[StateSnapshot]:
        """List all snapshots."""
        return self._snapshots.copy()

    def truncate_after(self, index: int) -> None:
        """
        Truncate history after specified index.

        Args:
            index: Last snapshot to keep
        """
        if 0 <= index < len(self._snapshots):
            self._snapshots = self._snapshots[: index + 1]
            self._next_index = index + 1

    def get_latest(self) -> Optional[StateSnapshot]:
        """Get the latest snapshot."""
        if self._snapshots:
            return self._snapshots[-1]
        return None


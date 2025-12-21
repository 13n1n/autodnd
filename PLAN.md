# Design Plan: AutoDnD Game Engine

## Brief

### Architecture Overview

**Core Components:**
1. **Game Engine** - State machine managing game logic, rules, and progression
2. **LLM Agent System** - Multi-agent architecture using LangChain for different roles
3. **State Management** - Immutable state with history/revert capability
4. **Security Layer** - Prompt injection prevention and input sanitization
5. **Frontend** - HTML/Telegram interface with image rendering

### Agent Architecture (LangChain)

**Primary Agents:**
- **Game Master Agent** - Main orchestrator, interprets player actions, manages world state
- **NPC Agent** - Handles NPC dialogue and interactions (separate LLM instance)
- **RAG Agent** - Retrieves world lore, enemy info, setting details from vector store
- **Action Validator Agent** - Validates player actions against game rules

**Agent Communication:**
- Agents communicate via structured state objects (not raw text)
- State passed through LangChain's agent executor with tool calling
- Each agent receives only necessary state subset (principle of least privilege)
- Agent outputs validated before state updates

### State Management

**State Structure (Comprehensive Pydantic Model):**
The `GameState` model is a LARGE, self-contained Pydantic model that includes EVERY piece of information needed to fully restore the game to any point. This ensures easy and reliable reversion.

```python
@dataclass(frozen=True)  # Immutable Pydantic model
class GameState(BaseModel):
    # Game identification
    game_id: str
    state_version: int  # Increments with each state change
    created_at: datetime
    
    # All players' complete information
    players: List[Player]  # See Player model below - includes ALL stats, inventory, position
    
    # World state
    world_map: HexMap  # Complete hex map with all cells and their contents
    current_time: TimeState  # Day, half-day increments, total time elapsed
    
    # Complete message history - ALL interactions
    message_history: List[Message]  # See Message model below
    # Includes: player actions, master responses, NPC dialogue, tool outputs
    
    # Combat state (if in combat)
    combat_state: Optional[CombatState]  # Current combat if active, None otherwise
    
    # Active effects (buffs/debuffs)
    active_effects: List[Effect]  # All temporary stat modifiers, potion effects, etc.
    
    # Game metadata
    metadata: GameMetadata  # Settings, difficulty, etc.
    
    # Note: history is NOT stored in state itself
    # History is managed separately as snapshots of GameState
```

**Player Model (Pydantic) - Complete Player Information:**
```python
class Player(BaseModel):
    player_id: str
    name: str
    
    # ALL stats (base + current effective values)
    base_stats: PlayerStats  # Health, Strength, Dexterity, Intelligence, Charisma
    current_stats: PlayerStats  # Effective stats (base + items + buffs/debuffs)
    level: int
    experience: int
    
    # Complete inventory - ALL items
    inventory: PlayerInventory  # See Inventory model below
    # Includes: all bags with all items, equipped items (weapons, cloth, hat, pants, large item)
    
    # Position and movement
    position: HexCoordinate  # Current hex cell position
    movement_history: List[HexCoordinate]  # Path taken (optional, for debugging)
    
    # Status
    is_alive: bool
    status_conditions: List[str]  # e.g., "poisoned", "stunned"
```

**Inventory Model (Pydantic) - Complete Item Information:**
```python
class PlayerInventory(BaseModel):
    # All bags (up to 7)
    bags: List[Bag]  # Each bag has 3-12 slots, contains items
    
    # Equipped items
    primary_weapon: Optional[Item]
    secondary_weapon: Optional[Item]
    active_weapon: Literal["primary", "secondary"]  # Which weapon is active
    cloth: Optional[Item]
    hat: Optional[Item]
    pants: Optional[Item]
    large_item: Optional[Item]  # Items that take 2 hex cells (e.g., horse)
    
    # All items with complete information
    all_items: List[Item]  # Complete list of every item the player owns
```

**Item Model (Pydantic) - Complete Item Information:**
```python
class Item(BaseModel):
    item_id: str
    name: str
    description: str
    
    # Item properties
    tags: List[ItemTag]  # instant, heavy, large, weapon, hat, cloth, pants, potion, book/scroll
    slot_size: int  # 1 for normal, 2 for heavy
    stat_modifiers: Dict[str, int]  # Stat changes when equipped/used
    
    # Location tracking
    location: ItemLocation  # Which bag slot, or "equipped", or "inventory"
    bag_index: Optional[int]  # If in bag, which bag
    slot_index: Optional[int]  # If in bag, which slot
```

**Message Model (Pydantic) - Complete Message History:**
```python
class Message(BaseModel):
    message_id: str
    timestamp: datetime
    sequence_number: int  # Order of messages
    
    # Message source
    source: MessageSource  # "player", "master", "npc", "tool", "system"
    source_id: Optional[str]  # player_id, npc_id, tool_name, etc.
    
    # Message content
    content: str  # The actual message text
    message_type: MessageType  # "action", "response", "dialogue", "tool_output", "system"
    
    # Context
    action_id: Optional[str]  # If related to a specific action
    tool_name: Optional[str]  # If from a tool
    npc_id: Optional[str]  # If from NPC
    
    # Metadata
    metadata: Dict[str, Any]  # Additional context (dice rolls, validation results, etc.)
```

**State Immutability:**
- States are immutable Pydantic models (using `frozen=True` or `@dataclass(frozen=True)`)
- All nested models are also immutable Pydantic models
- New states created via `model_copy()` or `model_dump()` + reconstruction
- History stored as snapshots (full state copies at decision points)
- Revert by restoring complete state from snapshot - no external dependencies needed

**State Transitions:**
- All state changes go through `GameEngine.apply_action()`
- Validates action → Creates new state (using Pydantic `model_copy()` or reconstruction)
- New state includes:
  - Updated players (with all stats, inventory, position)
  - Updated message history (appends new messages)
  - Updated time state
  - Updated world map if needed
  - Updated active effects
  - All other state components
- State snapshot created and appended to history
- History indexed by snapshot index
- Each snapshot contains complete, self-contained GameState
- Supports branching (multiple possible futures) if needed

### Security: Prompt Injection Prevention

**Input Sanitization:**
1. **Player Input Layer:**
   - Strip/escape special tokens: `{`, `}`, `<|`, `|>`, `[INST]`, etc.
   - Validate against whitelist of allowed action types
   - Length limits on user input
   - Character encoding normalization

2. **Agent Prompt Construction:**
   - Use LangChain's `PromptTemplate` with strict variable substitution
   - Separate system prompts from user content
   - Use structured output parsers (Pydantic) for agent responses
   - Never concatenate user input directly into prompts

3. **Agent-to-Agent Communication:**
   - Agents receive structured data (JSON/dict), not raw text
   - Use LangChain's `StructuredTool` for tool definitions
   - Validate tool outputs before passing to next agent
   - Sanitize any text extracted from RAG before agent consumption

4. **RAG Security:**
   - Vector store contains only curated game content
   - Query results filtered/sanitized before agent use
   - Use metadata filtering to prevent retrieval of malicious content

5. **Output Validation:**
   - All agent outputs parsed through Pydantic models
   - Reject outputs that don't match expected schema
   - Log suspicious patterns for monitoring

### Game Engine Design

**Core Classes:**
- `GameEngine` - Main state machine, action processing
- `ActionValidator` - Validates actions against rules
- `DiceRoller` - DnD dice mechanics (d20, d6, etc.)
- `TimeManager` - Tracks game time (half-day increments)
- `InventoryManager` - Handles player inventory, bags, equipment
- `StatCalculator` - Computes effective stats (base + items + buffs)
- `CombatSystem` - Turn-based combat resolution

**Action Processing Flow:**
1. Player submits action → Sanitized
2. `ActionValidator` checks action validity
3. `GameEngine` determines time cost
4. LLM agents called if needed (NPC dialogue, world generation)
   - All agent interactions logged as Messages in state
   - Tool outputs logged as Messages
5. State updated (new immutable Pydantic state created via `model_copy()`)
   - All player stats updated
   - All items updated if inventory changed
   - Message history appended (player action, master response, NPC dialogue, tool outputs)
   - Time state updated
   - World map updated if movement occurred
   - Active effects updated
6. Complete state snapshot created and added to history
7. Response generated for player (from message history)

### LLM Integration (LangChain)

**Agent Setup:**
- Use `langchain.agents.AgentExecutor` for each agent type
- Custom tools for: `roll_dice`, `query_world_lore`, `validate_action`, `calculate_damage`
- Structured output parsers for agent responses
- Separate LLM instances/configs for different agent roles

**Tool Definitions:**
- `RollDiceTool` - DnD dice rolling with modifiers
- `QueryLoreTool` - RAG retrieval for world/enemy info
- `ValidateActionTool` - Rule-based action validation
- `GetPlayerStatsTool` - Retrieve current player stats
- `GetInventoryTool` - Retrieve player inventory state
- `GetMapStateTool` - Retrieve current hex map position

**Agent Prompts:**
- System prompts stored separately, loaded at runtime
- Include role definition, available tools, output format
- Never include user input in system prompt
- Use few-shot examples for consistent behavior

### State History & Revert

**History Storage:**
- Full state snapshots at decision points (not every action)
- Snapshot before: combat, level-up, major world events
- Snapshot on explicit save points
- History stored as list of `StateSnapshot` objects

**Revert Mechanism:**
- `GameEngine.revert_to(snapshot_index)` method
- Rebuilds state from snapshot
- Truncates history after revert point
- Validates revert target before execution

**History Structure:**
```python
class StateSnapshot(BaseModel):
    index: int  # Sequential snapshot number
    timestamp: datetime  # When snapshot was created
    state: GameState  # Complete, self-contained state (all players, items, messages, etc.)
    metadata: Dict[str, Any]  # Additional snapshot metadata (reason, tags, etc.)

    # Note: state contains EVERYTHING needed to restore game
    # No external dependencies required for reversion
```

### Data Models (All Pydantic)

**All models use Pydantic v2 with strict validation and immutability.**

**Core Models:**
- `GameState` - LARGE, comprehensive state model (immutable, frozen)
  - Contains ALL players, ALL items, ALL messages, complete world state
  - Self-contained for easy reversion
- `Player` - Complete player information
  - Base and current stats, level, experience
  - Complete inventory (all bags, all items, all equipment)
  - Position, status conditions
- `PlayerInventory` - Complete inventory state
  - All bags with all items
  - All equipped items (weapons, cloth, hat, pants, large item)
  - Complete item list
- `Item` - Complete item information
  - All properties, tags, stat modifiers
  - Location tracking (which bag/slot or equipped)
- `Bag` - Bag container
  - Size (3-12 slots), contents
- `PlayerStats` - All player statistics
  - Health, Strength, Dexterity, Intelligence, Charisma
- `Message` - Message in history
  - Source (player/master/NPC/tool/system)
  - Content, type, context, metadata
- `MessageHistory` - Ordered list of all messages
- `HexMap` - Complete world map
  - All hex cells with terrain, contents, coordinates
- `HexCell` - Individual map cell
  - Coordinates, terrain type, contents, discovered state
- `HexCoordinate` - Hex grid coordinates (q, r)
- `TimeState` - Game time tracking
  - Current day, half-day increments, total time
- `CombatState` - Current combat information (if active)
  - Participants, turn order, combat log
- `Effect` - Active buff/debuff
  - Type, duration, stat modifications, source
- `StateSnapshot` - Historical state snapshot
  - Complete GameState copy, index, timestamp, action that led here
- `GameMetadata` - Game settings and metadata
  - Difficulty, rules variant, etc.

**All models:**
- Use Pydantic BaseModel
- Are immutable (frozen=True where applicable)
- Include complete validation
- Support JSON serialization for persistence
- Include type hints for all fields

### Frontend Integration

**API Endpoints:**
- `POST /api/game/action` - Submit player action
- `GET /api/game/state` - Get current state
- `POST /api/game/revert` - Revert to snapshot
- `GET /api/game/history` - List available snapshots
- `POST /api/game/image` - Generate scene image

**State Serialization:**
- GameState serialized to JSON for frontend
- Sensitive data (LLM prompts, internal state) excluded
- Only player-visible state exposed

---

## TODO

### Phase 1: Foundation & Core Engine

1. **Setup Project Structure**
   - [v] Create package structure: `autodnd/engine/`, `autodnd/agents/`, `autodnd/models/`, `autodnd/security/`
   - [v] Add dependencies: `pydantic`, `langchain-openai` (or `langchain-anthropic`), `langchain-community`, `langchain-core`
   - [v] Setup type checking (mypy) and linting

2. **Data Models (Pydantic - Comprehensive State)**
   - [v] Define `PlayerStats` model (Health, Strength, Dexterity, Intelligence, Charisma)
   - [v] Define `Item` model (complete: tags, properties, stat modifiers, location tracking)
   - [v] Define `Bag` model (size, slots, contents)
   - [v] Define `PlayerInventory` model (all bags, all equipped items, complete item list)
   - [v] Define `Player` model (complete: base/current stats, level, experience, inventory, position, status)
   - [v] Define `HexCoordinate` model (q, r coordinates)
   - [v] Define `HexCell` model (terrain, coordinates, contents, discovered state)
   - [v] Define `HexMap` model (complete map with all cells)
   - [v] Define `TimeState` model (day, half-day increments, total time)
   - [v] Define `Message` model (source, content, type, context, metadata)
   - [v] Define `MessageHistory` model (ordered list of all messages)
   - [v] Define `Action` model (type enum, parameters dict, validation)
   - [v] Define `CombatState` model (if needed for combat tracking)
   - [v] Define `Effect` model (buffs/debuffs with duration and modifiers)
   - [v] Define `GameMetadata` model (settings, difficulty, etc.)
   - [v] Define `GameState` model (LARGE, comprehensive, immutable, frozen)
     - Include ALL players with ALL stats and ALL items
     - Include ALL messages (player, master, NPC, tool, system)
     - Include complete world map
     - Include current time
     - Include active effects
     - Include combat state if active
     - Self-contained for easy reversion
   - [v] Define `StateSnapshot` model (complete GameState copy + metadata)
   - [v] Ensure all models are immutable Pydantic models with proper validation
   - [v] Add JSON serialization support for all models

3. **Core Game Engine**
   - [v] Implement `GameEngine` class (state management, action processing)
   - [v] Implement `ActionValidator` (rule-based validation)
   - [v] Implement `TimeManager` (half-day increments)
   - [v] Implement `InventoryManager` (bags, equipment slots, item placement)
   - [v] Implement `StatCalculator` (base + items + buffs/debuffs)
   - [v] Implement `DiceRoller` (d20, d6, modifiers)
   - [v] Implement `CombatSystem` (turn-based combat)

4. **State History System**
   - [v] Implement state snapshot creation (complete GameState copy)
   - [v] Implement history storage (in-memory, later persistent)
   - [v] Implement `revert_to()` method (restore complete state from snapshot)
   - [v] Add snapshot triggers (combat start, level-up, save points)
   - [v] Ensure snapshots are self-contained (no external dependencies)
   - [v] Test reversion restores all players, items, messages, time, map state

### Phase 2: Security & Input Sanitization

5. **Input Sanitization Layer**
   - [v] Create `InputSanitizer` class
   - [v] Implement special token stripping/escaping
   - [v] Implement action type whitelist validation
   - [v] Implement length limits and encoding normalization
   - [v] Add unit tests for injection attempts

6. **Prompt Security**
   - [v] Create `PromptBuilder` class using LangChain `PromptTemplate`
   - [v] Separate system prompts from user content
   - [v] Implement structured output parsers (Pydantic)
   - [ ] Create prompt templates for each agent type (SecurityAgent done, others can be added as needed)
   - [v] Add prompt injection test cases

7. **Agent Output Validation**
   - [v] Create `OutputValidator` class
   - [v] Validate all agent outputs against Pydantic schemas
   - [v] Implement rejection handling for invalid outputs
   - [v] Add logging for suspicious patterns

### Phase 3: LangChain Agent System

8. **Tool Definitions**
   - [v] Create `RollDiceTool` (LangChain `StructuredTool`)
   - [v] Create `QueryLoreTool` (RAG integration) - Implemented inside RAGAgent
   - [v] Create `ValidateActionTool` (rule validation) - Implemented inside ActionValidatorAgent
   - [v] Create `GetPlayerStatsTool`
   - [v] Create `GetInventoryTool`
   - [v] Create `GetMapStateTool`
   - [v] Create `GetNPCInfoTool` - Added for NPC interactions
   - [ ] Create `CalculateDamageTool` (combat) - Can be added later

9. **Game Master Agent**
   - [v] Setup LangChain `AgentExecutor` for Game Master
   - [v] Configure LLM (OpenAI/Ollama) with structured output
   - [v] Create system prompt for Game Master role
   - [v] Wire tools to agent
   - [v] Implement agent response parsing

10. **NPC Agent**
    - [v] Setup separate `AgentExecutor` for NPCs
    - [v] Create NPC-specific system prompt
    - [v] Implement NPC dialogue tool
    - [v] Add NPC personality/context injection

11. **RAG Agent**
    - [v] Setup vector store (Chroma/FAISS) for world lore - Basic structure created, can be extended
    - [ ] Create embedding model integration - Structure ready for implementation
    - [v] Implement `QueryLoreTool` with RAG retrieval - Basic tool created, ready for vector store integration
    - [ ] Add metadata filtering for security - Can be added when vector store is implemented
    - [ ] Populate initial vector store with game content - Can be added when vector store is implemented

12. **Action Validator Agent**
    - [v] Setup lightweight agent for action validation
    - [v] Create validation rules/prompts
    - [v] Integrate with `ActionValidator` class

### Phase 4: Agent Communication & State Passing

13. **State Serialization for Agents**
    - [ ] Create `StateSerializer` (serialize GameState to agent-readable format)
    - [ ] Implement state subset extraction (only relevant data per agent)
    - [ ] Create agent context builders (what each agent needs to see)

14. **Agent Orchestration**
    - [ ] Create `AgentOrchestrator` class
    - [ ] Implement agent calling sequence (Game Master → NPC/RAG as needed)
    - [ ] Implement state passing between agents (structured data only)
    - [ ] Add error handling and retry logic

15. **Agent Response Integration**
    - [ ] Parse agent outputs into game actions
    - [ ] Validate agent-suggested actions
    - [ ] Apply agent decisions to game state
    - [ ] Log ALL agent interactions as Messages in state
      - Game Master responses → Message(source="master")
      - NPC dialogue → Message(source="npc")
      - Tool outputs → Message(source="tool")
      - System messages → Message(source="system")
    - [ ] Ensure message history is complete and ordered
    - [ ] Handle agent errors gracefully (log as system messages)

### Phase 5: Game Logic Integration

16. **Hex Map System**
    - [ ] Implement `HexMap` class (hexagonal grid)
    - [ ] Implement cell navigation (hex coordinates)
    - [ ] Add terrain types and movement costs
    - [ ] Integrate with `TimeManager` (half-day per move)

17. **Combat System Integration**
    - [ ] Integrate `CombatSystem` with Game Master agent
    - [ ] Agent decides combat outcomes using dice rolls
    - [ ] Implement turn-based combat flow
    - [ ] Handle combat state in game state

18. **Inventory & Equipment System**
    - [ ] Complete `InventoryManager` implementation
    - [ ] Validate item placement (bags, equipment slots)
    - [ ] Implement stat modifiers from equipment
    - [ ] Handle item tags (heavy, large, instant, etc.)

19. **Time-Based Actions**
    - [ ] Implement action time costs (NO_TIME, HALF_DAY, WHOLE_DAY)
    - [ ] Integrate with `TimeManager`
    - [ ] Update game state based on time progression

### Phase 6: API & Frontend

20. **Flask API Endpoints**
    - [v] `POST /api/game/start` - Initialize new game
    - [v] `POST /api/game/action` - Submit player action (with sanitization)
    - [v] `GET /api/game/state` - Get current state (sanitized for frontend)
    - [v] `POST /api/game/revert` - Revert to snapshot
    - [v] `GET /api/game/history` - List snapshots
    - [ ] `POST /api/game/image` - Generate scene image (future)

21. **State Serialization for API**
    - [v] Create API-safe state serializer (exclude internal data)
    - [v] Implement JSON response formatting
    - [v] Add error response handling

22. **Frontend Integration**
    - [v] Create HTML frontend (or Telegram bot structure)
    - [v] Implement action submission UI
    - [v] Display game state (map, inventory, stats) - Chat interface with messages
    - [v] Add history/revert UI - API endpoints ready
    - [v] Chat-like UI with message type styling (master large, NPC quoted, tool spoilered)
    - [v] LLM configuration modal with hot-reload (OpenAI/Ollama switchable)

### Phase 7: Testing & Refinement

23. **Unit Tests**
    - [ ] Test game engine state transitions
    - [ ] Test action validation
    - [ ] Test inventory management
    - [ ] Test state revert functionality
    - [ ] Test prompt injection prevention

24. **Integration Tests**
    - [ ] Test agent interactions
    - [ ] Test end-to-end game flow
    - [ ] Test RAG integration
    - [ ] Test combat system

25. **Security Testing**
    - [ ] Test prompt injection attempts
    - [ ] Test input sanitization
    - [ ] Test agent output validation
    - [ ] Test state serialization security

### Phase 8: Advanced Features

26. **Image Generation Integration**
    - [ ] Integrate diffusion model API (Stable Diffusion, etc.)
    - [ ] Generate scene images from game state
    - [ ] Cache generated images

27. **Persistence**
    - [ ] Add database for game state persistence
    - [ ] Implement save/load functionality
    - [ ] Store history in database

28. **Multiplayer Support**
    - [ ] Extend state for multiple players
    - [ ] Handle turn order
    - [ ] Sync state across players

29. **Performance Optimization**
    - [ ] Optimize state snapshot creation
    - [ ] Implement lazy loading for history
    - [ ] Cache agent responses where appropriate
    - [ ] Optimize RAG queries

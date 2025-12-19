# Simple LLM-based Dungeon Master

Play DnD solo (or, mb, with friends) without any game master.


+ Can PLAN game with "external" first request 
+ Supports SIMPLIFIED set of DnD rules
+ Supports hexoganical MAP
+ Supports players inventory
    - Every player can carry bags limited to 7 of them
    - Also, player can handle (wear) 2 weapons (with one active), 1 cloth, 1 hat, 1 pants and one large item
    - Every bag can have from 3 to 12 slots
    - Items can have several tags: 
        * instant
        * heavy (takes two slots)
        * large (can't be placed in bag, for example - horse which option two cells instead of one)
        * weapon
        * hat
        * cloth
        * pants
        * potion
        * book/scroll
+ Supports players stats:
    - Stats are:
        * Health(ness)
        * Strength
        * Dexterity
        * Intelligence
        * Charisma
    - Stats can be changed by buffs and debuffs:
        * Temporarily, from potions
        * Temporarily, from enemy skills/items
        * Permanently, by level up (every level up should ask which stats will be raised)
        * By wearing items
+ Simple time-based states with very limited set of dialogue-interaction:
    - Any move from hex-cell to hex-cell has (almost) ALWAYS HALF OF DAY of duration
    - Answering to master's request can take HALF OF DAY or NO time at all
        * Fighting, using instant items or asking (with no action) anything - takes NO time (but processing gameplay forward to next request)
        * Fighting can have multiple turns to defeat enemy
        * Resting, preparing, moving another cell, exploring, using other items - takes HALF OF DAY
        * One exceptional - if specially tagged - rarely action can take WHOLE DAY
+ Written on python with HTML (or mb telegram) frontend
+ Renders image to each state with diffuse-models
+ Game Master should use tools to decide progression:
    - Repsects player stats
    - Rolling dices if needed (inspired from DnD rules)
    - Finding setting/lore about enemies and world with RAG
    - Launching another LLM with self-prompted on communicate with NPC

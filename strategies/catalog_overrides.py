# strategies/catalog_overrides.py
OVERRIDES = {
    # key is "module.ClassName"
    "strategies.library.range_breakout.RangeBreakout": {
        "desc": "Breakout of recent range with optional quick retest filter.",
        # You can also force schema/category/name if you want:
        # "category": "Breakout",
        # "schema": {...}
    },
    # Add more overrides here...
}

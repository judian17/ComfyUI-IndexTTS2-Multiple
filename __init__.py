# __init__.py

from .nodes import IndexTTS2_Dialogue_Studio

NODE_CLASS_MAPPINGS = {
    "IndexTTS2_Dialogue_Studio": IndexTTS2_Dialogue_Studio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_Dialogue_Studio": "🎤 IndexTTS-2 Dialogue Studio",
}

print("""
---
[IndexTTS2] Refactored Dialogue Studio node loaded successfully.
---
""")
import os
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[1] / 'logs'

for p in LOGS_DIR.iterdir():
    if 'Dieudonne' in p.name:
        new_name = p.name.replace('Dieudonne', 'albert')
        new_path = p.with_name(new_name)
        text = p.read_text(encoding='utf-8')
        new_text = text.replace('Dieudonne', 'albert')
        new_path.write_text(new_text, encoding='utf-8')
        p.unlink()
        print(f"Renamed {p.name} -> {new_name}")
print('Done')

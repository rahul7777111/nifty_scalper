import os
import glob
import re

for fpath in glob.glob('c:/Users/rahul/Downloads/NiftyScalper/src/*.py'):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    new_content = new_content.replace('"gpt_timeout_sec", 8.0) or 8.0', '"gpt_timeout_sec", 45.0) or 45.0')
    new_content = new_content.replace('gpt_timeout_sec: float = 8.0', 'gpt_timeout_sec: float = 45.0')
    new_content = new_content.replace('timeout_sec: float = 8.0', 'timeout_sec: float = 45.0')
    new_content = new_content.replace('timeout_sec = 8.0', 'timeout_sec = 45.0')
    new_content = new_content.replace('analyze_timeout = 8.0', 'analyze_timeout = 45.0')
    new_content = new_content.replace('timeout_sec=float(getattr(self.cfg, "gpt_timeout_sec", 8.0) or 8.0)', 'timeout_sec=float(getattr(self.cfg, "gpt_timeout_sec", 45.0) or 45.0)')

    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {fpath}')

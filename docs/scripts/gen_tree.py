import os

IGNORE = {'.git', '__pycache__', 'venv', 'node_modules', '.ipynb_checkpoints'}

def print_tree(root, prefix=""):
    entries = sorted(e for e in os.listdir(root) if e not in IGNORE)
    entries = [e for e in entries if not e.startswith('.') or e in ('.github',)]
    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)

print_tree(".")
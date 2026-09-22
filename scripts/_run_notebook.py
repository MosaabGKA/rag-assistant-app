import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
nb_path = ROOT / "notebooks" / "rag_pipeline.ipynb"

print("exec notebook:", nb_path)
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(
    nb,
    timeout=1800,
    kernel_name="rag-venv",
    resources={"metadata": {"path": str(ROOT / "notebooks")}},
)
client.execute()
nbformat.write(nb, nb_path)
print("EXECUTION OK — outputs written back")
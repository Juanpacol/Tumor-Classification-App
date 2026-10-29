import ast
import pathlib

FORBIDDEN_IMPORTS = {"fastapi", "sqlalchemy", "httpx", "ml_engine", "src"}
PURE_LAYERS = ("domain", "application")


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_domain_and_application_have_no_framework_imports():
    app_root = pathlib.Path(__file__).resolve().parents[1]
    violations = []
    for layer in PURE_LAYERS:
        for py_file in (app_root / layer).rglob("*.py"):
            found = _top_level_imports(py_file) & FORBIDDEN_IMPORTS
            if found:
                violations.append((str(py_file), found))
    assert not violations, f"Forbidden imports found: {violations}"

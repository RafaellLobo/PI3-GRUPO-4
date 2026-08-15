"""Valida as dependencias principais do ambiente da Sprint 1."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


DEPENDENCIES = (
    ("Python", None, None),
    ("NumPy", "numpy", "numpy"),
    ("SciPy", "scipy", "scipy"),
    ("pydicom", "pydicom", "pydicom"),
    ("pylidc", "pylidc", "pylidc"),
    ("SimpleITK", "SimpleITK", "SimpleITK"),
    ("PyRadiomics", "radiomics", "pyradiomics"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn", "scikit-learn"),
    ("SQLAlchemy", "sqlalchemy", "SQLAlchemy"),
    ("ipykernel", "ipykernel", "ipykernel"),
)


def installed_version(module: object, distribution: str) -> str:
    """Retorna a versao exposta pelo modulo ou pelos metadados do pacote."""
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    return importlib.metadata.version(distribution)


def main() -> int:
    failures: list[str] = []

    for name, module_name, distribution in DEPENDENCIES:
        if module_name is None:
            print(f"{name}: {sys.version.split()[0]}")
            continue

        try:
            module = importlib.import_module(module_name)
            version = installed_version(module, distribution)
        except Exception as exc:  # noqa: BLE001 - todos os imports devem ser testados
            failures.append(f"{name} ({module_name}): {exc}")
            print(f"{name}: ERRO", file=sys.stderr)
        else:
            print(f"{name}: {version}")

    if failures:
        print("\nFalhas de importacao:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nAmbiente validado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

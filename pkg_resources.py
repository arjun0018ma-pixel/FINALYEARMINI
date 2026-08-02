import importlib.resources as resources
from pathlib import Path


def resource_filename(package, resource_name):
    package_path = Path(package.replace('.', '/'))
    if not package_path.exists():
        # For vit_keras, the resource lives next to the package module.
        root = Path(__file__).resolve().parent
        package_path = root / package.replace('.', '/')
    resource = package_path / resource_name
    return str(resource)

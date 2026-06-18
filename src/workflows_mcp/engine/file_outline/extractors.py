"""Per-format outline extraction seam.

Each file format has one extractor with the uniform signature
``(file_path, mode) -> str``; ``OUTLINE_EXTRACTORS`` maps file extension to its
extractor. ``extract_outline`` is the single dispatch — there is no per-call
if-elif over extensions and no second extension->language map.

Code languages route through the shared tree-sitter home in
``treesitter_languages`` (``detect_language`` + ``get_parser``); this module owns
only the outline-specific knowledge those modules do not: which AST node types
count as outline symbols per language.
"""

from __future__ import annotations

import ast
import configparser
import json
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml

from ..treesitter_languages import detect_language, get_parser

OutlineMode = Literal["outline", "summary"]
OutlineExtractor = Callable[[Path, OutlineMode], str]

# Tree-sitter symbol node types per language. Python is excluded — it has a
# dedicated AST extractor with richer signature output.
_TREESITTER_SYMBOL_TYPES: dict[str, set[str]] = {
    "typescript": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
    },
    "tsx": {
        "function_declaration",
        "method_definition",
        "class_declaration",
        "interface_declaration",
    },
    "javascript": {"function_declaration", "method_definition", "class_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
}


def extract_treesitter_outline(file_path: Path, language_name: str) -> str:
    """Extract symbol outline using tree-sitter for multi-language support.

    Args:
        file_path: Path to source file
        language_name: Tree-sitter language identifier (e.g. 'typescript', 'go')

    Returns:
        Formatted outline string with symbol tree
    """
    target_types = _TREESITTER_SYMBOL_TYPES.get(language_name)
    if not target_types:
        return "[Unsupported language for tree-sitter extraction]"

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source_code = f.read()
    except Exception as e:
        return f"[Read Error: {e}]"

    try:
        parser = get_parser(language_name)  # type: ignore[arg-type]
        tree = parser.parse(bytes(source_code, "utf8"))
    except Exception as e:
        return f"[Parse Error: {e}]"

    lines: list[str] = []
    source_bytes = source_code.encode("utf8")

    def format_node(node: object, indent: str = "  ", depth: int = 0, max_depth: int = 10) -> None:
        """Walk tree and extract symbols."""
        if depth >= max_depth:
            return

        node_type = getattr(node, "type", None)
        if not node_type:
            return

        if node_type in target_types:
            name = "unknown"
            start_line = getattr(node, "start_point", (0, 0))[0] + 1
            end_line = getattr(node, "end_point", (0, 0))[0] + 1

            # Try to extract name from common field names
            for field in ("name", "identifier", "field_identifier"):
                name_node = getattr(node, "child_by_field_name", lambda x: None)(field)
                if name_node:
                    name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                        "utf8", errors="replace"
                    )
                    break

            symbol_label = node_type.replace("_", " ").title().replace(" ", "")
            lines.append(f"{indent}├── {symbol_label}: {name} [{start_line}-{end_line}]")

            # Recursively process children with increased indent
            new_indent = indent + "│   "
            for child in getattr(node, "children", []):
                format_node(child, new_indent, depth + 1, max_depth)
        else:
            for child in getattr(node, "children", []):
                format_node(child, indent, depth, max_depth)

    format_node(tree.root_node)

    return "\n".join(lines) if lines else "  [No symbols found]"


def extract_python_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract symbol outline from Python file using AST.

    Returns tree structure with classes, functions, methods, signatures,
    and line ranges. Handles syntax errors gracefully. ``summary`` mode
    additionally includes the first line of each docstring.

    Args:
        file_path: Path to Python file
        mode: Outline mode ('summary' includes docstrings)

    Returns:
        Formatted outline string with symbol tree
    """
    include_docstrings = mode == "summary"
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        return f"[Syntax Error at line {e.lineno}: {e.msg}]"
    except Exception as e:
        return f"[Parse Error: {type(e).__name__}: {e}]"

    lines = []

    # Extract imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    if imports:
        unique_imports = sorted(set(imports))
        lines.append(f"  ├── imports ({len(unique_imports)}): {', '.join(unique_imports)}")

    # Extract top-level symbols
    def format_function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str = "  ") -> str:
        params = []
        for arg in node.args.args:
            param_str = arg.arg
            if arg.annotation:
                try:
                    param_str += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            params.append(param_str)

        return_type = ""
        if node.returns:
            try:
                return_type = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        sig = f"({', '.join(params)}){return_type}"
        line_info = f"[{node.lineno}"
        if hasattr(node, "end_lineno") and node.end_lineno:
            line_info += f"-{node.end_lineno}"
        line_info += "]"

        result = f"{indent}├── def {node.name}{sig} {line_info}"

        if include_docstrings and ast.get_docstring(node):
            docstring = ast.get_docstring(node).split("\n")[0][:60]  # type: ignore
            result += f'\n{indent}│   └── "{docstring}..."'

        return result

    def format_class(node: ast.ClassDef, indent: str = "  ") -> str:
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass

        bases_str = f"({', '.join(bases)})" if bases else ""
        line_info = f"[{node.lineno}"
        if hasattr(node, "end_lineno") and node.end_lineno:
            line_info += f"-{node.end_lineno}"
        line_info += "]"

        class_line = f"{indent}├── class {node.name}{bases_str} {line_info}"
        result = [class_line]

        if include_docstrings and ast.get_docstring(node):
            docstring = ast.get_docstring(node).split("\n")[0][:60]  # type: ignore
            result.append(f'{indent}│   └── "{docstring}..."')

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(format_function(item, indent + "│   "))

        if methods:
            result.extend(methods)

        return "\n".join(result)

    # Process top-level nodes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(format_function(node))
        elif isinstance(node, ast.ClassDef):
            lines.append(format_class(node))

    return "\n".join(lines) if lines else "  [No symbols found]"


def extract_markdown_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract header structure from Markdown file.

    Args:
        file_path: Path to Markdown file
        mode: Outline mode (does not affect header structure)

    Returns:
        Formatted outline string with header tree
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"[Read Error: {e}]"

    lines = []
    in_code_block = False
    for i, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        # Track fenced code blocks (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("#"):
            # Count heading level
            level = 0
            for char in stripped:
                if char == "#":
                    level += 1
                else:
                    break
            level = min(level, 6)  # Max 6 levels in Markdown

            indent = "  " + ("  " * (level - 1))
            header = stripped.lstrip("#").strip()[:60]
            lines.append(f"{indent}├── {header} [{i}]")

    return "\n".join(lines) if lines else "  [No headers found]"


def _format_structured_value(val: object) -> str:
    """Format value with type info and preview (shared helper).

    Args:
        val: Value to format

    Returns:
        Formatted string with type and preview
    """
    if isinstance(val, dict):
        return f"object ({len(val)} keys)"
    elif isinstance(val, list):
        return f"array ({len(val)} items)"
    elif isinstance(val, str):
        preview = val[:50] + "..." if len(val) > 50 else val
        return f'"{preview}"'
    elif isinstance(val, (int, float, bool)) or val is None:
        return str(val)
    else:
        return f"{type(val).__name__}"


def _walk_structured_data(
    obj: object, indent: str = "  ", depth: int = 0, max_depth: int = 5
) -> list[str]:
    """Recursively walk structured data (JSON/YAML) - shared helper.

    Args:
        obj: Object to walk (dict, list, or scalar)
        indent: Current indentation
        depth: Current depth in tree
        max_depth: Maximum depth to traverse

    Returns:
        List of formatted outline lines
    """
    if depth >= max_depth:
        return [f"{indent}├── [max depth reached]"]

    lines = []
    if isinstance(obj, dict):
        for i, (key, value) in enumerate(obj.items()):
            connector = "├──" if i < len(obj) - 1 else "└──"
            value_preview = _format_structured_value(value)
            lines.append(f"{indent}{connector} {key}: {value_preview}")

            # Recursively show nested structures
            if isinstance(value, (dict, list)) and depth < max_depth - 1:
                new_indent = indent + ("│   " if i < len(obj) - 1 else "    ")
                lines.extend(_walk_structured_data(value, new_indent, depth + 1, max_depth))

    elif isinstance(obj, list):
        preview_count = min(3, len(obj))
        for i in range(preview_count):
            connector = "├──" if i < preview_count - 1 else "└──"
            value_preview = _format_structured_value(obj[i])
            lines.append(f"{indent}{connector} [{i}]: {value_preview}")

            # Recursively show nested structures
            if isinstance(obj[i], (dict, list)) and depth < max_depth - 1:
                new_indent = indent + ("│   " if i < preview_count - 1 else "    ")
                lines.extend(_walk_structured_data(obj[i], new_indent, depth + 1, max_depth))

        if len(obj) > preview_count:
            lines.append(f"{indent}└── ... +{len(obj) - preview_count} more items")

    return lines


def extract_json_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract structure from JSON file.

    Args:
        file_path: Path to JSON file
        mode: Outline mode (does not affect structure)

    Returns:
        Formatted outline with JSON structure tree
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return f"[JSON Parse Error at line {e.lineno}: {e.msg}]"
    except Exception as e:
        return f"[Read Error: {e}]"

    return "\n".join(_walk_structured_data(data))


def extract_yaml_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract structure from YAML file.

    Args:
        file_path: Path to YAML file
        mode: Outline mode (does not affect structure)

    Returns:
        Formatted outline with YAML structure tree
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return f"[YAML Parse Error: {e}]"
    except Exception as e:
        return f"[Read Error: {e}]"

    # YAML uses same structure as JSON, reuse the shared walker
    return "\n".join(_walk_structured_data(data))


def extract_toml_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract structure from TOML file.

    Args:
        file_path: Path to TOML file
        mode: Outline mode (does not affect structure)

    Returns:
        Formatted outline with TOML table structure
    """
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        return f"[TOML Parse Error: {e}]"
    except Exception as e:
        return f"[Read Error: {e}]"

    def format_toml_structure(obj: object, indent: str = "  ", table_path: str = "") -> list[str]:
        """Format TOML structure showing tables and keys."""
        lines = []

        if isinstance(obj, dict):
            for i, (key, value) in enumerate(obj.items()):
                connector = "├──" if i < len(obj) - 1 else "└──"

                if isinstance(value, dict):
                    # Table
                    new_path = f"{table_path}.{key}" if table_path else key
                    lines.append(f"{indent}{connector} [{new_path}]")
                    new_indent = indent + ("│   " if i < len(obj) - 1 else "    ")
                    lines.extend(format_toml_structure(value, new_indent, new_path))
                elif isinstance(value, list):
                    lines.append(f"{indent}{connector} {key} = array ({len(value)} items)")
                elif isinstance(value, str):
                    preview = value[:40] + "..." if len(value) > 40 else value
                    lines.append(f'{indent}{connector} {key} = "{preview}"')
                else:
                    lines.append(f"{indent}{connector} {key} = {value}")

        return lines

    return "\n".join(format_toml_structure(data))


def extract_xml_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract structure from XML/HTML file.

    Args:
        file_path: Path to XML or HTML file
        mode: Outline mode (does not affect structure)

    Returns:
        Formatted outline with element hierarchy
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        return f"[XML Parse Error at line {e.position[0]}: {e.msg}]"
    except Exception as e:
        return f"[Read Error: {e}]"

    def format_element(
        elem: ET.Element, indent: str = "  ", depth: int = 0, max_depth: int = 8
    ) -> list[str]:
        """Format XML element tree."""
        if depth >= max_depth:
            return [f"{indent}├── [max depth reached]"]

        lines = []
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag  # Remove namespace

        # Element with attributes
        attrs = ""
        if elem.attrib:
            attr_list = [
                f'{k}="{v[:20]}..."' if len(v) > 20 else f'{k}="{v}"'
                for k, v in list(elem.attrib.items())[:3]
            ]
            attrs = f" [{', '.join(attr_list)}]"
            if len(elem.attrib) > 3:
                attrs += f" +{len(elem.attrib) - 3} more"

        # Text content preview
        text_preview = ""
        if elem.text and elem.text.strip():
            text = elem.text.strip()[:40]
            text_preview = f': "{text}..."' if len(elem.text.strip()) > 40 else f': "{text}"'

        children = list(elem)
        child_count = f" ({len(children)} children)" if children else ""

        lines.append(f"{indent}├── <{tag}>{attrs}{text_preview}{child_count}")

        # Show children
        if children and depth < max_depth - 1:
            preview_count = min(5, len(children))
            for i, child in enumerate(children[:preview_count]):
                new_indent = indent + "│   "
                lines.extend(format_element(child, new_indent, depth + 1, max_depth))

            if len(children) > preview_count:
                lines.append(f"{indent}│   └── ... +{len(children) - preview_count} more elements")

        return lines

    return "\n".join(format_element(root))


def extract_config_outline(file_path: Path, mode: OutlineMode) -> str:
    """Extract structure from config files (INI, ENV, properties).

    Args:
        file_path: Path to config file
        mode: Outline mode (does not affect structure)

    Returns:
        Formatted outline with sections and keys
    """
    file_ext = file_path.suffix.lower()

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"[Read Error: {e}]"

    # ENV/properties files: simple key=value format
    if file_ext in [".env", ".properties"]:
        lines = []
        for i, line in enumerate(content.split("\n"), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                value_preview = value[:40] + "..." if len(value) > 40 else value
                lines.append(f"  ├── {key.strip()} = {value_preview}")
        return "\n".join(lines) if lines else "  [No keys found]"

    # INI/CFG/CONF files: section-based format
    config = configparser.ConfigParser()
    try:
        config.read_string(content)
    except configparser.Error as e:
        return f"[Config Parse Error: {e}]"

    lines = []
    for section in config.sections():
        lines.append(f"  ├── [{section}]")
        items = list(config.items(section))
        for i, (key, value) in enumerate(items):
            connector = "│   ├──" if i < len(items) - 1 else "│   └──"
            value_preview = value[:40] + "..." if len(value) > 40 else value
            lines.append(f"  {connector} {key} = {value_preview}")

    return "\n".join(lines) if lines else "  [No sections found]"


def extract_adaptive_preview(file_path: Path, mode: OutlineMode) -> str:
    """Adaptive preview for generic text files, scaled to file size.

    Args:
        file_path: Path to file
        mode: Outline mode (does not affect preview)

    Returns:
        Formatted preview with adaptive line sampling
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines_content = f.readlines()
        total_lines = len(lines_content)

        preview_lines: list[tuple[int, str]] = []

        # Adaptive scaling based on file size
        if total_lines <= 20:
            # Small files: show all lines
            preview_lines = [(i + 1, line.rstrip()) for i, line in enumerate(lines_content)]
        elif total_lines <= 100:
            # Medium files: first 10 + last 10
            head_count = 10
            tail_count = 10
            preview_lines = (
                [(i + 1, line.rstrip()) for i, line in enumerate(lines_content[:head_count])]
                + [(-1, f"... ({total_lines - head_count - tail_count} lines omitted) ...")]
                + [
                    (total_lines - tail_count + i + 1, line.rstrip())
                    for i, line in enumerate(lines_content[-tail_count:])
                ]
            )
        else:
            # Large files: first 15 + middle sample + last 15
            head_count = 15
            tail_count = 15
            middle_count = 5
            middle_index = total_lines // 2
            middle_start = middle_index - middle_count // 2

            preview_lines = (
                [(i + 1, line.rstrip()) for i, line in enumerate(lines_content[:head_count])]
                + [(-1, f"... ({middle_start - head_count} lines omitted) ...")]
                + [
                    (middle_start + i + 1, line.rstrip())
                    for i, line in enumerate(
                        lines_content[middle_start : middle_start + middle_count]
                    )
                ]
                + [
                    (
                        -1,
                        (
                            f"... ({total_lines - tail_count - middle_start - middle_count} "
                            "lines omitted) ..."
                        ),
                    )
                ]
                + [
                    (total_lines - tail_count + i + 1, line.rstrip())
                    for i, line in enumerate(lines_content[-tail_count:])
                ]
            )

        # Format output (no arbitrary char truncation)
        return "\n".join(
            f"  {num}: {line}" if num > 0 else f"  {line}" for num, line in preview_lines
        )
    except Exception as e:
        return f"[Read Error: {e}]"


def _extract_code_or_preview(file_path: Path, mode: OutlineMode) -> str:
    """Outline a source file via tree-sitter, falling back to adaptive preview.

    Routes through the shared ``detect_language`` home; only languages with a
    symbol map yield a symbol outline. Unknown languages and parse failures
    fall back to the adaptive preview.
    """
    language = detect_language(str(file_path))
    if language in _TREESITTER_SYMBOL_TYPES:
        result = extract_treesitter_outline(file_path, language)
        if not result.startswith("["):
            return result

    return extract_adaptive_preview(file_path, mode)


# Extension -> extractor. The single home for outline format knowledge. Code
# extensions not listed here (.ts/.go/...) are detected via the shared
# tree-sitter language home by the default branch in extract_outline().
OUTLINE_EXTRACTORS: dict[str, OutlineExtractor] = {
    ".py": extract_python_outline,
    ".pyi": extract_python_outline,
    ".md": extract_markdown_outline,
    ".markdown": extract_markdown_outline,
    ".json": extract_json_outline,
    ".yaml": extract_yaml_outline,
    ".yml": extract_yaml_outline,
    ".toml": extract_toml_outline,
    ".xml": extract_xml_outline,
    ".html": extract_xml_outline,
    ".htm": extract_xml_outline,
    ".ini": extract_config_outline,
    ".cfg": extract_config_outline,
    ".conf": extract_config_outline,
    ".env": extract_config_outline,
    ".properties": extract_config_outline,
}


def extract_outline(file_path: Path, mode: OutlineMode) -> str:
    """Dispatch to the extractor for the file's format.

    Extensions in ``OUTLINE_EXTRACTORS`` use their registered extractor;
    everything else is treated as source code (tree-sitter symbol outline when
    the language is supported, adaptive preview otherwise).

    Args:
        file_path: Path to file
        mode: Outline mode ('summary' enriches the Python extractor)

    Returns:
        Formatted outline appropriate for the file type
    """
    extractor = OUTLINE_EXTRACTORS.get(file_path.suffix.lower(), _extract_code_or_preview)
    return extractor(file_path, mode)

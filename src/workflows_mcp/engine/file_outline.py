"""File outline extraction utilities for ReadFiles executor.

Provides AST-based outline extraction for Python files, header extraction
for Markdown files, and generic preview for other file types.
"""

from __future__ import annotations

import ast
import configparser
import fnmatch
import json
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import pathspec
import yaml
from tree_sitter_languages import get_parser  # type: ignore

# Default exclusion patterns (version control, dependencies, build artifacts)
DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    # Version control
    "**/.git/*",
    "**/.git/**",
    "**/.svn/*",
    "**/.hg/*",
    "**/.CVS/*",
    "**/.DS_Store",
    # Dependencies
    "**/node_modules/**",
    "**/bower_components/**",
    "**/vendor/**",
    "**/venv/**",
    "**/env/**",
    "**/.venv/**",
    # Python specific
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.egg-info/**",
    # Build artifacts and logs
    "**/build/**",
    "**/dist/**",
    "**/*.log",
    "**/*.tmp",
    "**/*.swp",
    # Compiled files and archives
    "**/*.so",
    "**/*.dll",
    "**/*.exe",
    "**/*.jar",
    "**/*.class",
    "**/*.zip",
    "**/*.tar.gz",
    "**/*.rar",
]

# Binary file extensions to base64 encode
BASE64_ENCODE_EXTENSIONS: set[str] = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    # Documents
    ".pdf",
    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    # Video
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


def is_binary(file_path: Path) -> bool:
    """Check if file is binary by reading first 4KB.

    Args:
        file_path: Path to file to check

    Returns:
        True if file contains null bytes (binary), False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(4096)
            return b"\x00" in chunk
    except OSError:
        return True


def load_gitignore_patterns(base_path: Path) -> list[str]:
    """Load .gitignore patterns from base directory.

    Args:
        base_path: Base directory containing .gitignore file

    Returns:
        List of gitignore patterns (empty if no .gitignore found)
    """
    gitignore_path = base_path / ".gitignore"
    patterns: list[str] = []

    if not gitignore_path.exists():
        return patterns

    try:
        with open(gitignore_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    except OSError:
        pass

    return patterns


def matches_pattern(file_path: Path, base_path: Path, pattern: str) -> bool:
    """Check if file matches a glob pattern (handles ** recursive patterns).

    Args:
        file_path: Absolute file path to check
        base_path: Base directory for relative path calculation
        pattern: Glob pattern (may contain ** for recursive matching)

    Returns:
        True if file matches pattern, False otherwise
    """
    try:
        relative_path = file_path.relative_to(base_path)
        relative_str = str(relative_path)

        # Handle different pattern types
        if pattern.startswith("**/"):
            # Pattern like **/foo matches foo at any depth
            sub_pattern = pattern[3:]
            if fnmatch.fnmatch(relative_str, f"*/{sub_pattern}") or fnmatch.fnmatch(
                relative_str, sub_pattern
            ):
                return True
        elif pattern.endswith("/**"):
            # Pattern like foo/** matches everything under foo/
            dir_pattern = pattern[:-3]
            if relative_str.startswith(dir_pattern + "/") or relative_str == dir_pattern:
                return True
        elif "**" in pattern:
            # General ** pattern - convert to simpler form
            converted = pattern.replace("**/", "*/").replace("/**", "/*")
            if fnmatch.fnmatch(relative_str, converted):
                return True

        # Direct pattern match
        if fnmatch.fnmatch(relative_str, pattern):
            return True

        # Check path components
        parts = relative_path.parts
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(partial, pattern.rstrip("/**")):
                return True

    except ValueError:
        pass

    return False


def create_gitignore_spec(patterns: list[str]) -> pathspec.PathSpec | None:
    """Create PathSpec for gitignore pattern matching.

    Args:
        patterns: List of gitignore patterns

    Returns:
        PathSpec instance, or None if no patterns or creation fails
    """
    if not patterns:
        return None

    try:
        return pathspec.GitIgnoreSpec.from_lines(patterns)
    except Exception:
        return None


def matches_gitignore(file_path: Path, base_path: Path, spec: pathspec.PathSpec) -> bool:
    """Check if file matches gitignore patterns using pathspec.

    Args:
        file_path: Absolute file path to check
        base_path: Base directory for relative path calculation
        spec: PathSpec instance from create_gitignore_spec

    Returns:
        True if file should be ignored, False otherwise
    """
    try:
        relative_path = str(file_path.relative_to(base_path))
        return spec.match_file(relative_path)
    except (ValueError, AttributeError):
        return False


def extract_treesitter_outline(file_path: Path, language_name: str) -> str:
    """Extract symbol outline using tree-sitter for multi-language support.

    Args:
        file_path: Path to source file
        language_name: Tree-sitter language name (e.g., 'typescript', 'go', 'rust')

    Returns:
        Formatted outline string with symbol tree
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            source_code = f.read()
    except Exception as e:
        return f"[Read Error: {e}]"

    try:
        parser = get_parser(language_name)
        tree = parser.parse(bytes(source_code, "utf8"))
    except Exception as e:
        return f"[Parse Error: {e}]"

    # Language-specific symbol types
    symbol_types = {
        "typescript": [
            "function_declaration",
            "method_definition",
            "class_declaration",
            "interface_declaration",
        ],
        "javascript": ["function_declaration", "method_definition", "class_declaration"],
        "go": ["function_declaration", "method_declaration", "type_declaration"],
        "rust": ["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
        "java": [
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "constructor_declaration",
        ],
        "cpp": [
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
        ],
        "c": ["function_definition", "struct_specifier"],
        "csharp": [
            "method_declaration",
            "class_declaration",
            "interface_declaration",
            "constructor_declaration",
        ],
        "ruby": ["method", "class", "module"],
        "php": ["function_definition", "method_declaration", "class_declaration"],
    }

    target_types = set(symbol_types.get(language_name, []))
    if not target_types:
        return "[Unsupported language for tree-sitter extraction]"

    lines = []

    def format_node(node: object, indent: str = "  ", depth: int = 0, max_depth: int = 10) -> None:
        """Walk tree and extract symbols."""
        if depth >= max_depth:
            return

        # Extract node properties
        node_type = getattr(node, "type", None)
        if not node_type:
            return

        # Check if this is a target symbol type
        if node_type in target_types:
            # Get node text (name)
            name = "unknown"
            start_line = getattr(node, "start_point", (0, 0))[0] + 1
            end_line = getattr(node, "end_point", (0, 0))[0] + 1

            # Try to extract name from common field names
            name_fields = ["name", "identifier", "field_identifier"]
            for field in name_fields:
                name_node = getattr(node, "child_by_field_name", lambda x: None)(field)
                if name_node:
                    name_bytes = source_code.encode("utf8")[
                        name_node.start_byte : name_node.end_byte
                    ]
                    name = name_bytes.decode("utf8", errors="replace")
                    break

            # Format symbol line
            symbol_label = node_type.replace("_", " ").title().replace(" ", "")
            lines.append(f"{indent}├── {symbol_label}: {name} [{start_line}-{end_line}]")

            # Recursively process children with increased indent
            new_indent = indent + "│   "
            children = getattr(node, "children", [])
            for child in children:
                format_node(child, new_indent, depth + 1, max_depth)
        else:
            # Not a target symbol, but process children
            children = getattr(node, "children", [])
            for child in children:
                format_node(child, indent, depth, max_depth)

    # Start traversal from root
    root_node = tree.root_node
    format_node(root_node)

    return "\n".join(lines) if lines else "  [No symbols found]"


def extract_python_outline(file_path: Path, include_docstrings: bool) -> str:
    """Extract symbol outline from Python file using AST.

    Returns tree structure with classes, functions, methods, signatures,
    and line ranges. Handles syntax errors gracefully.

    Args:
        file_path: Path to Python file
        include_docstrings: Whether to include docstrings (summary mode)

    Returns:
        Formatted outline string with symbol tree
    """
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


def extract_markdown_outline(file_path: Path) -> str:
    """Extract header structure from Markdown file.

    Args:
        file_path: Path to Markdown file

    Returns:
        Formatted outline string with header tree
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"[Read Error: {e}]"

    lines = []
    for i, line in enumerate(content.split("\n"), 1):
        if line.strip().startswith("#"):
            # Count heading level
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break
            level = min(level, 6)  # Max 6 levels in Markdown

            indent = "  " + ("  " * (level - 1))
            header = line.strip("#").strip()[:60]
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


def extract_json_outline(file_path: Path) -> str:
    """Extract structure from JSON file.

    Args:
        file_path: Path to JSON file

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


def extract_yaml_outline(file_path: Path) -> str:
    """Extract structure from YAML file.

    Args:
        file_path: Path to YAML file

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


def extract_toml_outline(file_path: Path) -> str:
    """Extract structure from TOML file.

    Args:
        file_path: Path to TOML file

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


def extract_xml_outline(file_path: Path) -> str:
    """Extract structure from XML/HTML file.

    Args:
        file_path: Path to XML or HTML file

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


def extract_config_outline(file_path: Path) -> str:
    """Extract structure from config files (INI, ENV, properties).

    Args:
        file_path: Path to config file

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


def extract_adaptive_preview(file_path: Path) -> str:
    """Adaptive preview for generic text files, scaled to file size.

    Args:
        file_path: Path to file

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


def extract_generic_outline(file_path: Path) -> str:
    """Smart outline extraction based on file type (hybrid approach).

    Dispatches to appropriate handler:
    - Python: AST-based (stays as-is for compatibility)
    - Tree-sitter supported languages: TypeScript, JavaScript, Go, Rust, Java, C++, C, C#, Ruby, PHP
    - Structured formats (JSON, YAML, TOML, XML, HTML): Tree-based structure
    - Config files (INI, ENV, properties): Key-value/section extraction
    - Generic text: Adaptive preview scaled to file size

    Args:
        file_path: Path to file

    Returns:
        Formatted outline appropriate for file type
    """
    file_ext = file_path.suffix.lower()

    # Structured formats: parse and extract structure
    if file_ext == ".json":
        return extract_json_outline(file_path)
    elif file_ext in [".yaml", ".yml"]:
        return extract_yaml_outline(file_path)
    elif file_ext == ".toml":
        return extract_toml_outline(file_path)
    elif file_ext in [".xml", ".html", ".htm"]:
        return extract_xml_outline(file_path)

    # Config files: extract sections/keys
    elif file_ext in [".ini", ".cfg", ".conf", ".env", ".properties"]:
        return extract_config_outline(file_path)

    # Tree-sitter supported languages (multi-language code outline)
    else:
        # Map file extensions to tree-sitter language names
        treesitter_lang_map = {
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".h": "c",  # Could be C or C++, default to C
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
        }

        lang_name = treesitter_lang_map.get(file_ext)
        if lang_name:
            # Try tree-sitter first
            result = extract_treesitter_outline(file_path, lang_name)
            # If tree-sitter fails or returns error, fall back to adaptive preview
            if not result.startswith("["):
                return result

        # Fallback: adaptive preview for unsupported/failed languages
        return extract_adaptive_preview(file_path)


def generate_file_outline(file_path: Path, mode: Literal["outline", "summary"]) -> str:
    """Generate outline for a file based on its type and mode.

    Extracts structure/symbols with context reduction (90-97%).
    This function is only called for outline/summary modes.
    Full mode is handled separately by the executor (direct file read).

    Args:
        file_path: Path to file
        mode: Outline extraction mode
            - "outline": Structure/symbol extraction only (90-97% reduction)
            - "summary": Outline + docstrings/comments (85-95% reduction)

    Returns:
        Formatted file outline with header
    """
    file_ext = file_path.suffix.lower()

    # Count total lines for header
    total_lines = 0
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        total_lines = 0

    header = f"--- {file_path.name} ({total_lines} lines) ---"

    # Python: use AST with mode-specific docstring inclusion
    if file_ext == ".py":
        include_docs = mode == "summary"
        outline = extract_python_outline(file_path, include_docs)
    # Markdown: header extraction (mode doesn't affect structure)
    elif file_ext in [".md", ".markdown"]:
        outline = extract_markdown_outline(file_path)
    # All other files: use hybrid extraction system
    # (JSON, YAML, TOML, XML, HTML, config files, tree-sitter languages, adaptive preview)
    else:
        outline = extract_generic_outline(file_path)

    return f"{header}\n{outline}"

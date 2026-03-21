from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from hnsw_logic.agents.runtime.models import MemoryLearningBundle
from hnsw_logic.domain.serialization import ensure_dir


class ControlledSelfUpdateManager:
    def __init__(self, repo_root: Path, allowlist: list[str]):
        self.repo_root = repo_root
        self.allowlist = allowlist

    def _is_allowed(self, relative_path: str) -> bool:
        return any(fnmatch(relative_path, pattern) for pattern in self.allowlist)

    def _replace_section(self, content: str, heading: str, lines: list[str]) -> str:
        marker = f"## {heading}"
        start = content.find(marker)
        if start < 0:
            return content
        after = content.find("\n## ", start + len(marker))
        end = len(content) if after < 0 else after
        section_body = "\n".join(f"- {line}" for line in lines) if lines else "- No learned patterns recorded yet."
        replacement = f"{marker}\n\n{section_body}\n"
        prefix = content[:start]
        suffix = content[end:]
        return prefix + replacement + suffix.lstrip("\n")

    def _upsert_reference_updates(self, content: str, lines: list[str]) -> str:
        marker = "## Learned Updates"
        normalized = [line.strip() for line in lines if str(line).strip()]
        if not normalized:
            return content
        existing_items: list[str] = []
        start = content.find(marker)
        if start >= 0:
            after = content.find("\n## ", start + len(marker))
            end = len(content) if after < 0 else after
            section = content[start:end]
            for line in section.splitlines()[1:]:
                stripped = line.strip()
                if stripped.startswith("- "):
                    existing_items.append(stripped[2:].strip())
            prefix = content[:start]
            suffix = content[end:]
        else:
            prefix = content.rstrip()
            suffix = "\n" if prefix else ""
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*existing_items, *normalized]:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        section_body = "\n".join(f"- {line}" for line in merged) if merged else "- No updates."
        replacement = f"{marker}\n\n{section_body}\n"
        if start >= 0:
            return prefix + replacement + suffix.lstrip("\n")
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{replacement}"

    def update_agents_memory(self, bundle: MemoryLearningBundle, memory_path: Path) -> None:
        relative_path = str(memory_path.relative_to(self.repo_root))
        if not self._is_allowed(relative_path):
            return
        content = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
        content = self._replace_section(content, "Known Failure Patterns", bundle.failure_patterns)
        content = self._replace_section(content, "Learned Patterns", bundle.learned_patterns)
        ensure_dir(memory_path.parent)
        memory_path.write_text(content, encoding="utf-8")

    def update_references(self, updates: dict[str, list[str]]) -> None:
        for relative_path, lines in updates.items():
            normalized = relative_path[2:] if relative_path.startswith("./") else relative_path
            if not self._is_allowed(normalized):
                continue
            path = self.repo_root / normalized
            ensure_dir(path.parent)
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            updated = self._upsert_reference_updates(content, lines)
            path.write_text(updated, encoding="utf-8")

    def apply(self, bundle: MemoryLearningBundle, memory_path: Path) -> None:
        self.update_agents_memory(bundle, memory_path)
        self.update_references(bundle.reference_updates)

"""In-memory state for the landmarks placed during one landmark-picking run.

A WIP run is a single place-everything / confirm / close action, so
`LandmarkState` itself never touches disk. `to_dict`/`from_dict` are a
pure data round-trip only, with no file I/O - the debug-mode CLI runner
(`wips/atrial_regionalisation/main.py`) uses them to save/reload a session
as JSON purely as a debugging convenience (so landmarks don't have to be
re-clicked on every debug run); the EP Workbench WIP flow never calls
them."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LandmarkState:
    chamber: str
    points: dict[str, dict] = field(default_factory=dict)  # name -> {vertex_id, xyz}
    loops: dict[str, dict] = field(default_factory=dict)  # name -> {vertex_ids, xyz} (closed)
    paths: dict[str, dict] = field(default_factory=dict)  # name -> {vertex_ids, xyz} (open, incl. endpoints)

    def set_point(self, name: str, vertex_id: int, xyz) -> None:
        self.points[name] = {"vertex_id": int(vertex_id), "xyz": [float(v) for v in xyz]}

    def set_loop(self, name: str, vertex_ids: list[int], xyz_list) -> None:
        self.loops[name] = {
            "vertex_ids": [int(v) for v in vertex_ids],
            "xyz": [[float(v) for v in xyz] for xyz in xyz_list],
        }

    def set_path(self, name: str, vertex_ids: list[int], xyz_list) -> None:
        self.paths[name] = {
            "vertex_ids": [int(v) for v in vertex_ids],
            "xyz": [[float(v) for v in xyz] for xyz in xyz_list],
        }

    def remove(self, name: str) -> None:
        self.points.pop(name, None)
        self.loops.pop(name, None)
        self.paths.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self.points or name in self.loops or name in self.paths

    def to_dict(self) -> dict:
        return {"chamber": self.chamber, "points": self.points, "loops": self.loops, "paths": self.paths}

    @classmethod
    def from_dict(cls, data: dict) -> "LandmarkState":
        return cls(
            chamber=data["chamber"],
            points=data.get("points", {}),
            loops=data.get("loops", {}),
            paths=data.get("paths", {}),
        )

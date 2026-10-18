# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ imports
from dataclasses import dataclass, replace
import math

# @@ declare the type
@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    # @@ functions on it
    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def plus(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def scaled(self, k):
        return Vec2(k * self.x, k * self.y)

# @@ build values and call the functions
p = Vec2(3.0, 4.0)
q = Vec2(x=1.0, y=-2.0)
r = p.plus(q.scaled(2.0))
print(f"p = ({p.x}, {p.y}), |p| = {p.norm()}")
print(f"r = ({r.x}, {r.y})")

# @@ a changed copy
moved = replace(p, y=0.0)
print(f"moved = ({moved.x}, {moved.y}), p unchanged: ({p.x}, {p.y})")

# @@ a record with an array field
@dataclass
class Body:
    name: str
    mass: float
    pos: list

earth = Body("earth", 5.97e24, [1.0, 0.0])
earth.pos[1] = 0.25
print(earth.name, earth.mass, earth.pos)

# @@ checks
assert p.norm() == 5.0
assert r == Vec2(5.0, 0.0)

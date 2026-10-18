# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
# @@ the input, one reading per line
LOG = [
    "station=north temp=21.5 rain=0",
    "station=south temp=18.25 rain=12",
    "station=east  temp=-3.0 rain=4",
    "# a comment line",
    "station=west temp=1e1 rain=7",
]

# @@ read the fields of one line
def parse(line):
    fields = {}
    for part in line.split():
        key, value = part.split("=")
        fields[key] = value
    return fields

# @@ convert the numbers
names = []
temps = []
total_rain = 0
for line in LOG:
    if line.startswith("#"):
        continue
    f = parse(line)
    names.append(f["station"])
    temps.append(float(f["temp"]))
    total_rain += int(f["rain"])

# @@ report
for name, t in zip(names, temps):
    print(f"{name:>6}: {t:6.2f}")
print(f"mean temperature {sum(temps) / len(temps):.3f}")
print(f"total rain {total_rain:d}")
print(int("ff", 16), float("  2.5e-3 "), float("inf"))

# @@ checks
assert len(names) == 4
assert temps[1] == 18.25

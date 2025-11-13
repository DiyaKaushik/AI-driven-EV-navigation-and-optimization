import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Select a GUI backend first; fallback to non-GUI if needed
import matplotlib as _mpl
try:
    _mpl.use("TkAgg")
except Exception:
    _mpl.use("Agg")  # headless fallback

import matplotlib.pyplot as plt
from matplotlib import animation
import os
from PIL import Image
import numpy as np
import csv



# -----------------------------
# Simple EV Bus + Charger Sim
# -----------------------------
# This script simulates a few electric buses driving a loop route and charging at
# stations when needed. It animates:
# - Route with bus positions
# - SOC (state of charge) over time for each bus
# - Total charger load over time


# -----------------------------
# Configuration
# -----------------------------
SIM_SECONDS = 60 * 12  # 12 minutes so at least one full loop is visible
DT = 2                 # smaller timestep for smoother curves
RUN_LIVE_ANIMATION = False  # save to file (simulation.gif)
GIF_FRAME_STEP = 3     # subsample modestly to keep GIF size manageable

NUM_BUSES = 3
SEED = 7

# Bus parameters
BATTERY_KWH = 120.0
CONSUMPTION_KWH_PER_KM = 3.0          # energy use while driving (higher so SOC drops sooner)
CRUISE_SPEED_KMH = 30.0               # ~8 minutes per loop on 4 km route
REJOIN_SOC_THRESHOLD = 0.8            # percent SOC required to leave charger
CHARGE_TARGET_SOC = 0.85              # charge until this SOC before leaving

# Initialize buses with lower SOC to demonstrate charging activity
INIT_SOC_RANGE = (0.55, 0.65)         # start slightly above threshold so buses move first

# When to dock (start charging)
DOCK_SOC_THRESHOLD = 0.50             # dock when SOC < 50%

# Bus must be sufficiently near a charger to dock
DOCK_PROXIMITY_KM = 0.10
REQUIRE_PROXIMITY_TO_DOCK = True   # require passing near assigned charger to dock

# Charging parameters
CHARGER_LOCATIONS = [(0.0, 0.0), (1.0, 1.0)]  # two chargers on the loop
PORTS_PER_CHARGER = 2
CHARGING_POWER_KW = 120.0             # power per active port

# Route: simple unit square loop (km coordinates for simplicity)
ROUTE_WAYPOINTS = [
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.0),  # close the loop
]


# -----------------------------
# Utilities
# -----------------------------
def distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.hypot(dx, dy)


def interpolate(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def km_per_second_from_kmh(kmh: float) -> float:
    return kmh / 3600.0


# -----------------------------
# Core Models
# -----------------------------
@dataclass
class ChargingStation:
    position_km: Tuple[float, float]
    num_ports: int
    power_kw: float
    connected_bus_ids: List[int] = field(default_factory=list)

    def has_free_port(self) -> bool:
        return len(self.connected_bus_ids) < self.num_ports

    def connect(self, bus_id: int) -> bool:
        if self.has_free_port() and bus_id not in self.connected_bus_ids:
            self.connected_bus_ids.append(bus_id)
            return True
        return False

    def disconnect(self, bus_id: int) -> None:
        if bus_id in self.connected_bus_ids:
            self.connected_bus_ids.remove(bus_id)

    @property
    def total_load_kw(self) -> float:
        return len(self.connected_bus_ids) * self.power_kw


@dataclass
class Bus:
    bus_id: int
    battery_kwh: float
    consumption_kwh_per_km: float
    cruise_speed_kmh: float
    route_waypoints: List[Tuple[float, float]]
    start_offset_km: float = 0.0

    # dynamic state
    soc_kwh: float = field(init=False)
    route_index: int = field(default=0, init=False)
    segment_progress_km: float = field(default=0.0, init=False)
    charging_station_id: Optional[int] = field(default=None, init=False)
    is_charging: bool = field(default=False, init=False)
    laps_completed: int = field(default=0, init=False)
    charging_started_s: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Start with a configurable SOC range, default low to show charging
        self.soc_kwh = self.battery_kwh * random.uniform(*INIT_SOC_RANGE)
        # Apply initial spatial offset along the loop
        self._apply_start_offset(self.start_offset_km)

    def _apply_start_offset(self, offset_km: float) -> None:
        if offset_km <= 0:
            return
        while offset_km > 0:
            a = self.route_waypoints[self.route_index]
            b = self.route_waypoints[self.route_index + 1]
            seg_len = distance_km(a, b)
            if offset_km >= seg_len:
                self.route_index = (self.route_index + 1) % (len(self.route_waypoints) - 1)
                offset_km -= seg_len
            else:
                self.segment_progress_km = offset_km
                offset_km = 0

    @property
    def soc_fraction(self) -> float:
        return max(0.0, min(1.0, self.soc_kwh / self.battery_kwh))

    @property
    def position_km(self) -> Tuple[float, float]:
        a = self.route_waypoints[self.route_index]
        b = self.route_waypoints[self.route_index + 1]
        seg_len = distance_km(a, b)
        t = 0.0 if seg_len == 0 else self.segment_progress_km / seg_len
        return interpolate(a, b, t)

    def update_drive(self, dt_s: float) -> float:
        if self.is_charging:
            return 0.0
        distance_this_step_km = km_per_second_from_kmh(self.cruise_speed_kmh) * dt_s
        remaining = distance_this_step_km
        traveled = 0.0
        while remaining > 1e-9:
            a = self.route_waypoints[self.route_index]
            b = self.route_waypoints[self.route_index + 1]
            seg_len = distance_km(a, b)
            seg_left = max(0.0, seg_len - self.segment_progress_km)
            step = min(seg_left, remaining)
            self.segment_progress_km += step
            traveled += step
            if abs(self.segment_progress_km - seg_len) < 1e-9 or self.segment_progress_km >= seg_len:
                # advance to next segment (looping)
                prev_index = self.route_index
                self.route_index = (self.route_index + 1) % (len(self.route_waypoints) - 1)
                self.segment_progress_km = 0.0
                # Detect wrap-around to start of loop
                if self.route_index == 0 and prev_index == (len(self.route_waypoints) - 2):
                    self.laps_completed += 1
            remaining -= step
            if traveled >= distance_this_step_km - 1e-9:
                break
        # consume energy based on distance traveled
        energy_used_kwh = traveled * self.consumption_kwh_per_km
        self.soc_kwh = max(0.0, self.soc_kwh - energy_used_kwh)
        return traveled

    def start_charging(self, station_id: int) -> None:
        self.is_charging = True
        self.charging_station_id = station_id

    def stop_charging(self) -> None:
        self.is_charging = False
        self.charging_station_id = None

    def update_charge(self, dt_s: float, station_power_kw: float) -> None:
        if not self.is_charging:
            return
        added_kwh = station_power_kw * (dt_s / 3600.0)
        self.soc_kwh = min(self.battery_kwh, self.soc_kwh + added_kwh)


# -----------------------------
# Simulation Engine
# -----------------------------
class Simulation:
    def __init__(self) -> None:
        random.seed(SEED)

        # Build chargers
        self.chargers: List[ChargingStation] = [
            ChargingStation(position_km=pos, num_ports=PORTS_PER_CHARGER, power_kw=CHARGING_POWER_KW)
            for pos in CHARGER_LOCATIONS
        ]

        # Build buses, staggered along the route
        loop_length = self._loop_length_km(ROUTE_WAYPOINTS)
        # Stagger offsets so no bus starts exactly at a charger location
        base_offsets = np.linspace(0.0, loop_length, NUM_BUSES, endpoint=False)
        phase = loop_length / (NUM_BUSES * 3.0)
        offsets = (base_offsets + phase) % loop_length
        self.buses: List[Bus] = []
        for i in range(NUM_BUSES):
            per_bus_consumption = CONSUMPTION_KWH_PER_KM
            if i == 0:
                # Increase Bus 0 consumption so it hits threshold sooner
                per_bus_consumption = CONSUMPTION_KWH_PER_KM * 1.5
            self.buses.append(
                Bus(
                    bus_id=i,
                    battery_kwh=BATTERY_KWH,
                    consumption_kwh_per_km=per_bus_consumption,
                    cruise_speed_kmh=CRUISE_SPEED_KMH,
                    route_waypoints=ROUTE_WAYPOINTS,
                    start_offset_km=float(offsets[i]),
                )
            )

        # Histories for plotting
        self.time_s: List[int] = []
        self.soc_history: List[List[float]] = [[] for _ in self.buses]  # per bus
        self.total_charger_load_kw: List[float] = []

        # Convenience knobs
        self.rejoin_soc_threshold = REJOIN_SOC_THRESHOLD
        self.charge_target_soc = CHARGE_TARGET_SOC
        self.dock_soc_threshold = DOCK_SOC_THRESHOLD
        # Preferred charger mapping per bus: b1->charger 0, b0->charger 1, b2 special rule
        self.preferred_charger_for_bus = {0: 1, 1: 0}

        self.t_s = 0

    @staticmethod
    def _loop_length_km(points: List[Tuple[float, float]]) -> float:
        return sum(distance_km(points[i], points[i + 1]) for i in range(len(points) - 1))

    def _nearest_charger(self, pos: Tuple[float, float]) -> Tuple[int, float]:
        best_id = 0
        best_d = 1e9
        for i, ch in enumerate(self.chargers):
            d = distance_km(pos, ch.position_km)
            if d < best_d:
                best_d = d
                best_id = i
        return best_id, best_d

    def _consider_charging(self, bus: Bus) -> None:
        # Customized docking rules per user request
        if bus.is_charging:
            return
        target_charger_id: Optional[int] = None

        # Bus 2: complete one whole cycle, then charge at Charger 1 (id 0)
        if bus.bus_id == 2:
            if bus.laps_completed >= 1:
                target_charger_id = 0
            else:
                return  # must finish first lap before docking
        # Bus 1 -> Charger 1 (id 0), Bus 0 -> Charger 2 (id 1)
        elif bus.bus_id in self.preferred_charger_for_bus:
            # Only consider docking when under threshold for these buses
            if bus.soc_fraction >= self.dock_soc_threshold:
                return
            target_charger_id = self.preferred_charger_for_bus[bus.bus_id]
        else:
            # Fallback: original policy
            if bus.soc_fraction < self.dock_soc_threshold:
                target_charger_id, _ = self._nearest_charger(bus.position_km)

        if target_charger_id is None:
            return

        # Optionally require proximity; otherwise allow docking when rule satisfied
        _, dist = self._nearest_charger(bus.position_km)
        if REQUIRE_PROXIMITY_TO_DOCK:
            # Check distance to the specific target charger
            dist_to_target = distance_km(bus.position_km, self.chargers[target_charger_id].position_km)
            if dist_to_target > DOCK_PROXIMITY_KM:
                return

        station = self.chargers[target_charger_id]
        if station.connect(bus.bus_id):
            bus.start_charging(target_charger_id)
            bus.charging_started_s = self.t_s

    def _maybe_leave_charger(self, bus: Bus) -> None:
        if not bus.is_charging:
            return
        # Leave when SOC reaches target threshold (custom for Bus 2).
        # Enforce a 10-second dwell for Bus 0 at Charger 2 (id 1)
        if bus.bus_id == 0 and bus.charging_station_id == 1:
            if bus.charging_started_s is not None and (self.t_s - bus.charging_started_s) < 10:
                return
        # Bus 2 leaves earlier with a smaller target SOC
        bus_target = 0.60 if bus.bus_id == 2 else self.charge_target_soc
        if bus.soc_fraction >= bus_target:
            station = self.chargers[bus.charging_station_id]  # type: ignore[index]
            station.disconnect(bus.bus_id)
            bus.stop_charging()

    def step(self, dt_s: int) -> None:
        # 1) Let charging buses charge
        for bus in self.buses:
            if bus.is_charging and bus.charging_station_id is not None:
                station = self.chargers[bus.charging_station_id]
                bus.update_charge(dt_s, station.power_kw)

        # 2) Let non-charging buses drive and consume energy
        for bus in self.buses:
            if not bus.is_charging:
                bus.update_drive(dt_s)

        # 3) Decide docking/undocking
        for bus in self.buses:
            if bus.is_charging:
                self._maybe_leave_charger(bus)
            else:
                self._consider_charging(bus)

        # 4) Update histories
        self.t_s += dt_s
        self.time_s.append(self.t_s)
        for i, bus in enumerate(self.buses):
            self.soc_history[i].append(bus.soc_fraction * 100.0)
        self.total_charger_load_kw.append(sum(ch.total_load_kw for ch in self.chargers))

    def run_precompute(self, total_frames: int, dt_s: int):
        # Run a fresh simulation copy for deterministic playback
        sim = Simulation()
        times = []
        soc_hist = [[] for _ in sim.buses]
        load_hist = []
        positions = []  # list of [[x,y] for buses]

        for _ in range(total_frames):
            sim.step(dt_s)
            times.append(sim.t_s)
            load_hist.append(sum(ch.total_load_kw for ch in sim.chargers))
            frame_positions = [(b.position_km[0], b.position_km[1]) for b in sim.buses]
            positions.append(frame_positions)
            for i, b in enumerate(sim.buses):
                soc_hist[i].append(b.soc_fraction * 100.0)

        return {
            "times": times,
            "soc_hist": soc_hist,
            "load_hist": load_hist,
            "positions": positions,
        }


# -----------------------------
# Visualization
# -----------------------------
class Visualizer:
    def __init__(self, sim: Simulation, pre: dict) -> None:
        self.sim = sim
        self.pre = pre

        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(2, 3)

        # Axes: route map (big), SOC, charger load
        self.ax_map = self.fig.add_subplot(gs[:, 0:2])
        self.ax_soc = self.fig.add_subplot(gs[0, 2])
        self.ax_load = self.fig.add_subplot(gs[1, 2])

        # Pre-plot route
        route_x = [p[0] for p in ROUTE_WAYPOINTS]
        route_y = [p[1] for p in ROUTE_WAYPOINTS]
        self.ax_map.plot(route_x, route_y, color="#444", lw=2, label="Route")

        # Plot charger locations
        cx = [pos[0] for pos in CHARGER_LOCATIONS]
        cy = [pos[1] for pos in CHARGER_LOCATIONS]
        self.ax_map.scatter(cx, cy, s=80, marker="s", c="orange", edgecolors="k", alpha=0.85, label="Chargers", zorder=3)
        # Charger labels
        for idx, (x, y) in enumerate(CHARGER_LOCATIONS, start=1):
            self.ax_map.text(
                x + 0.02,
                y - 0.06,
                f"Charger {idx}",
                color="black",
                fontsize=8,
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        # Bus scatters
        colors = plt.cm.tab10(np.arange(len(self.sim.buses)))
        self.bus_scat = self.ax_map.scatter(
            [self.pre["positions"][0][i][0] for i in range(len(self.sim.buses))],
            [self.pre["positions"][0][i][1] for i in range(len(self.sim.buses))],
            s=120,
            c=colors,
            label="Buses",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        self.bus_labels = []
        for i in range(len(self.sim.buses)):
            (tx, ty) = (self.pre["positions"][0][i][0], self.pre["positions"][0][i][1])
            label = self.ax_map.text(
                tx + 0.02,
                ty + 0.02,
                f"B{i}",
                color="black",
                fontsize=8,
                weight="bold",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )
            self.bus_labels.append(label)
        self.ax_map.set_aspect("equal", adjustable="box")
        self.ax_map.set_title("Route with Buses and Chargers")
        self.ax_map.legend(loc="upper right")
        self.ax_map.set_xlim(-0.2, 1.2)
        self.ax_map.set_ylim(-0.2, 1.2)
        self.ax_map.grid(True, ls=":", alpha=0.3)

        # SOC lines, one per bus
        self.soc_lines = []
        for i, _ in enumerate(self.sim.buses):
            (line,) = self.ax_soc.plot([], [], color=colors[i], lw=2, label=f"Bus {i}")
            self.soc_lines.append(line)
        self.ax_soc.set_title("SOC (%)")
        self.ax_soc.set_ylim(0, 100)
        self.ax_soc.set_xlim(0, SIM_SECONDS)
        self.ax_soc.grid(True, ls=":", alpha=0.3)
        self.ax_soc.legend(loc="lower left", fontsize=8)

        # Charger load line (kW)
        (self.load_line,) = self.ax_load.plot([], [], color="purple", lw=2)
        self.ax_load.set_title("Total Charger Load (kW)")
        self.ax_load.set_ylim(0, len(self.sim.chargers) * PORTS_PER_CHARGER * CHARGING_POWER_KW)
        self.ax_load.set_xlim(0, SIM_SECONDS)
        self.ax_load.grid(True, ls=":", alpha=0.3)

        self.fig.tight_layout()

    def init_anim(self):
        return (*self.soc_lines, self.load_line, self.bus_scat)

    def update(self, frame_idx: int):
        # Use precomputed data for stable animation
        bx = [self.pre["positions"][frame_idx][i][0] for i in range(len(self.sim.buses))]
        by = [self.pre["positions"][frame_idx][i][1] for i in range(len(self.sim.buses))]
        self.bus_scat.set_offsets(np.c_[bx, by])
        for i, label in enumerate(self.bus_labels):
            label.set_position((bx[i] + 0.02, by[i] + 0.02))

        x = self.pre["times"][: frame_idx + 1]
        for i, line in enumerate(self.soc_lines):
            y = self.pre["soc_hist"][i][: frame_idx + 1]
            line.set_data(x, y)

        self.load_line.set_data(x, self.pre["load_hist"][: frame_idx + 1])

        return (*self.soc_lines, self.load_line, self.bus_scat)


def main() -> None:
    sim = Simulation()
    total_frames = SIM_SECONDS // DT
    pre = sim.run_precompute(total_frames, DT)
    # Export metrics to CSV for analysis
    try:
        with open("metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["time_s", "total_load_kw"] + [f"bus_{i}_soc_pct" for i in range(len(sim.buses))]
            writer.writerow(header)
            for idx, t in enumerate(pre["times"]):
                row = [t, pre["load_hist"][idx]] + [pre["soc_hist"][i][idx] for i in range(len(sim.buses))]
                writer.writerow(row)
        print("Saved metrics to metrics.csv")
    except Exception as e:
        print("Failed to write metrics.csv:", e)
    vis = Visualizer(sim, pre)
    # If exporting to GIF, subsample frames to reduce file size and playback issues
    frame_iter = range(0, total_frames, GIF_FRAME_STEP) if not RUN_LIVE_ANIMATION else total_frames
    anim_obj = animation.FuncAnimation(
        vis.fig,
        vis.update,
        init_func=vis.init_anim,
        frames=frame_iter,
        interval=100,  # ms per frame (visual only, slower playback)
        blit=RUN_LIVE_ANIMATION,  # turn blitting on for smoother live playback
        repeat=False,
        save_count=total_frames,
    )
    # Always save a static snapshot for easy viewing
    try:
        vis.update(0)
        vis.fig.savefig("snapshot.png", dpi=130, facecolor="white")
        print("Saved snapshot.png")
    except Exception as e:
        print("Failed to save snapshot.png:", e)
    # Always save GIF for file-based animation
    try:
        from matplotlib.animation import PillowWriter
        gif_path = "simulation.gif"
        writer = PillowWriter(fps=5, metadata={"title": "Bus Charging Simulation"})
        anim_obj.save(gif_path, writer=writer, dpi=120, savefig_kwargs={"facecolor": "white"})
        print(f"Saved animation to {gif_path}")
    except Exception as exc3:
        print("GIF save failed. Saving PNG frames to 'frames'.")
        os.makedirs("frames", exist_ok=True)
        step = max(1, total_frames // 100)
        plt.close(vis.fig)
        sim2 = Simulation()
        pre2 = sim2.run_precompute(total_frames, DT)
        vis2 = Visualizer(sim2, pre2)
        for idx in range(0, total_frames, step):
            vis2.update(idx)
            vis2.fig.savefig(os.path.join("frames", f"frame_{idx:04d}.png"), dpi=130)
        plt.close(vis2.fig)
        print("Saved PNG frame sequence in 'frames/' directory.")
    else:
        try:
            plt.show()
        except Exception as exc:
            print("GUI display failed. Falling back to saving GIF.")
            try:
                from matplotlib.animation import PillowWriter
                output_path = "simulation.gif"
                anim_obj.save(output_path, writer=PillowWriter(fps=max(1, int(1000/50))), dpi=120)
                print(f"Saved animation to {output_path}")
            except Exception as exc2:
                print("Failed to save GIF. Consider installing 'pillow'. Error:", exc2)


if __name__ == "__main__":
    main()



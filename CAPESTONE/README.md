# Electric Bus and Charging Station Simulation

## Overview

This Python simulation models electric buses operating on a loop route with charging stations. It visualizes bus movement, State of Charge (SOC), and charging station load over time using matplotlib animations.

## Features Implemented

### 1. **Core Simulation Components**

#### Bus Model (`Bus` class)
- **Battery capacity**: 120 kWh per bus
- **Energy consumption**: 3.0 kWh/km (Bus 0 uses 1.5× = 4.5 kWh/km for faster depletion)
- **Cruise speed**: 30 km/h
- **Route tracking**: Follows a square loop route with waypoints
- **SOC management**: Tracks battery state of charge as buses drive and charge
- **Lap counting**: Tracks completed loops around the route

#### Charging Station Model (`ChargingStation` class)
- **Two charging stations**: 
  - Charger 1 at position (0.0, 0.0)
  - Charger 2 at position (1.0, 1.0)
- **Ports per charger**: 2 ports each (4 total charging ports)
- **Charging power**: 120 kW per active port
- **Connection management**: Tracks which buses are connected

### 2. **Bus Assignment Rules**

Each bus has specific charging behavior:

- **Bus 0 (Blue)**:
  - Assigned to **Charger 2** (position 1.0, 1.0)
  - Higher consumption (4.5 kWh/km) to reach threshold faster
  - Must charge for **minimum 10 seconds** at Charger 2 before leaving
  - Docks when SOC < 50%

- **Bus 1 (Orange)**:
  - Assigned to **Charger 1** (position 0.0, 0.0)
  - Standard consumption (3.0 kWh/km)
  - Docks when SOC < 50%

- **Bus 2 (Green)**:
  - Must complete **one full loop** before charging
  - Then charges at **Charger 1** (position 0.0, 0.0)
  - Leaves charger earlier at **60% SOC** (vs 85% for others)
  - Standard consumption (3.0 kWh/km)

### 3. **Simulation Parameters**

```python
SIM_SECONDS = 60 * 12      # 12 minutes total simulation time
DT = 2                      # 2-second time steps
NUM_BUSES = 3               # Three buses in simulation
BATTERY_KWH = 120.0         # Battery capacity per bus
CONSUMPTION_KWH_PER_KM = 3.0  # Energy consumption (Bus 0 uses 4.5)
CRUISE_SPEED_KMH = 30.0     # Driving speed
DOCK_SOC_THRESHOLD = 0.50   # Dock when SOC < 50%
CHARGE_TARGET_SOC = 0.85    # Charge until 85% (Bus 2 leaves at 60%)
REQUIRE_PROXIMITY_TO_DOCK = True  # Must be near charger to dock
DOCK_PROXIMITY_KM = 0.20    # Within 200m of charger to dock
```

### 4. **Visualization Features**

The animation displays three panels:

#### **Route Map (Left Panel)**
- Shows the square loop route (4 km total)
- Orange squares mark charging station locations
- Colored circles represent buses with labels (B0, B1, B2)
- Bus colors: Blue (B0), Orange (B1), Green (B2)
- Real-time position updates as buses move

#### **SOC Panel (Top Right)**
- Line graph showing State of Charge (%) over time
- One line per bus (color-coded)
- Shows:
  - **Downward slopes**: Buses driving and consuming energy
  - **Upward slopes**: Buses charging at stations
  - **Plateaus**: Buses waiting or at full charge

#### **Total Charger Load Panel (Bottom Right)**
- Line graph showing total power consumption (kW) across all chargers
- Calculated as: `(number of connected buses) × 120 kW`
- Shows:
  - **0 kW**: No buses charging
  - **120 kW**: One bus charging
  - **240 kW**: Two buses charging
  - **240+ kW**: Multiple buses charging simultaneously

### 5. **Output Files**

#### `simulation.gif`
- Animated visualization of the entire simulation
- Shows route, SOC, and charger load evolving over time
- Frame rate: 5 fps
- Duration: ~12 minutes of simulation time compressed into animation

#### `snapshot.png`
- Static image of the initial simulation state
- Useful for quick reference or documentation

#### `metrics.csv`
- Time-series data export
- Columns:
  - `time_s`: Simulation time in seconds
  - `total_load_kw`: Total charger load at that time
  - `bus_0_soc_pct`: Bus 0 SOC percentage
  - `bus_1_soc_pct`: Bus 1 SOC percentage
  - `bus_2_soc_pct`: Bus 2 SOC percentage
- Can be imported into Excel, Python, or other analysis tools

## How the Simulation Works

### Simulation Loop

1. **Initialization**:
   - Buses start at random positions along the route
   - Initial SOC: 55-65% (above docking threshold)
   - All buses begin driving

2. **Each Time Step (2 seconds)**:
   - **Charging buses**: Add energy based on charger power (120 kW)
   - **Driving buses**: Move along route and consume energy
   - **Docking decisions**: Check if buses should start charging
   - **Undocking decisions**: Check if charging buses should leave

3. **Docking Logic**:
   - Bus 0 & 1: Dock when SOC < 50% AND near assigned charger
   - Bus 2: Dock only after completing 1 lap AND near Charger 1

4. **Undocking Logic**:
   - Bus 0: Leave Charger 2 after 10 seconds minimum AND SOC ≥ 85%
   - Bus 1: Leave Charger 1 when SOC ≥ 85%
   - Bus 2: Leave Charger 1 when SOC ≥ 60%

### Energy Calculations

- **Energy consumed while driving**: `distance_km × consumption_kwh_per_km`
- **Energy added while charging**: `120 kW × (time_seconds / 3600)`
- **SOC calculation**: `(current_energy_kwh / battery_capacity_kwh) × 100%`

## Interpreting the Output

### What to Look For

1. **Bus Movement**:
   - Buses should move continuously around the loop
   - When they stop, they're charging (positioned at a charger)

2. **SOC Patterns**:
   - **Declining lines**: Normal operation, energy being consumed
   - **Rising lines**: Charging in progress
   - **Rapid drops**: High consumption periods (Bus 0 drops fastest)
   - **Cycles**: Multiple charge/discharge cycles indicate sustainable operation

3. **Charger Load**:
   - **Spikes**: Multiple buses charging simultaneously
   - **Sustained periods**: Long charging sessions
   - **Zero periods**: All buses driving, no charging activity

### Expected Behavior

- **Bus 0**: Should reach Charger 2 first (due to higher consumption), charge for at least 10 seconds
- **Bus 1**: Should reach Charger 1 when SOC drops, charge to 85%
- **Bus 2**: Should complete one full loop before charging, then charge at Charger 1 to 60%

### Performance Metrics

From `metrics.csv`, you can analyze:
- **Average SOC**: Overall battery health
- **Charging frequency**: How often buses need to charge
- **Peak charger load**: Maximum simultaneous charging demand
- **Energy efficiency**: Distance traveled per kWh consumed

## Technical Details

### Dependencies
- `matplotlib`: Visualization and animation
- `numpy`: Numerical calculations
- `PIL` (Pillow): Image processing for GIF export
- `csv`: Data export

### Animation Settings
- **Backend**: TkAgg (with Agg fallback)
- **Frame interval**: 100 ms per frame
- **GIF FPS**: 5 frames per second
- **DPI**: 120 for GIF, 130 for snapshot

### Route Geometry
- Square loop: (0,0) → (1,0) → (1,1) → (0,1) → (0,0)
- Total distance: ~4 km per loop
- At 30 km/h: ~8 minutes per complete loop

## Usage

```bash
python main.py
```

The script will:
1. Run the simulation (pre-compute all frames)
2. Generate `metrics.csv` with time-series data
3. Create `snapshot.png` with initial state
4. Export `simulation.gif` with full animation

## Customization

Key parameters to adjust in `main.py`:

- **Simulation duration**: Change `SIM_SECONDS`
- **Bus speed**: Modify `CRUISE_SPEED_KMH`
- **Energy consumption**: Adjust `CONSUMPTION_KWH_PER_KM`
- **Charging thresholds**: Modify `DOCK_SOC_THRESHOLD` and `CHARGE_TARGET_SOC`
- **Charger locations**: Update `CHARGER_LOCATIONS`
- **Number of buses**: Change `NUM_BUSES` (update colors accordingly)

## Notes

- The simulation uses deterministic physics (no randomness after initialization)
- All buses follow the same route but with different timing
- Charging stations have limited ports (2 per station)
- Proximity requirement ensures realistic docking behavior
- Pre-computation ensures smooth, glitch-free animation


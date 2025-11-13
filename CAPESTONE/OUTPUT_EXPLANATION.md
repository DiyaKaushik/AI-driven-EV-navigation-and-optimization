# Output Explanation: What Each File Shows

## 1. simulation.gif - Animated Visualization

This animated GIF shows the simulation running over 12 minutes of simulated time. It has three panels:

### Left Panel: Route Map
**What you see:**
- A square loop route (black line connecting 4 corners)
- Two orange squares: Charger 1 (bottom-left at 0,0) and Charger 2 (top-right at 1,1)
- Three colored circles moving around the route:
  - **Blue circle (B0)**: Bus 0
  - **Orange circle (B1)**: Bus 1  
  - **Green circle (B2)**: Bus 2

**What it means:**
- **Moving circles**: Buses are driving along the route, consuming battery energy
- **Stopped circles at orange squares**: Buses are charging at that station
- **Movement direction**: Buses travel clockwise around the square loop
- **Position changes**: Each frame shows buses advancing along the route

**Key observations:**
- When a bus stops moving, it's connected to a charger and charging
- Bus 0 (blue) should stop at Charger 2 (top-right orange square)
- Bus 1 (orange) should stop at Charger 1 (bottom-left orange square)
- Bus 2 (green) completes one full loop before stopping at Charger 1

---

### Top Right Panel: SOC (State of Charge) Graph
**What you see:**
- Three colored lines over time (matching bus colors)
- X-axis: Time in seconds (0 to 720 seconds = 12 minutes)
- Y-axis: SOC percentage (0% to 100%)

**What it means:**
- **Downward sloping lines**: Bus is driving and losing battery charge
  - Steeper slope = faster energy consumption
  - Bus 0 (blue) should drop fastest (uses 1.5× more energy)
  
- **Upward sloping lines**: Bus is charging at a station
  - Steeper slope = faster charging (120 kW per port)
  - All buses charge at same rate when connected
  
- **Horizontal/flat lines**: 
  - At 100% = Battery fully charged
  - At 0% = Battery empty (shouldn't happen in this simulation)
  - Brief flat = Transition between driving and charging

**Key patterns to observe:**
- **Cycles**: Each bus should show multiple charge/discharge cycles
- **Bus 0**: Drops quickly, charges at Charger 2, must stay 10+ seconds
- **Bus 1**: Moderate drops, charges at Charger 1 to 85%
- **Bus 2**: Completes one loop before first charge, then charges to only 60%

**Example interpretation:**
- If you see Bus 0's line drop from 60% to 40%, it drove and consumed 20% battery
- If it then rises from 40% to 85%, it charged for a period of time
- The time between drops shows how long the bus can drive before needing to charge

---

### Bottom Right Panel: Total Charger Load (kW)
**What you see:**
- One purple line over time
- X-axis: Time in seconds (0 to 720)
- Y-axis: Total power in kilowatts (0 to 480 kW max)

**What it means:**
- **0 kW**: No buses are charging (all driving)
- **120 kW**: One bus is charging (one port active)
- **240 kW**: Two buses charging simultaneously (two ports active)
- **360 kW**: Three buses charging (three ports active)
- **480 kW**: All four ports active (maximum capacity)

**Key observations:**
- **Spikes/peaks**: Multiple buses charging at the same time
- **Sustained periods**: Long charging sessions (e.g., Bus 0's 10-second minimum)
- **Zero periods**: All buses driving, no charging activity
- **Step changes**: Sudden jumps up = bus connects, sudden drops = bus disconnects

**Example interpretation:**
- Line jumps from 0 to 120 kW: One bus just connected to a charger
- Line jumps from 120 to 240 kW: Another bus started charging
- Line drops from 240 to 120 kW: One bus finished charging and left
- Line drops to 0 kW: All buses finished charging and are driving

**What this tells you:**
- **Peak demand**: Maximum simultaneous charging power needed
- **Charging frequency**: How often buses need to charge
- **Station utilization**: How many ports are in use over time
- **Load balancing**: Whether chargers are evenly used

---

## 2. snapshot.png - Static Initial State

**What you see:**
- A frozen frame showing the simulation at time = 0 seconds
- Same three panels as the GIF, but showing only the starting state

**What it means:**
- **Route map**: Shows initial positions of all three buses
- **SOC graph**: Shows starting battery levels (55-65% range)
- **Charger load**: Shows 0 kW (no buses charging initially)

**Purpose:**
- Quick reference for initial conditions
- Useful for presentations or reports
- Shows the setup before simulation starts

---

## 3. metrics.csv - Time-Series Data

**What it contains:**
A table with one row per time step (every 2 seconds), with columns:

| Column | Description | Example Values |
|--------|-------------|----------------|
| `time_s` | Simulation time in seconds | 0, 2, 4, 6, ... 720 |
| `total_load_kw` | Total charger power consumption | 0, 120, 240, 360 |
| `bus_0_soc_pct` | Bus 0 SOC percentage | 58.3, 57.1, 55.9, ... |
| `bus_1_soc_pct` | Bus 1 SOC percentage | 62.4, 61.2, 60.0, ... |
| `bus_2_soc_pct` | Bus 2 SOC percentage | 60.1, 59.9, 59.7, ... |

**What you can analyze:**

1. **Energy Consumption Rate**
   - Calculate: `(SOC_start - SOC_end) / time` for driving periods
   - Compare consumption between buses (Bus 0 should be highest)

2. **Charging Time**
   - Find rows where SOC increases
   - Count time steps to see how long each charge session lasts
   - Bus 0 should show minimum 5 rows (10 seconds) at Charger 2

3. **Charging Frequency**
   - Count how many times each bus's SOC starts increasing
   - Shows how often buses need to charge

4. **Peak Load Analysis**
   - Find maximum value in `total_load_kw` column
   - Shows peak simultaneous charging demand

5. **Average SOC**
   - Calculate mean of all SOC columns
   - Shows overall battery health and system sustainability

6. **Charging Efficiency**
   - Compare energy added during charging vs. energy consumed while driving
   - Verify buses can sustain operation

**Example data interpretation:**
```
time_s, total_load_kw, bus_0_soc_pct, bus_1_soc_pct, bus_2_soc_pct
0,      0,             58.3,          62.4,          60.1
2,      0,             57.1,          61.2,          59.9
4,      0,             55.9,          60.0,          59.7
...
120,    120,           45.2,          55.1,          58.5  ← Bus 0 started charging
122,    120,           46.8,          54.3,          58.3  ← Bus 0 charging, Bus 1 driving
124,    240,           48.4,          53.5,          58.1  ← Both Bus 0 and Bus 1 charging
```

**How to use this data:**
- Import into Excel/Python for statistical analysis
- Create custom graphs beyond the animation
- Calculate metrics like average charging time, energy efficiency
- Export for reports or further analysis
- Compare different simulation scenarios

---

## Summary: What to Look For

### ✅ Good Signs (Healthy Simulation):
- All three buses moving continuously around the route
- SOC lines showing regular charge/discharge cycles
- Charger load varying between 0-240 kW (showing activity)
- No bus dropping below 20% SOC (safety margin)
- Buses following their assigned charger rules

### ⚠️ Issues to Watch For:
- Bus stuck at 0% SOC (battery depleted - shouldn't happen)
- Charger load always at 0 (buses never charging - check thresholds)
- Bus never moving (stuck at charger - check undocking logic)
- SOC lines flat at 100% (overcharging - check target SOC)
- All buses charging simultaneously for long periods (may indicate high consumption)

### 📊 Key Metrics to Extract:
1. **Average time between charges** for each bus
2. **Peak simultaneous charging demand** (max total_load_kw)
3. **Charging station utilization** (% of time ports are in use)
4. **Energy efficiency** (distance per kWh consumed)
5. **SOC range** (min/max for each bus over simulation)


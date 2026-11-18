"""
guelph_run.py
=============
Full simulation using REAL Guelph Transit GTFS data.
Bus: Nova Bus LFSe+ (actual Guelph fleet)
Route: Route 17 Woodlawn-Watson (32.65 km, 71 stops)

Realistic parameters sourced from:
- Nova Bus LFSe+: 564 kWh battery, 340-470 km range
- Guelph Transit: 87 buses total, 11 electric delivered/planned
- En-route charging: Guelph Central Station fast chargers (planned 2026)
- Ontario electricity: peak ~$0.17/kWh, off-peak ~$0.082/kWh (OEB 2024)

Fleet sizing rationale:
  Route 17 is 32.65 km. At 35 km/h average with 71 stops it takes ~65 min.
  Headway on Route 17 = 30 min peak -> 3 buses active simultaneously.
  We simulate N=10 buses (3 active + spares/layovers) = REALISTIC.
  Chargers: 4 en-route (Guelph Transit plan includes on-route chargers at
  Central Station and key terminals = realistic for this corridor).
"""

import os, sys, random, time, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, ttest_ind
from collections import defaultdict

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)

sys.path.insert(0, '/home/claude/project')
from gtfs_loader import load_route_from_gtfs

OUT = '/mnt/user-data/outputs/ieee_ev_results'
os.makedirs(OUT, exist_ok=True)

# ── LOAD REAL GTFS ────────────────────────────────────────────────────────────
print('='*65)
print('  Loading REAL Guelph Transit GTFS — Route 17 Woodlawn-Watson')
print('='*65)

gtfs_route = load_route_from_gtfs(
    shapes_path  = '/mnt/user-data/uploads/shapes.txt',
    routes_path  = '/mnt/user-data/uploads/routes.txt',
    trips_path   = '/mnt/user-data/uploads/trips.txt',
    route_short_name = '17',
    n_stops      = 71,
    n_chargers   = 4,
)

WAYPOINTS  = np.array(gtfs_route.waypoints, dtype=float)
ROUTE_KM   = gtfs_route.total_km          # 32.65 km
STOP_POS   = gtfs_route.stop_positions
CHPOS      = gtfs_route.charger_positions
ROUTE_NAME = 'Guelph Transit Route 17 Woodlawn-Watson (32.65 km)'

print(f'\n  Route: {ROUTE_NAME}')
print(f'  {len(WAYPOINTS)} waypoints | {len(STOP_POS)} stops | {len(CHPOS)} chargers')

_segs = np.linalg.norm(np.diff(WAYPOINTS, axis=0), axis=1)
_cum  = np.concatenate([[0.], np.cumsum(_segs)])

def route_pos(d_km):
    d = d_km % ROUTE_KM
    i = min(int(np.searchsorted(_cum, d, 'right'))-1, len(WAYPOINTS)-2)
    f = (d - _cum[i]) / max(_segs[i], 1e-9)
    p = WAYPOINTS[i] + f*(WAYPOINTS[i+1]-WAYPOINTS[i])
    return float(p[0]), float(p[1])

# ── REALISTIC CONFIG  (Nova Bus LFSe+ + Guelph Transit) ──────────────────────
class Cfg:
    SEED       = 42
    DT         = 15           # seconds
    WIN        = 480          # 8-hour shift (Guelph peak service window)
    NUM_BUSES  = 10           # REALISTIC: Route 17 runs ~3 buses/headway + spares
    BUS_CAP    = 60           # Nova LFS standard capacity
    # Nova Bus LFSe+: 564 kWh usable
    BAT_KWH    = 564.0        # kWh — REAL spec
    # Energy consumption: ~1.2 kWh/km for 40-ft transit bus in Ontario climate
    KWH_KM     = 1.2
    REGEN      = 0.10         # regenerative braking recovery
    CRUISE_KMH = 35.0         # Guelph Transit average speed
    CRIT_SOC   = 0.20
    SOC_FLOOR  = 0.22         # hard action mask
    DOCK_SOC   = 0.40
    TARG_SOC   = 0.85
    INIT_LO    = 0.45         # start of shift: partly charged
    INIT_HI    = 0.80
    # En-route OppCharge: pantograph fast charger
    CHG_KW     = 150.0        # 150 kW OppCharge (realistic for on-route)
    CHG_EFF    = 0.93
    PLUGIN_S   = 60           # pantograph: faster connect than plug
    UNPLUG_S   = 45
    STOPS      = 71
    BOARD_RATE = 3.0          # Guelph is smaller city, lower ridership
    DWELL_BASE = 25           # seconds at stop
    DWELL_PAX  = 2.0
    LOAD_CONS  = 0.20
    # Ontario electricity prices (OEB 2024)
    PRICE_PEAK = 0.170        # $/kWh Ontario on-peak
    PRICE_OFF  = 0.082        # $/kWh Ontario off-peak
    PEAK_H     = {7,8,9,16,17,18}
    # Reduced state space
    SOC_B      = 10
    DIST_B     = 6
    TIME_B     = 6
    PAX_B      = 3
    QUEUE_B    = 3
    UTIL_B     = 3
    ALPHA      = 0.15
    GAMMA      = 0.95
    EPISODES   = 1500
    EPS0       = 1.0
    EPS_MIN    = 0.01
    EPS_DEC    = (0.01/1.0)**(1/1500)
    PAX_W      = 2.0
    IDLE_W     = 0.5
    N_MC       = 30
    DEMAND = {7:1.6,8:1.9,9:1.5,10:.9,11:.8,12:1.0,
              13:1.1,14:.85,15:.9,16:1.3,17:1.8,18:1.6,19:1.0,20:.7}

cfg = Cfg()

# Sanity check: with 564 kWh and 1.2 kWh/km, range = 470 km (matches spec)
range_km = cfg.BAT_KWH / cfg.KWH_KM
print(f'\n  Sanity: 564kWh / 1.2kWh/km = {range_km:.0f}km range (Nova spec: 340-470km ✓)')
print(f'  State space: {cfg.SOC_B}×{cfg.DIST_B}×{cfg.TIME_B}×{cfg.PAX_B}×{cfg.QUEUE_B}×{cfg.UTIL_B}',
      f'= {cfg.SOC_B*cfg.DIST_B*cfg.TIME_B*cfg.PAX_B*cfg.QUEUE_B*cfg.UTIL_B:,} states')
print(f'  Charger util target: 40-65% (realistic for 10 buses, 4 chargers, 32km route)')

# ── IEEE FIGURE STYLE ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'figure.dpi':300,'axes.linewidth':0.8,'lines.linewidth':1.4,
    'axes.grid':True,'grid.alpha':0.3,'grid.linewidth':0.5,
    'axes.spines.top':False,'axes.spines.right':False,
})
COLORS = {'baseline':'#2c3e50','heuristic':'#e67e22',
          'rl_tabular':'#8e44ad','rl_coordinated':'#27ae60'}
LABELS = {'baseline':'Baseline','heuristic':'Heuristic',
          'rl_tabular':'RL-Tabular','rl_coordinated':'RL-Coordinated'}
STRATEGIES = ['baseline','heuristic','rl_tabular','rl_coordinated']

def _sv(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'    saved {name}')

# ── Q-AGENT ───────────────────────────────────────────────────────────────────
class QAgent:
    MAX_D = 20.0
    def __init__(self):
        shape=(cfg.SOC_B,cfg.DIST_B,cfg.TIME_B,cfg.PAX_B,cfg.QUEUE_B,cfg.UTIL_B,2)
        self.Q=np.zeros(shape,dtype=np.float64)
        self._v=np.zeros(shape[:-1],dtype=np.int64)
        self.eps=cfg.EPS0

    def _disc(self,soc,dist,tf,pf,q,cu):
        b=lambda v,n: min(n-1,max(0,int(v*n)))
        return (b(soc,cfg.SOC_B),b(dist/self.MAX_D,cfg.DIST_B),
                b(tf,cfg.TIME_B),b(pf,cfg.PAX_B),
                b(q/max(cfg.QUEUE_B,1),cfg.QUEUE_B),b(cu,cfg.UTIL_B))

    def act(self,soc,dist,tf,pf,q=0.,cu=0.,force_charge=False):
        if force_charge or soc<cfg.SOC_FLOOR: return 1
        if random.random()<self.eps: return random.randint(0,1)
        return int(np.argmax(self.Q[self._disc(soc,dist,tf,pf,q,cu)]))

    def greedy(self,soc,dist=8.,tf=.4,pf=.5,q=0.,cu=0.):
        if soc<cfg.SOC_FLOOR: return 1
        return int(np.argmax(self.Q[self._disc(soc,dist,tf,pf,q,cu)]))

    def update(self,s,d,t,p,q,cu,a,r,ns,nd,nt,np_,nq,ncu):
        st=self._disc(s,d,t,p,q,cu); nst=self._disc(ns,nd,nt,np_,nq,ncu)
        self._v[st]+=1
        td=r+cfg.GAMMA*float(np.max(self.Q[nst]))-self.Q[st+(a,)]
        self.Q[st+(a,)]+=cfg.ALPHA*td

    def decay(self): self.eps=max(cfg.EPS_MIN,self.eps*cfg.EPS_DEC)
    def total_states(self): return int(np.prod(self.Q.shape[:-1]))
    def states_visited(self): return int(np.count_nonzero(self._v))
    def coverage(self): return self.states_visited()/max(self.total_states(),1)
    def clone(self):
        a=QAgent(); a.Q=self.Q.copy(); a._v=self._v.copy(); a.eps=0.0; return a

class FixedAgent:
    def __init__(self,a): self._a=a; self.eps=0.
    def act(self,*a,**k): return self._a
    def update(self,*a,**k): pass
    def decay(self): pass
    def clone(self): return FixedAgent(self._a)

class RandomAgent:
    eps=1.0
    def act(self,*a,**k): return random.randint(0,1)
    def update(self,*a,**k): pass
    def decay(self): pass
    def clone(self): return RandomAgent()

# ── REWARD ────────────────────────────────────────────────────────────────────
def compute_reward(soc,action,is_peak,queue_len,pax_frac,cu,pax_served,bl_pax):
    C,D,T=cfg.CRIT_SOC,cfg.DOCK_SOC,cfg.TARG_SOC
    if action==1:
        urg=(T-soc)/max(T-C,1e-9)
        r=5.0*urg-1.0
        if is_peak: r-=1.5
        r-=0.5*min(queue_len,4)-0.3*cu+0.3*pax_frac
        if soc>D+0.15:
            r-=cfg.IDLE_W*(soc-D-0.15)/(1.0-D-0.15+1e-9)
    else:
        if   soc<C: r=-8.0*abs(soc-C)/max(C,1e-9)
        elif soc<D: r=-1.0*(D-soc)/max(D-C,1e-9)
        else: r=2.0*(soc-D)/max(T-D,1e-9)
        r+=0.2*pax_frac
    if bl_pax>0:
        r-=cfg.PAX_W*max(0.,1.0-pax_served/bl_pax)
    return float(r)

# ── SIMULATION ────────────────────────────────────────────────────────────────
class Charger:
    def __init__(self,cid,pos,kw=None):
        self.cid=cid; self.pos=pos; self.kw=kw or cfg.CHG_KW
        self.occupied=0; self.queue=[]; self.kwh=0.
        self._busy=0; self._steps=0
    @property
    def queue_len(self): return len(self.queue)
    @property
    def util(self): return self._busy/max(self._steps,1)
    def tick(self): self._steps+=1; self._busy+=(1 if self.occupied>0 else 0)
    def request(self,bid):
        if self.occupied<1: self.occupied=1; return True
        self.queue.append(bid); return False
    def release(self):
        self.occupied=0
        if self.queue: nxt=self.queue.pop(0); self.occupied=1; return nxt
        return None

class Bus:
    def __init__(self,bid,soc0,rng):
        self.bid=bid; self.soc=float(np.clip(soc0,0,1))
        self.state='running'; self.dist=float(rng.uniform(0,ROUTE_KM))
        self.pax=int(rng.integers(0,cfg.BUS_CAP//3))
        self.cid=None; self.timer=0.
        self.pax_served=0; self.wait_s=0.; self.energy_cost=0.
        self.emergencies=0; self.pos=route_pos(self.dist)
    @property
    def pax_frac(self): return self.pax/max(cfg.BUS_CAP,1)

class SimEnv:
    def __init__(self,strategy,agent=None,seed=42,bl_pax=None,
                 train_mode=False,n_buses=None,chpos=None,chg_kw=None):
        self.strategy=strategy; self.agent=agent
        self.train_mode=train_mode; self.rng=np.random.default_rng(seed)
        self.bl_pax=bl_pax
        self.total_steps=int(cfg.WIN*60/cfg.DT)
        self.t=0
        _cp=list(chpos or CHPOS); _kw=chg_kw or cfg.CHG_KW
        self.chargers=[Charger(i,p,_kw) for i,p in enumerate(_cp)]
        n=n_buses or cfg.NUM_BUSES
        socs=self.rng.uniform(cfg.INIT_LO,cfg.INIT_HI,n)
        self.buses=[Bus(i,socs[i],self.rng) for i in range(n)]
        self._rewards=[]; self._pax=0; self._grid=[]

    def _hour(self): return 7.0+self.t*cfg.DT/3600.
    def _peak(self): return int(self._hour()) in cfg.PEAK_H
    def _demand(self): return cfg.DEMAND.get(int(self._hour()),1.0)
    def _speed(self): return self.rng.uniform(.8,1.2)*(0.6 if self.rng.random()<.12 else 1.)

    def _nearest(self,bus):
        bd,bc,bq,bcu=1e9,self.chargers[0],0,0.
        for ch in self.chargers:
            d=np.hypot(bus.pos[0]-ch.pos[0],bus.pos[1]-ch.pos[1])
            if d<bd: bd,bc,bq,bcu=d,ch,ch.queue_len,ch.util
        return bd,bc,bq,bcu

    def _best_charger(self,bus):
        if self.strategy=='rl_coordinated':
            bs,bc=1e9,self.chargers[0]
            for ch in self.chargers:
                d=np.hypot(bus.pos[0]-ch.pos[0],bus.pos[1]-ch.pos[1])
                s=d+2.*ch.queue_len
                if s<bs: bs,bc=s,ch
            return bc
        _,bc,_,_=self._nearest(bus); return bc

    def _want_charge(self,bus):
        if bus.soc<cfg.SOC_FLOOR: return True
        if self.strategy=='baseline': return bus.soc<cfg.DOCK_SOC
        if self.strategy=='heuristic':
            return (bus.soc<cfg.CRIT_SOC+.05) or (bus.soc<cfg.DOCK_SOC and not self._peak())
        if self.agent is None: return bus.soc<cfg.DOCK_SOC
        d,_,q,cu=self._nearest(bus)
        tf=self.t/max(self.total_steps,1)
        return self.agent.act(bus.soc,d,tf,bus.pax_frac,q,cu)==1

    def step(self):
        peak=self._peak(); dem=self._demand(); step_kw=0.
        for b in self.buses:
            d_pre,_,q_pre,cu_pre=self._nearest(b)
            tf_pre=self.t/max(self.total_steps,1)
            if b.state=='running':
                want=self._want_charge(b); act=1 if want else 0
                if want:
                    ch=self._best_charger(b)
                    got=ch.request(b.bid); b.cid=ch.cid
                    b.state='plugging' if got else 'waiting'
                    b.timer=cfg.PLUGIN_S if got else 0.
                else:
                    sf=self._speed()
                    km=cfg.CRUISE_KMH*sf*cfg.DT/3600.
                    kwh_km=cfg.KWH_KM*(1+cfg.LOAD_CONS*b.pax_frac)
                    b.soc=float(np.clip(b.soc-kwh_km*km/cfg.BAT_KWH,0,1))
                    b.dist=(b.dist+km)%ROUTE_KM; b.pos=route_pos(b.dist)
                    if any(abs(b.dist-sp)<km*.5 for sp in STOP_POS):
                        board=int(self.rng.poisson(cfg.BOARD_RATE*dem))
                        alight=int(self.rng.integers(0,max(b.pax//3+1,1)))
                        b.pax=max(0,min(cfg.BUS_CAP,b.pax-alight+board))
                        b.pax_served+=board; self._pax+=board
                    if b.soc<cfg.CRIT_SOC: b.emergencies+=1
                bl=self.bl_pax or max(self._pax,1)
                r=compute_reward(b.soc,act,peak,q_pre,b.pax_frac,cu_pre,self._pax,bl)
                self._rewards.append(r)
                if self.train_mode and self.agent:
                    d_po,_,q_po,cu_po=self._nearest(b)
                    tf_po=(self.t+1)/max(self.total_steps,1)
                    self.agent.update(b.soc,d_pre,tf_pre,b.pax_frac,q_pre,cu_pre,
                                      act,r,b.soc,d_po,tf_po,b.pax_frac,q_po,cu_po)
            elif b.state=='waiting':
                b.wait_s+=cfg.DT
                ch=self.chargers[b.cid]
                if ch.request(b.bid): b.state='plugging'; b.timer=cfg.PLUGIN_S
            elif b.state=='plugging':
                b.timer-=cfg.DT
                if b.timer<=0: b.state='charging'; b.timer=0
            elif b.state=='charging':
                ch=self.chargers[b.cid]; ch.tick()
                kwh=ch.kw*cfg.DT/3600.*cfg.CHG_EFF
                b.soc=min(1.,b.soc+kwh/cfg.BAT_KWH)
                b.energy_cost+=kwh*(cfg.PRICE_PEAK if peak else cfg.PRICE_OFF)
                ch.kwh+=kwh; step_kw+=ch.kw
                if b.soc>=cfg.TARG_SOC:
                    b.state='unplugging'; b.timer=cfg.UNPLUG_S; ch.release()
            elif b.state=='unplugging':
                b.timer-=cfg.DT
                if b.timer<=0: b.state='running'; b.cid=None
        self._grid.append(step_kw)
        for ch in self.chargers: ch.tick()
        self.t+=1

    def run(self):
        for _ in range(self.total_steps): self.step()

    def metrics(self):
        socs=[b.soc*100 for b in self.buses]
        waits=[b.wait_s for b in self.buses]
        gl=np.array(self._grid)
        return {
            'avg_soc':           float(np.mean(socs)),
            'charger_util_rate': float(np.mean([ch.util for ch in self.chargers])),
            'grid_load_variance':float(np.var(gl)) if len(gl)>1 else 0.,
            'total_cost':        float(sum(b.energy_cost for b in self.buses)),
            'wait_per_bus':      float(sum(waits)/max(len(self.buses),1)),
            'passengers_served': float(self._pax),
            'emergencies':       float(sum(b.emergencies for b in self.buses)),
            'peak_load_kw':      float(max(gl)) if len(gl) else 0.,
            'cost_per_pax':      float(sum(b.energy_cost for b in self.buses)/max(self._pax,1)),
        }

def cohens_d(a,b):
    a,b=np.array(a),np.array(b)
    ps=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return (np.mean(a)-np.mean(b))/ps if ps>1e-6 else None

# ── TRAINING ─────────────────────────────────────────────────────────────────
def train(bl_pax):
    agent=QAgent()
    print(f'\n[TRAINING] {cfg.EPISODES} episodes  ε 1.0→0.01')
    ep_r=[]; ep_s=[]; ep_e=[]
    t0=time.time()
    for ep in range(cfg.EPISODES):
        env=SimEnv('rl_tabular',agent=agent,seed=cfg.SEED+ep,
                   bl_pax=bl_pax,train_mode=True)
        env.run(); agent.decay()
        m=env.metrics()
        ep_s.append(m['avg_soc'])
        ep_r.append(float(np.mean(env._rewards)) if env._rewards else 0.)
        ep_e.append(agent.eps)
        if (ep+1)%150==0:
            print(f'  ep {ep+1:5d}  ε={agent.eps:.4f}  '
                  f'soc={np.mean(ep_s[-100:]):.1f}%  '
                  f'rew={np.mean(ep_r[-100:]):.4f}  '
                  f'cov={agent.coverage()*100:.1f}%  [{time.time()-t0:.0f}s]')
    print(f'  DONE ε={agent.eps:.5f}  coverage={agent.coverage()*100:.1f}%')
    return agent,ep_s,ep_r,ep_e

def run_mc(agent,bl_pax,n=None,strategies=None,n_buses=None,chpos=None,chg_kw=None):
    n=n or cfg.N_MC; strats=strategies or STRATEGIES
    results={s:defaultdict(list) for s in strats}
    for si,strat in enumerate(strats):
        ag=agent.clone() if 'rl' in strat else None
        for run in range(n):
            env=SimEnv(strat,agent=ag,seed=cfg.SEED+run*100+si,
                       bl_pax=bl_pax,n_buses=n_buses,chpos=chpos,chg_kw=chg_kw)
            env.run()
            for k,v in env.metrics().items(): results[strat][k].append(v)
    return results

def run_ablation(trained_agent,bl_pax,n=25):
    protos={'Never Charge':FixedAgent(0),'Always Charge':FixedAgent(1),
            'Random':RandomAgent(),'Trained RL':trained_agent}
    socs={k:[] for k in protos}; rews={k:[] for k in protos}
    print(f'\n[ABLATION] n={n}')
    for name,proto in protos.items():
        for run in range(n):
            ag=proto.clone(); ag.eps=0.
            env=SimEnv('rl_tabular',agent=ag,seed=cfg.SEED+run*77,bl_pax=bl_pax)
            env.run(); m=env.metrics()
            socs[name].append(m['avg_soc'])
            rews[name].append(float(np.mean(env._rewards)) if env._rewards else 0.)
        print(f'  {name:15s}: soc={np.mean(socs[name]):.1f}±{np.std(socs[name],ddof=1):.1f}%')
    return socs,rews

def run_sensitivity(agent,bl_pax,n=12):
    print(f'\n[SENSITIVITY] n={n}')
    res={'fleet':[],'chargers':[],'power':[]}
    for nb in [5,10,20]:
        r=run_mc(agent,bl_pax,n=n,n_buses=nb)
        for s in ['baseline','rl_coordinated']:
            d=r[s]
            res['fleet'].append({'fleet_size':nb,'strategy':LABELS[s],
                'avg_soc':np.mean(d['avg_soc']),'soc_std':np.std(d['avg_soc'],ddof=1),
                'charger_util':np.mean(d['charger_util_rate']),
                'passengers':np.mean(d['passengers_served'])})
        print(f'  fleet={nb}: done')
    for nc in [2,4,8]:
        sp=[(float(t),float(t*.3)) for t in np.linspace(0,ROUTE_KM,nc,endpoint=False)]
        r=run_mc(agent,bl_pax,n=n,chpos=sp)
        for s in ['baseline','rl_coordinated']:
            d=r[s]
            res['chargers'].append({'n_chargers':nc,'strategy':LABELS[s],
                'avg_soc':np.mean(d['avg_soc']),'soc_std':np.std(d['avg_soc'],ddof=1),
                'charger_util':np.mean(d['charger_util_rate']),
                'passengers':np.mean(d['passengers_served'])})
        print(f'  chargers={nc}: done')
    for kw in [100.,150.,200.]:
        r=run_mc(agent,bl_pax,n=n,chg_kw=kw)
        for s in ['baseline','rl_coordinated']:
            d=r[s]
            res['power'].append({'charge_kw':kw,'strategy':LABELS[s],
                'avg_soc':np.mean(d['avg_soc']),'soc_std':np.std(d['avg_soc'],ddof=1),
                'charger_util':np.mean(d['charger_util_rate']),
                'passengers':np.mean(d['passengers_served'])})
        print(f'  power={kw}kW: done')
    return {k:pd.DataFrame(v) for k,v in res.items()}

# ── FIGURES ───────────────────────────────────────────────────────────────────
def fig_route_map():
    fig,ax=plt.subplots(figsize=(9,5))
    wp=np.array(WAYPOINTS)
    ax.plot(wp[:,0],wp[:,1],'-',color='#3498db',lw=2.,label='Route 17 Woodlawn-Watson',zorder=2)
    sx=[]; sy=[]
    for sp in STOP_POS:
        x,y=route_pos(sp); sx.append(x); sy.append(y)
    ax.scatter(sx,sy,s=18,c='#e67e22',zorder=3,label=f'{len(STOP_POS)} Stops',
               edgecolors='white',lw=0.4)
    cx=[p[0] for p in CHPOS]; cy=[p[1] for p in CHPOS]
    ax.scatter(cx,cy,s=150,c='#27ae60',marker='^',zorder=4,
               label=f'{len(CHPOS)} En-route Chargers (150 kW OppCharge)',
               edgecolors='white',lw=0.8)
    for i,(x,y) in enumerate(zip(cx,cy)):
        ax.annotate(f'C{i}',xy=(x,y),xytext=(0.15,0.12),textcoords='offset points',
                    fontsize=7.5,color='#27ae60',fontweight='bold')
    ax.annotate('Woodlawn\n(Start)',(wp[0,0],wp[0,1]),
                xytext=(0.3,0.3),textcoords='offset points',fontsize=8)
    ax.annotate('Watson\n(End)',(wp[-1,0],wp[-1,1]),
                xytext=(0.3,0.3),textcoords='offset points',fontsize=8)
    ax.set_xlabel('East–West Offset (km)'); ax.set_ylabel('North–South Offset (km)')
    ax.set_title('Fig. 0 — Guelph Transit Route 17: Woodlawn → Watson  '
                 f'({ROUTE_KM:.1f} km, {len(STOP_POS)} stops)\n'
                 'Real GTFS geometry (Guelph Transit, Ontario, Canada). '
                 '▲ = OppCharge pantograph stations (150 kW).')
    ax.legend(fontsize=8); ax.set_aspect('equal')
    plt.tight_layout(); _sv(fig,'fig0_route_map')

def fig_training(ep_s,ep_r,ep_e):
    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(7,7.5),sharex=True)
    ep_x=np.arange(1,len(ep_r)+1); w=75
    def sm(x): return np.convolve(np.array(x),np.ones(w)/w,mode='valid')
    sx=ep_x[w-1:]
    ax1.plot(ep_x,ep_r,color='#bdc3c7',lw=.4,alpha=.5)
    ax1.plot(sx,sm(ep_r),color='#c0392b',lw=1.8,label=f'Smoothed (w={w})')
    ax1.axhline(0,color='black',lw=.6,ls=':'); ax1.set_ylabel('Mean Reward')
    ax1.legend(); ax1.set_title('(a) Reward Convergence')
    ax2.plot(ep_x,ep_s,color='#aed6f1',lw=.4,alpha=.5)
    ax2.plot(sx,sm(ep_s),color='#2471a3',lw=1.8,label=f'Smoothed (w={w})')
    ax2.set_ylabel('Avg SoC (%)'); ax2.legend(); ax2.set_title('(b) Fleet SoC During Training')
    ax3.semilogy(ep_x,np.array(ep_e)*100,color='#27ae60',lw=1.6)
    ax3.axhline(cfg.EPS_MIN*100,color='#c0392b',lw=1.,ls='--',
                label=f'ε_min={cfg.EPS_MIN*100:.0f}%')
    ax3.set_xlabel('Episode'); ax3.set_ylabel('ε (%, log)')
    ax3.set_title(f'(c) ε Decay  [final={ep_e[-1]*100:.2f}%]'); ax3.legend()
    fig.suptitle(f'Fig. 1 — RL Training ({cfg.EPISODES} Episodes | Guelph Route 17)',
                 fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig1_training_convergence')

def fig_soc(results):
    fig,ax=plt.subplots(figsize=(7,4))
    data=[np.array(results[s]['avg_soc']) for s in STRATEGIES]
    bp=ax.boxplot(data,patch_artist=True,widths=.55,notch=True,
                  medianprops=dict(color='white',lw=2.),
                  whiskerprops=dict(lw=.9),capprops=dict(lw=.9),
                  flierprops=dict(marker='o',ms=3,alpha=.5))
    for patch,s in zip(bp['boxes'],STRATEGIES): patch.set_facecolor(COLORS[s]); patch.set_alpha(.82)
    bl=np.array(results['baseline']['avg_soc']); y_top=max(np.max(d) for d in data)+1.5
    for xi,s in enumerate(STRATEGIES[1:],start=2):
        arr=np.array(results[s]['avg_soc'])
        _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
        sig='**' if pm<.01 else ('*' if pm<.05 else 'ns')
        ax.annotate('',xy=(xi,y_top),xytext=(1,y_top),
                    arrowprops=dict(arrowstyle='-',lw=.7,color='#aaa'))
        ax.text((1+xi)/2,y_top+.3,sig,ha='center',fontsize=8)
    ax.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
    ax.set_ylabel('Average SoC (%)')
    ax.set_title(f'Fig. 2 — Fleet SoC (N={cfg.N_MC} runs | {ROUTE_NAME})')
    patches=[mpatches.Patch(color=COLORS[s],alpha=.82,label=LABELS[s]) for s in STRATEGIES]
    ax.legend(handles=patches,fontsize=7); plt.tight_layout(); _sv(fig,'fig2_soc_comparison')

def fig_passengers(results):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
    means=[np.mean(results[s]['passengers_served']) for s in STRATEGIES]
    stds=[np.std(results[s]['passengers_served'],ddof=1) for s in STRATEGIES]
    cl=[COLORS[s] for s in STRATEGIES]
    ax1.bar(range(len(STRATEGIES)),means,yerr=stds,capsize=4,color=cl,alpha=.85,edgecolor='black',lw=.6)
    ax1.axhline(means[0]*.9,color='#c0392b',ls='--',lw=1.2,label='90% BL target')
    for i,(mn,sd) in enumerate(zip(means,stds)): ax1.text(i,mn+sd+2,f'{mn:.0f}',ha='center',fontsize=7)
    ax1.set_xticks(range(len(STRATEGIES)))
    ax1.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=12)
    ax1.set_ylabel('Passengers Served'); ax1.set_title('(a) Passengers Served'); ax1.legend()
    means2=[np.mean(results[s]['cost_per_pax']) for s in STRATEGIES]
    stds2=[np.std(results[s]['cost_per_pax'],ddof=1) for s in STRATEGIES]
    ax2.bar(range(len(STRATEGIES)),means2,yerr=stds2,capsize=4,color=cl,alpha=.85,edgecolor='black',lw=.6)
    ax2.set_xticks(range(len(STRATEGIES)))
    ax2.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=12)
    ax2.set_ylabel('Cost/Pax (CAD $)'); ax2.set_title('(b) Energy Cost per Passenger')
    fig.suptitle('Fig. 3 — Passenger Service & Energy Efficiency',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig3_passenger_service')

def fig_coordination(results):
    mets=['avg_soc','charger_util_rate','wait_per_bus','grid_load_variance','emergencies','passengers_served']
    ttls=['Avg SoC (%)','Charger Util.','Wait/Bus (s)','Grid Var. (kW²)','Emergencies','Passengers']
    fig,axes=plt.subplots(2,3,figsize=(10,6)); axes=axes.flatten()
    for idx,(m,ttl) in enumerate(zip(mets,ttls)):
        ax=axes[idx]
        rt=np.array(results['rl_tabular'][m]); rc=np.array(results['rl_coordinated'][m])
        _,pm=mannwhitneyu(rc,rt,alternative='two-sided')
        sig='***' if pm<.001 else ('**' if pm<.01 else ('*' if pm<.05 else 'ns'))
        d=cohens_d(rc,rt); d_str=f'd={d:.2f}' if d else 'd≈0'
        ax.bar([0,1],[np.mean(rt),np.mean(rc)],
               yerr=[np.std(rt,ddof=1),np.std(rc,ddof=1)],
               color=[COLORS['rl_tabular'],COLORS['rl_coordinated']],
               capsize=4,alpha=.85,edgecolor='black',lw=.6,
               tick_label=['RL-Tab.','RL-Coord.'])
        ax.set_title(f'{ttl}\n{sig}  {d_str}',fontsize=8.5)
    p=[mpatches.Patch(color=COLORS[s],alpha=.85,label=LABELS[s]) for s in ['rl_tabular','rl_coordinated']]
    fig.legend(handles=p,loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(.5,0))
    fig.suptitle('Fig. 4 — Coordination Benefit: RL-Coordinated vs. RL-Tabular',fontsize=10,fontweight='bold')
    plt.tight_layout(rect=[0,.05,1,1]); _sv(fig,'fig4_coordination_benefit')

def fig_stats(results):
    key=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
    lbl={'avg_soc':'Avg SoC (%)','charger_util_rate':'Charger Util.',
         'passengers_served':'Passengers','wait_per_bus':'Wait/Bus (s)',
         'total_cost':'Cost (CAD $)','emergencies':'Emergencies'}
    fig,axes=plt.subplots(2,3,figsize=(11,6.5)); axes=axes.flatten()
    for idx,m in enumerate(key):
        ax=axes[idx]
        bl=np.array(results['baseline'][m]); bm=np.mean(bl)
        bse=np.std(bl,ddof=1)/np.sqrt(len(bl))
        ax.axhspan(bm-1.96*bse,bm+1.96*bse,alpha=.12,color=COLORS['baseline'])
        ax.axhline(bm,color=COLORS['baseline'],lw=1.2,ls='--',zorder=3)
        for xi,s in enumerate(['heuristic','rl_tabular','rl_coordinated']):
            arr=np.array(results[s][m]); mn=np.mean(arr)
            se=np.std(arr,ddof=1)/np.sqrt(len(arr))
            _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
            sig='**' if pm<.01 else ('*' if pm<.05 else 'ns')
            ax.errorbar(xi,mn,yerr=1.96*se,fmt='o',color=COLORS[s],ms=7,lw=2,capsize=5,zorder=4)
            offset=max(1.96*se,abs(bm)*0.04,.5)
            ax.text(xi,mn+1.96*se+offset,sig,ha='center',fontsize=9,fontweight='bold')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['Heuristic','RL-Tab.','RL-Coord.'],rotation=10,fontsize=7.5)
        ax.set_title(lbl[m],fontsize=9)
    fig.suptitle('Fig. 5 — Statistical Comparison vs. Baseline  (mean ± 95% CI)',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig5_statistical_significance')

def fig_charger_grid(results):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
    data=[np.array(results[s]['charger_util_rate']) for s in STRATEGIES]
    vp=ax1.violinplot(data,showmedians=True,showextrema=True)
    for body,s in zip(vp['bodies'],STRATEGIES): body.set_facecolor(COLORS[s]); body.set_alpha(.75)
    vp['cmedians'].set_color('white'); vp['cmedians'].set_lw(2)
    ax1.set_xticks(range(1,len(STRATEGIES)+1))
    ax1.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
    ax1.set_ylabel('Charger Utilisation Rate'); ax1.set_title('(a) Charger Utilisation')
    means=[np.mean(results[s]['grid_load_variance']) for s in STRATEGIES]
    stds=[np.std(results[s]['grid_load_variance'],ddof=1) for s in STRATEGIES]
    ax2.bar(range(len(STRATEGIES)),means,yerr=stds,capsize=4,
            color=[COLORS[s] for s in STRATEGIES],alpha=.85,edgecolor='black',lw=.6)
    ax2.set_xticks(range(len(STRATEGIES)))
    ax2.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
    ax2.set_ylabel('Grid Load Variance (kW²)'); ax2.set_title('(b) Grid Stability')
    fig.suptitle('Fig. 6 — Charger Utilisation & Grid Stability',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig6_charger_grid')

def fig_policy(agent):
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    soc_v=np.linspace(0,1,300); ax=axes[0]
    for dist,ls,lbl in [(3.,'-','Near (3 km)'),(10.,'--','Mid (10 km)'),(20.,':','Far (20 km)')]:
        acts=[agent.greedy(s,dist,tf=.4,pf=.5) for s in soc_v]
        ax.plot(soc_v*100,acts,ls=ls,lw=1.8,label=lbl)
    ax.axvline(cfg.SOC_FLOOR*100,color='#8e44ad',lw=1.2,ls=':',label=f'Floor ({cfg.SOC_FLOOR*100:.0f}%)')
    ax.axvline(cfg.CRIT_SOC*100,color='#c0392b',lw=1.,ls='--',label=f'Critical ({cfg.CRIT_SOC*100:.0f}%)')
    ax.axvline(cfg.DOCK_SOC*100,color='#e67e22',lw=1.,ls='--',label=f'Dock ({cfg.DOCK_SOC*100:.0f}%)')
    ax.set_xlabel('SoC (%)'); ax.set_ylabel('Action')
    ax.set_yticks([0,1]); ax.set_yticklabels(['Drive','Charge'])
    ax.legend(fontsize=7); ax.set_title('(a) Policy vs. SoC')
    ax=axes[1]; S=cfg.SOC_B; D=cfg.DIST_B
    q_slice=np.zeros((S,D))
    for si in range(S):
        for di in range(D):
            q_slice[si,di]=agent.Q[si,di,2,1,0,0,1]-agent.Q[si,di,2,1,0,0,0]
    im=ax.imshow(q_slice,cmap='RdYlGn',aspect='auto',origin='lower',extent=[0,20,0,100])
    plt.colorbar(im,ax=ax,label='Q(charge)−Q(drive)',shrink=.85)
    ax.set_xlabel('Distance to Charger (km)'); ax.set_ylabel('SoC (%)')
    ax.set_title('(b) Q-Value Advantage Map')
    fig.suptitle('Fig. 7 — Learned Charging Policy Analysis',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig7_policy_analysis')

def fig_ablation(socs,rews):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
    names=list(socs.keys())
    ab_c={'Never Charge':'#95a5a6','Always Charge':'#e74c3c','Random':'#f39c12','Trained RL':'#27ae60'}
    for ax,data,ylabel,title in [(ax1,socs,'Avg SoC (%)','(a) SoC by Condition'),
                                  (ax2,rews,'Mean Reward','(b) Reward by Condition')]:
        data_l=[np.array(data[n]) for n in names]
        bp=ax.boxplot(data_l,patch_artist=True,widths=.55,
                      medianprops=dict(color='white',lw=2.),
                      whiskerprops=dict(lw=.9),capprops=dict(lw=.9),
                      flierprops=dict(marker='o',ms=3,alpha=.5))
        for patch,n in zip(bp['boxes'],names): patch.set_facecolor(ab_c[n]); patch.set_alpha(.8)
        ax.set_xticklabels(names,rotation=12,fontsize=7.5)
        ax.set_ylabel(ylabel); ax.set_title(title)
    fig.suptitle('Fig. 8 — Ablation Study',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig8_ablation')

def fig_sensitivity(sens):
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    c_map={'Baseline':COLORS['baseline'],'RL-Coordinated':COLORS['rl_coordinated']}
    for ax,(key,xlabel,col) in zip(axes,[
            ('fleet','Fleet Size (buses)','fleet_size'),
            ('chargers','Number of Chargers','n_chargers'),
            ('power','Charging Power (kW)','charge_kw')]):
        df=sens[key]
        for strat in ['Baseline','RL-Coordinated']:
            sub=df[df.strategy==strat]
            ax.errorbar(sub[col],sub.avg_soc,yerr=sub.soc_std,
                        fmt='-o',color=c_map[strat],ms=5,lw=1.5,capsize=4,label=strat)
        ax.set_xlabel(xlabel); ax.set_ylabel('Avg SoC (%)')
        ax.legend(fontsize=7)
    fig.suptitle('Fig. 9 — Sensitivity Analysis  (±std, 12 runs each)',fontsize=11,fontweight='bold')
    plt.tight_layout(); _sv(fig,'fig9_sensitivity')

def fig_perf_matrix(results):
    mets=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
    m8=['Avg SoC','Util.','Pax','Wait','Cost','Emg']
    mat=np.zeros((len(STRATEGIES),len(mets))); annot=[]
    for si,s in enumerate(STRATEGIES):
        ra=[]
        for mi,m in enumerate(mets):
            arr=np.array(results[s][m]); bl=np.array(results['baseline'][m])
            mn=np.mean(arr)
            if s=='baseline': mat[si,mi]=0; ra.append(f'{mn:.1f}')
            else:
                diff=(mn-np.mean(bl))/(np.mean(bl)+1e-9)*100; mat[si,mi]=diff
                se=np.std(arr,ddof=1)/np.sqrt(len(arr))
                _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
                star='*' if pm<.05 else ''
                ra.append(f'{mn:.1f}\n({diff:+.0f}%){star}')
        annot.append(ra)
    fig,ax=plt.subplots(figsize=(10,4))
    im=ax.imshow(mat,cmap=plt.cm.RdYlGn,aspect='auto',vmin=-30,vmax=30)
    ax.set_xticks(range(len(mets))); ax.set_xticklabels(m8,fontsize=9)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels([LABELS[s] for s in STRATEGIES],fontsize=9)
    for si in range(len(STRATEGIES)):
        for mi in range(len(mets)): ax.text(mi,si,annot[si][mi],ha='center',va='center',fontsize=8)
    plt.colorbar(im,ax=ax,label='% Change vs. Baseline',shrink=.8)
    ax.set_title('Fig. 10 — Performance Matrix (* p<0.05 | Guelph Route 17)',fontsize=9)
    plt.tight_layout(); _sv(fig,'fig10_performance_matrix')

def save_tables(results,sens):
    mets=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
    csv_rows=[]
    for s in STRATEGIES:
        crow={'Strategy':LABELS[s]}
        for m in mets:
            arr=np.array(results[s][m]); mn=np.mean(arr); sd=np.std(arr,ddof=1)
            se=sd/np.sqrt(len(arr))
            crow[m+'_mean']=round(mn,2); crow[m+'_std']=round(sd,2)
            crow[m+'_95ci']=f'[{mn-1.96*se:.2f},{mn+1.96*se:.2f}]'
        csv_rows.append(crow)
    pd.DataFrame(csv_rows).to_csv(f'{OUT}/table1_main.csv',index=False)

    sig_rows=[]
    for s in ['heuristic','rl_tabular','rl_coordinated']:
        for m in mets:
            arr=np.array(results[s][m]); bl=np.array(results['baseline'][m])
            mn=np.mean(arr); sd=np.std(arr,ddof=1); se=sd/np.sqrt(len(arr))
            _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
            _,pt=ttest_ind(arr,bl,equal_var=False)
            d=cohens_d(arr,bl)
            sig_rows.append({'Strategy':LABELS[s],'Metric':m,
                'Mean':round(mn,2),'Std':round(sd,2),
                '95%CI':f'[{mn-1.96*se:.2f},{mn+1.96*se:.2f}]',
                'Δ_BL':round(mn-np.mean(bl),2),
                'p_MWU':round(pm,4),'p_t':round(pt,4),
                'Cohen_d':round(d,3) if d else 'n/a',
                'Sig':'Yes' if pm<.05 else 'No'})
    pd.DataFrame(sig_rows).to_csv(f'{OUT}/table2_statistics.csv',index=False)
    for k,df in sens.items(): df.to_csv(f'{OUT}/table3_sensitivity_{k}.csv',index=False)
    print('  Tables saved.')

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    t0=time.time()

    print('\n[1] Baseline reference run...')
    bl_env=SimEnv('baseline',seed=cfg.SEED); bl_env.run(); bl_pax=bl_env._pax
    print(f'  bl_pax={bl_pax}')

    print(f'\n[2] Training ({cfg.EPISODES} episodes)...')
    agent,ep_s,ep_r,ep_e=train(bl_pax)

    print('\n[3] Monte Carlo (30 runs × 4 strategies)...')
    results=run_mc(agent,bl_pax)
    for s in STRATEGIES:
        mv={k:np.mean(v) for k,v in results[s].items()}
        print(f'  {s:22s}  SoC={mv["avg_soc"]:.1f}%  '
              f'pax={mv["passengers_served"]:.0f}  '
              f'util={mv["charger_util_rate"]:.3f}  '
              f'emg={mv["emergencies"]:.0f}')

    print('\n[4] Ablation...'); ab_socs,ab_rews=run_ablation(agent,bl_pax,n=25)
    print('\n[5] Sensitivity...'); sens=run_sensitivity(agent,bl_pax,n=12)

    print('\n[6] Figures...')
    fig_route_map(); fig_training(ep_s,ep_r,ep_e); fig_soc(results)
    fig_passengers(results); fig_coordination(results); fig_stats(results)
    fig_charger_grid(results); fig_policy(agent); fig_ablation(ab_socs,ab_rews)
    fig_sensitivity(sens); fig_perf_matrix(results)

    print('\n[7] Tables...'); save_tables(results,sens)

    print(f'\n{"="*65}')
    print('  BELIEVABILITY CHECK')
    print(f'{"="*65}')
    for s in STRATEGIES:
        util=np.mean(results[s]['charger_util_rate'])
        emg=np.mean(results[s]['emergencies'])
        pax=np.mean(results[s]['passengers_served'])
        bl_pax_mc=np.mean(results['baseline']['passengers_served'])
        util_ok='✓' if 0.30<util<0.85 else '⚠ CHECK'
        emg_ok='✓' if emg==0 else '✗ FAIL'
        print(f'  {s:22s}  util={util:.3f} {util_ok}  emg={emg:.0f} {emg_ok}  '
              f'pax={pax:.0f} ({pax/bl_pax_mc*100:.0f}%)')

    bl_arr=np.array(results['baseline']['avg_soc'])
    print()
    for s in ['heuristic','rl_tabular','rl_coordinated']:
        arr=np.array(results[s]['avg_soc'])
        d=cohens_d(arr,bl_arr); _,pm=mannwhitneyu(arr,bl_arr,alternative='two-sided')
        mn=np.mean(arr); se=np.std(arr,ddof=1)/np.sqrt(len(arr))
        print(f'  {LABELS[s]:22s}: SoC={mn:.1f}%  d={d:.3f if d else "n/a"}  '
              f'p={pm:.4f}  CI=[{mn-1.96*se:.1f},{mn+1.96*se:.1f}]')

    print(f'\n  Runtime: {(time.time()-t0)/60:.1f} min')
    print(f'  All outputs: {OUT}/')
    print('='*65)

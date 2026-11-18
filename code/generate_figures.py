import pickle, numpy as np, warnings, os, sys
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from collections import defaultdict

# Re-run the training inline (avoiding pickle class issue)
code = open('/home/claude/ieee_ev_simulation.py').read().split('if __name__')[0]
exec(code)

OUT = '/mnt/user-data/outputs/ieee_ev_results'
os.makedirs(OUT, exist_ok=True)

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
    print(f'  Saved {name}')

def cohens_d(a,b):
    a,b=np.array(a),np.array(b)
    ps=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return (np.mean(a)-np.mean(b))/ps if ps>1e-6 else None

import time

print('=== TRAINING ===')
EPISODES_FAST = 300
EPS_DEC_FAST = (0.01/1.0)**(1/300)

agent = QAgent()
bl_env = SimEnv('baseline', seed=42)
bl_env.run()
bl_pax = bl_env._pax_total
print(f'Baseline pax reference: {bl_pax}')

ep_rews, ep_socs, ep_eps = [], [], []
for ep in range(EPISODES_FAST):
    env = SimEnv('rl_tabular', agent=agent, seed=42+ep, baseline_pax=bl_pax, train_mode=True)
    env.run()
    agent.eps = max(0.01, agent.eps * EPS_DEC_FAST)
    m = env.metrics()
    ep_socs.append(m['avg_soc'])
    ep_rews.append(float(np.mean(env._rewards)) if env._rewards else 0.0)
    ep_eps.append(agent.eps)
    if (ep+1) % 100 == 0:
        print(f'  ep {ep+1}/{EPISODES_FAST}  eps={agent.eps:.4f}  soc={np.mean(ep_socs[-50:]):.1f}%')

print(f'Final eps={agent.eps:.5f}  CONVERGED: {agent.eps <= 0.05}')
agent.eps = 0.0

print('=== MONTE CARLO ===')
N_RUNS = 15
results = {s: defaultdict(list) for s in STRATEGIES}
for si, strat in enumerate(STRATEGIES):
    ag = agent.clone() if 'rl' in strat else None
    for run in range(N_RUNS):
        env = SimEnv(strat, agent=ag, seed=42+run*100+si, baseline_pax=bl_pax)
        env.run()
        m = env.metrics()
        for k,v in m.items():
            results[strat][k].append(v)
    mv = {k: np.mean(v) for k,v in results[strat].items()}
    print(f'  {strat:22s}  SOC={mv["avg_soc"]:.1f}%  pax={mv["passengers_served"]:.0f}  util={mv["charger_util_rate"]:.2f}')

print('=== GENERATING FIGURES ===')

# FIG 1: Training Convergence
fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(7,7),sharex=True)
ep_x=np.arange(1,len(ep_rews)+1)
w=30

def smooth(x,w):
    x=np.array(x)
    return np.convolve(x,np.ones(w)/w,mode='valid') if len(x)>=w else x

sm_ep=ep_x[w-1:]
ax1.plot(ep_x,ep_rews,color='#bdc3c7',lw=0.5,alpha=0.5)
ax1.plot(sm_ep,smooth(ep_rews,w),color='#e74c3c',lw=1.8,label=f'Smoothed (w={w})')
ax1.set_ylabel('Mean Reward'); ax1.legend(loc='upper left')
ax1.set_title('(a) Reward Convergence')
ax2.plot(ep_x,ep_socs,color='#aed6f1',lw=0.5,alpha=0.5)
ax2.plot(sm_ep,smooth(ep_socs,w),color='#2980b9',lw=1.8,label=f'Smoothed (w={w})')
ax2.set_ylabel('Avg SOC (%)'); ax2.legend(loc='lower right')
ax2.set_title('(b) Fleet SOC During Training')
ax3.plot(ep_x,np.array(ep_eps)*100,color='#27ae60',lw=1.6)
ax3.axhline(1.0,color='#e74c3c',lw=1.0,ls='--',label='ε_min=0.01 (1%)')
ax3.set_xlabel('Episode'); ax3.set_ylabel('Exploration ε (%)')
ax3.set_title(f'(c) ε-Decay  [final ε={ep_eps[-1]*100:.2f}%]')
ax3.legend()
fig.suptitle('Fig. 1 — RL Training Convergence (300 Episodes, ε: 100%→1%)',fontsize=11,fontweight='bold')
plt.tight_layout(); _sv(fig,'fig1_training_convergence')

# FIG 2: SOC Boxplot
fig,ax=plt.subplots(figsize=(7,4))
data=[np.array(results[s]['avg_soc']) for s in STRATEGIES]
bp=ax.boxplot(data,patch_artist=True,widths=0.55,
              medianprops=dict(color='white',lw=2),
              whiskerprops=dict(lw=1),capprops=dict(lw=1),
              flierprops=dict(marker='o',ms=3,alpha=0.5))
for patch,c in zip(bp['boxes'],[COLORS[s] for s in STRATEGIES]):
    patch.set_facecolor(c); patch.set_alpha(0.80)
ax.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
ax.set_ylabel('Average State-of-Charge (%)')
ax.set_title('Fig. 2 — Fleet SOC Distribution Across Strategies (N=15 runs/strategy)')
patches=[plt.Rectangle((0,0),1,1,fc=COLORS[s],alpha=0.8) for s in STRATEGIES]
ax.legend(patches,[LABELS[s] for s in STRATEGIES],loc='best',fontsize=7)
plt.tight_layout(); _sv(fig,'fig2_soc_comparison')

# FIG 3: Passenger Service
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
means=[np.mean(results[s]['passengers_served']) for s in STRATEGIES]
stds=[np.std(results[s]['passengers_served'],ddof=1) for s in STRATEGIES]
cl=[COLORS[s] for s in STRATEGIES]
ax1.bar(range(len(STRATEGIES)),means,yerr=stds,capsize=4,color=cl,alpha=0.85,edgecolor='black',lw=0.6)
ax1.set_xticks(range(len(STRATEGIES))); ax1.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=12)
ax1.set_ylabel('Passengers Served'); ax1.set_title('(a) Passengers Served (mean±std)')
ax1.axhline(means[0]*0.9,color='#e74c3c',ls='--',lw=1.2,label='90% of baseline (target)')
ax1.legend()
means2=[np.mean(results[s]['cost_per_pax']) for s in STRATEGIES]
stds2=[np.std(results[s]['cost_per_pax'],ddof=1) for s in STRATEGIES]
ax2.bar(range(len(STRATEGIES)),means2,yerr=stds2,capsize=4,color=cl,alpha=0.85,edgecolor='black',lw=0.6)
ax2.set_xticks(range(len(STRATEGIES))); ax2.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=12)
ax2.set_ylabel('Cost per Passenger ($)'); ax2.set_title('(b) Energy Cost per Passenger (mean±std)')
fig.suptitle('Fig. 3 — Passenger Service Preservation & Energy Efficiency',fontsize=11,fontweight='bold')
plt.tight_layout(); _sv(fig,'fig3_passenger_service')

# FIG 4: Coordination Benefit
metrics4=['avg_soc','charger_util_rate','wait_per_bus','grid_load_variance','emergencies','passengers_served']
titles4=['Avg SOC (%)','Charger Util.','Wait/Bus (s)','Grid Var.','Emergencies','Passengers']
fig,axes=plt.subplots(2,3,figsize=(10,6)); axes=axes.flatten()
for idx,(m,ttl) in enumerate(zip(metrics4,titles4)):
    ax=axes[idx]
    rt=np.array(results['rl_tabular'][m]); rc=np.array(results['rl_coordinated'][m])
    _,pm=mannwhitneyu(rc,rt,alternative='two-sided')
    sig='***' if pm<0.001 else ('**' if pm<0.01 else ('*' if pm<0.05 else f'ns'))
    d=cohens_d(rc,rt)
    d_str=f'd={d:.2f}' if d is not None else 'd=n/a'
    ax.bar([0,1],[np.mean(rt),np.mean(rc)],
           yerr=[np.std(rt,ddof=1),np.std(rc,ddof=1)],
           color=[COLORS['rl_tabular'],COLORS['rl_coordinated']],
           capsize=4,alpha=0.85,edgecolor='black',lw=0.6,
           tick_label=['RL-Tabular','RL-Coord.'])
    ax.set_title(f'{ttl}\n{sig} | {d_str}',fontsize=8)
    ax.tick_params(axis='x',labelsize=7)
p=[plt.Rectangle((0,0),1,1,fc=COLORS[s],alpha=0.8) for s in ['rl_tabular','rl_coordinated']]
fig.legend(p,['RL-Tabular','RL-Coordinated'],loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(0.5,0))
fig.suptitle('Fig. 4 — Coordination Benefit: RL-Coordinated vs. RL-Tabular\n(Same RL policy; difference = value of coordination layer)',fontsize=10,fontweight='bold')
plt.tight_layout(rect=[0,0.05,1,1]); _sv(fig,'fig4_coordination_benefit')

# FIG 5: Statistical Significance
key_m=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
m_lbl={'avg_soc':'Avg. SOC (%)','charger_util_rate':'Charger Util.','passengers_served':'Passengers',
       'wait_per_bus':'Wait/Bus (s)','total_cost':'Cost ($)','emergencies':'Emergencies'}
fig,axes=plt.subplots(2,3,figsize=(12,7)); axes=axes.flatten()
for idx,m in enumerate(key_m):
    ax=axes[idx]
    bl_arr=np.array(results['baseline'][m]); bl_mn=np.mean(bl_arr)
    bl_se=np.std(bl_arr,ddof=1)/np.sqrt(len(bl_arr))
    ax.axhspan(bl_mn-1.96*bl_se,bl_mn+1.96*bl_se,alpha=0.12,color=COLORS['baseline'])
    ax.axhline(bl_mn,color=COLORS['baseline'],lw=1.2,ls='--',label='Baseline')
    for xi,strat in enumerate(['heuristic','rl_tabular','rl_coordinated']):
        arr=np.array(results[strat][m]); mn=np.mean(arr); sd=np.std(arr,ddof=1); se=sd/np.sqrt(len(arr))
        _,pm=mannwhitneyu(arr,bl_arr,alternative='two-sided')
        sig='**' if pm<0.01 else ('*' if pm<0.05 else 'ns')
        ax.errorbar(xi,mn,yerr=1.96*se,fmt='o',color=COLORS[strat],ms=7,lw=2,capsize=5,elinewidth=1.5,label=LABELS[strat])
        offset=max(1.96*se,abs(bl_mn)*0.03)
        ax.text(xi,mn+1.96*se+offset,sig,ha='center',va='bottom',fontsize=9,fontweight='bold')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Heuristic','RL-Tab.','RL-Coord.'],rotation=10,fontsize=7)
    ax.set_title(m_lbl[m],fontsize=9)
    if idx==0:
        ax.legend(fontsize=6,loc='best')
fig.suptitle('Fig. 5 — Statistical Comparison vs. Baseline  (mean ± 95% CI)\n(*p<0.05, **p<0.01; Mann-Whitney U; shaded = baseline 95% CI)',fontsize=11,fontweight='bold')
plt.tight_layout(); _sv(fig,'fig5_statistical_significance')

# FIG 6: Charger Util & Grid
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
data=[np.array(results[s]['charger_util_rate']) for s in STRATEGIES]
vp=ax1.violinplot(data,showmedians=True,showextrema=True)
for body,c in zip(vp['bodies'],[COLORS[s] for s in STRATEGIES]):
    body.set_facecolor(c); body.set_alpha(0.75)
ax1.set_xticks(range(1,len(STRATEGIES)+1)); ax1.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
ax1.set_ylabel('Charger Utilization Rate'); ax1.set_title('(a) Charger Utilization Distribution')
means_gv=[np.mean(results[s]['grid_load_variance']) for s in STRATEGIES]
stds_gv=[np.std(results[s]['grid_load_variance'],ddof=1) for s in STRATEGIES]
ax2.bar(range(len(STRATEGIES)),means_gv,yerr=stds_gv,capsize=4,
        color=[COLORS[s] for s in STRATEGIES],alpha=0.85,edgecolor='black',lw=0.6)
ax2.set_xticks(range(len(STRATEGIES))); ax2.set_xticklabels([LABELS[s] for s in STRATEGIES],rotation=10)
ax2.set_ylabel('Grid Load Variance (kW²)'); ax2.set_title('(b) Grid Load Variance (lower = better)')
fig.suptitle('Fig. 6 — Charger Utilization & Grid Stability',fontsize=11,fontweight='bold')
plt.tight_layout(); _sv(fig,'fig6_charger_grid')

# FIG 7: Policy Threshold
fig,ax=plt.subplots(figsize=(7,4))
soc_vals=np.linspace(0,1,200)
for dist,ls,lbl in [(2.0,'-','Near (2 km)'),(7.0,'--','Mid (7 km)'),(13.0,':','Far (13 km)')]:
    acts=[agent.greedy(s,dist,tf=0.4,pf=0.5) for s in soc_vals]
    ax.plot(soc_vals*100,acts,ls=ls,lw=1.8,label=lbl)
ax.axvline(20,color='#e74c3c',lw=1.0,ls='--',label='Critical SOC (20%)')
ax.axvline(35,color='#e67e22',lw=1.0,ls='--',label='Dock SOC (35%)')
ax.set_xlabel('State of Charge (%)'); ax.set_ylabel('Action')
ax.set_yticks([0,1]); ax.set_yticklabels(['Drive','Charge'])
ax.legend(fontsize=8)
ax.set_title('Fig. 7 — Learned Charging Policy vs. SOC\n(Off-peak, moderate load, t=0.4 of window)')
plt.tight_layout(); _sv(fig,'fig7_policy_threshold')

# FIG 8: Performance Matrix
metrics8=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
m8=['Avg SOC','Charger Util','Passengers','Wait/Bus','Cost','Emergencies']
mat=np.zeros((len(STRATEGIES),len(metrics8))); annot=[]
for si,strat in enumerate(STRATEGIES):
    ra=[]
    for mi,m in enumerate(metrics8):
        arr=np.array(results[strat][m]); bl=np.array(results['baseline'][m])
        mn=np.mean(arr)
        if strat=='baseline': mat[si,mi]=0; ra.append(f'{mn:.1f}')
        else:
            diff=(mn-np.mean(bl))/(np.mean(bl)+1e-9)*100; mat[si,mi]=diff
            se=np.std(arr,ddof=1)/np.sqrt(len(arr))
            ra.append(f'{mn:.1f}\n({diff:+.1f}%)')
    annot.append(ra)
fig,ax=plt.subplots(figsize=(11,4))
im=ax.imshow(mat,cmap=plt.cm.RdYlGn,aspect='auto',vmin=-30,vmax=30)
ax.set_xticks(range(len(metrics8))); ax.set_xticklabels(m8,fontsize=9)
ax.set_yticks(range(len(STRATEGIES))); ax.set_yticklabels([LABELS[s] for s in STRATEGIES],fontsize=9)
for si in range(len(STRATEGIES)):
    for mi in range(len(metrics8)):
        ax.text(mi,si,annot[si][mi],ha='center',va='center',fontsize=8)
plt.colorbar(im,ax=ax,label='% Change vs Baseline',shrink=0.8)
ax.set_title('Fig. 8 — Performance Matrix: Values & % Change vs. Baseline\n(Green = improvement, Red = degradation relative to Baseline)',fontsize=10)
plt.tight_layout(); _sv(fig,'fig8_performance_matrix')

# === TABLES ===
import pandas as pd
print('\n=== TABLES ===')
metrics_t=['avg_soc','charger_util_rate','passengers_served','wait_per_bus','total_cost','emergencies']
rows=[]
for strat in STRATEGIES:
    row={'Strategy':LABELS[strat]}
    for m in metrics_t:
        arr=np.array(results[strat][m]); mn=np.mean(arr); sd=np.std(arr,ddof=1); se=sd/np.sqrt(len(arr))
        row[m+'_mean']=round(mn,2); row[m+'_std']=round(sd,2)
        row[m+'_95ci']=f'[{mn-1.96*se:.2f},{mn+1.96*se:.2f}]'
        if strat!='baseline':
            bl=np.array(results['baseline'][m]); _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
            d=cohens_d(arr,bl)
            row[m+'_p_mwu']=round(pm,4); row[m+'_d']=round(d,3) if d is not None else 'n/a'
    rows.append(row)
df=pd.DataFrame(rows)
df.to_csv(f'{OUT}/table1_performance.csv',index=False)
print(f'  Saved table1_performance.csv')

# LaTeX table
tex=['\\begin{table}[t]','\\centering',
     '\\caption{Performance Comparison (Mean $\\pm$ Std, 15 Monte-Carlo Runs)}',
     '\\label{tab:results}','\\begin{tabular}{lrrrrrr}','\\toprule',
     'Strategy & SOC(\\%) & Util. & Pax & Wait(s) & Cost(\\$) & Emg \\\\ \\midrule']
for row in rows:
    cells=[row['Strategy']]
    for m in metrics_t:
        cells.append(f"${row[m+'_mean']:.1f}\\pm{row[m+'_std']:.1f}$")
    tex.append(' & '.join(cells)+' \\\\')
tex+=['\\bottomrule','\\end{tabular}','\\end{table}']
with open(f'{OUT}/table1_performance.tex','w') as f: f.write('\n'.join(tex))
print(f'  Saved table1_performance.tex')

# Stat significance table
sig_rows=[]
for strat in ['heuristic','rl_tabular','rl_coordinated']:
    for m in metrics_t:
        arr=np.array(results[strat][m]); bl=np.array(results['baseline'][m])
        mn=np.mean(arr); sd=np.std(arr,ddof=1); se=sd/np.sqrt(len(arr))
        _,pm=mannwhitneyu(arr,bl,alternative='two-sided')
        d=cohens_d(arr,bl)
        sig_rows.append({'Strategy':LABELS[strat],'Metric':m,
            'Mean':round(mn,2),'Std':round(sd,2),
            '95%CI':f'[{mn-1.96*se:.2f},{mn+1.96*se:.2f}]',
            'Δ_vs_BL':round(mn-np.mean(bl),2),
            'p_MWU':round(pm,4),"Cohen's_d":round(d,3) if d is not None else 'n/a',
            'Significant':'Yes' if pm<0.05 else 'No'})
pd.DataFrame(sig_rows).to_csv(f'{OUT}/table2_statistical.csv',index=False)
print('  Saved table2_statistical.csv')

print('\n=== CLAIM VERIFICATION ===')
print(f'  Fix 1: Final eps={ep_eps[-1]:.5f}  {"PASS" if ep_eps[-1]<=0.05 else "FAIL"}')
bl_p=np.mean(results['baseline']['passengers_served'])
rc_p=np.mean(results['rl_coordinated']['passengers_served'])
print(f'  Fix 2: RL-Coord pax={rc_p:.0f} vs BL={bl_p:.0f} ({rc_p/bl_p*100:.1f}%)  {"PASS" if rc_p/bl_p>=0.90 else "NEEDS TUNING"}')
print(f'  Fix 3: rl_tabular in results: {"PASS" if "rl_tabular" in results else "FAIL"}')
bl_soc=np.array(results['baseline']['avg_soc'])
rc_soc=np.array(results['rl_coordinated']['avg_soc'])
d=cohens_d(rc_soc,bl_soc)
print(f"  Fix 4: Cohen's d (SOC, RC vs BL) = {d:.3f if d else 'n/a'}  (was 14.41 before fix)")
print('\nALL DONE')

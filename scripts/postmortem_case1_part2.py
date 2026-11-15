"""Разбор кейса 1, часть 2: как построен факт и более грубые варианты прогноза.

Запуск из корня:  python scripts/postmortem_case1_part2.py
Таблицы пишет в outputs/postmortem/. Основной пайплайн и result.csv не трогает.

  1. Варианты построения факта на 1250011604: что из них ломает r2.
  2. Грубые признаки (сглаживание, окна, счёт дефектов) на всех честных проверках.
  3. Гибрид «main + сглаженный минимум» против main.
Сдвиг 2018 -> 2022 по швам берётся из outputs/postmortem/joint_shift_2018_2022.csv (часть 1).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT + "/src")
from pipetwin import data, case1, pipeline as pl
from pipetwin.validate import pearson_r2, mae
warnings.filterwarnings("ignore"); pd.set_option("display.width", 300); pd.set_option("display.max_rows", 200)
OUT = ROOT + "/outputs/postmortem/"
cfg = yaml.safe_load(open(ROOT + "/config.yaml", encoding="utf-8")); c = cfg["case1"]; RAW = ROOT + "/" + cfg["paths"]["raw"]
P, V, J = data.load_passport(RAW), data.load_vtd(RAW), data.load_joints(RAW)
tid, A, B = int(c["target_id"]), pd.Timestamp(c["start_run"]), pd.Timestamp(c["target_run"])
pp = P.set_index("id"); pipe_ids = P.loc[P["pipeline"] == pp.loc[tid, "pipeline"], "id"].tolist()
acc = pl._offsets(V, A); L_t, wall_t, d_mm = pp.loc[tid, "length_m"], pp.loc[tid, "wall_mm"], pp.loc[tid, "d_mm"]
js = pl.joint_shifts(J, A, B, pipe_ids); vid = 1250011604; Lv, wall_v = pp.loc[vid, "length_m"], pp.loc[vid, "wall_mm"]
anc = {p: pl._anchor(V, js, p, pp.loc[p, "length_m"], d_mm, acc[p], A, B, c["match_tol_m"]) for p in pipe_ids if p != tid}
gm = pl._fit(list(anc.values()))

# ---- 1. варианты факта
val=pd.read_csv(ROOT+'/outputs/validation_case1_points.csv'); pb=val.pred_blend.values; pp_=val.pred_pers.values
b=V[(V.id==vid)&(V.date==B)].copy(); n=50

def segstat(d,fn,xcol='x',fill=np.nan,off=0.0):
    s=data.seg_of(d[xcol]+off,Lv); g=d.groupby(s)
    return fn(g).reindex(range(n)).fillna(fill).to_numpy()
facts={
 'минимум, пустые=стенка (наш)':segstat(b,lambda g:g.rw.min(),fill=8.0),
 'минимум, пустые выброшены':segstat(b,lambda g:g.rw.min()),
 'минимум, пустые=0':segstat(b,lambda g:g.rw.min(),fill=0.0),
 'минимум, пустые=средняя':None,
 'среднее по дефектам':segstat(b,lambda g:g.rw.mean(),fill=8.0),
 'медиана':segstat(b,lambda g:g.rw.median(),fill=8.0),
 '5-й перцентиль':segstat(b,lambda g:g.rw.quantile(.05),fill=8.0),
 'макс. глубина, %':segstat(b,lambda g:g.dp.max(),fill=0.0),
 'минимум без записей за концом участка':segstat(b[~b.beyond.astype(bool)],lambda g:g.rw.min(),fill=8.0),
 'минимум, только внутр.':segstat(b[b.internal],lambda g:g.rw.min(),fill=8.0),
 'минимум, округл. до 0,5 мм':np.round(segstat(b,lambda g:g.rw.min(),fill=8.0)*2)/2,
 'минимум, сетка +5 м':segstat(b,lambda g:g.rw.min(),fill=8.0,off=5.0),
 'минимум, сетка +10 м (нумерация с 1)':segstat(b,lambda g:g.rw.min(),fill=8.0,off=10.0),
 'минимум, сетка от Distance_raw':None,
 'число дефектов':segstat(b,lambda g:g.rw.size(),fill=0.0),
 'минимум по дефектам глубже 15%':segstat(b[b.dp>=15],lambda g:g.rw.min(),fill=8.0),
 'минимум по дефектам глубже 20%':segstat(b[b.dp>=20],lambda g:g.rw.min(),fill=8.0),
}
f=facts['минимум, пустые выброшены']; facts['минимум, пустые=средняя']=np.where(np.isnan(f),np.nanmean(f),f)
s=np.floor(b.x_raw/10).astype(int); s=s-s.min(); facts['минимум, сетка от Distance_raw']=b.groupby(s).rw.min().reindex(range(n)).fillna(8.0).to_numpy()
rows=[]
for k,y in facts.items():
    rows.append({'факт':k,'r2 итога':pearson_r2(y,pb),'r2 прошлого ВТД':pearson_r2(y,pp_),'n':int(np.isfinite(y).sum()),'sd факта':np.nanstd(y)})
F=pd.DataFrame(rows); print(F.round(3).to_string()); F.round(3).to_csv(OUT+'fact_variants_11604.csv',index=False)

# ---- 2. грубые признаки

def kern(v,w):
    k=np.array(w,float)/sum(w); return np.convolve(np.pad(v,len(k)//2,mode='edge'),k,mode='valid')
def z(v): return (v-v.mean())/(v.std() if v.std()>0 else 1)
def cands(xv,rw,dp,L,wall):
    n=data.n_segments(L); xv=np.asarray(xv,float); rw=np.asarray(rw,float); dp=np.asarray(dp,float)
    o=np.argsort(xv); xs,rs,ds=xv[o],rw[o],dp[o]
    def win(h,fn,fill):
        out=np.full(n,fill,float)
        for i in range(n):
            lo,hi=np.searchsorted(xs,i*10-h),np.searchsorted(xs,i*10+10+h)
            if hi>lo: out[i]=fn(rs[lo:hi],ds[lo:hi])
        return out
    C={}
    C['прошлый ВТД (минимум)']=win(0,lambda r,d:r.min(),wall)
    C['5-й перцентиль']=win(0,lambda r,d:np.quantile(r,.05),wall)
    C['минимум, сглаж. 1-2-1']=kern(C['прошлый ВТД (минимум)'],[1,2,1])
    C['5-й перц., сглаж. 1-2-1']=kern(C['5-й перцентиль'],[1,2,1])
    C['5-й перц., сглаж. 1-2-3-2-1']=kern(C['5-й перцентиль'],[1,2,3,2,1])
    C['минимум в окне ±3 м']=win(3,lambda r,d:r.min(),wall)
    C['минимум в окне ±5 м']=win(5,lambda r,d:r.min(),wall)
    C['5-й перц. в окне ±5 м']=win(5,lambda r,d:np.quantile(r,.05),wall)
    C['5-й перц. в окне ±10 м']=win(10,lambda r,d:np.quantile(r,.05),wall)
    C['среднее 3 худших в окне ±5 м']=win(5,lambda r,d:np.sort(r)[:3].mean(),wall)
    C['число дефектов глубже 20% в окне ±5 м (лог, со знаком)']=-np.log1p(win(5,lambda r,d:(d>=20).sum(),0.0))
    C['средняя глубина в окне ±5 м (со знаком)']=-win(5,lambda r,d:d.mean(),0.0)
    C['смесь: 5-й перц. ±5 м + среднее 3 худших ±5 м']=z(C['5-й перц. в окне ±5 м'])+z(C['среднее 3 худших в окне ±5 м'])
    C['смесь: минимум + 5-й перц. ±5 м']=z(C['прошлый ВТД (минимум)'])+z(C['5-й перц. в окне ±5 м'])
    return C
def score(y,C,mask=None,ref=None):
    out={}
    for k,v in C.items():
        yy,vv=(y,v) if mask is None else (y[mask],v[mask]); out[k]=pearson_r2(yy,vv)
    return out
T={}
# ---- T1: 11604, сдвиг только на концах
s=anc[vid]; fact=data.segment_state(s['b'],Lv,wall_v)['thk_min'].to_numpy(); clean=fact>=7.99
first,last=js['sec'][js['id_b']==vid].min(),js['sec'][js['id_b']==vid].max(); ends=js[(js['sec']<=first)|(js['sec']>=last)]
a=V[(V['id']==vid)&(V['date']==A)].copy(); a['xe']=a['x_raw']+pl._shift_fn(ends)(a['x_raw'])-acc[vid]
C=cands(a.xe,a.rw,a.dp,Lv,wall_v)
mc=gm.simulate(a['xe'],a['dp'],a['wt'],Lv,wall_v,n_sim=500,seed=42,pos_sigma=1.0,pos_bump=np.ones(len(a))).mean(axis=0)
C['main (смесь модели и перцентиля)']=pl.blend(mc,C['5-й перцентиль'],0.5)
T['11604 концы, все 50']=score(fact,C); T['11604 концы, без 2 чистых']=score(fact,C,~clean)
fm=np.where(clean,fact[~clean].mean(),fact); T['11604 концы, чистые=среднее']=score(fm,C)
# ---- T2: 2018 -> 2022 на целевом, сдвиг на концах
D18=pd.Timestamp('2018-09-23'); m=pd.read_csv(OUT+'joint_shift_2018_2022.csv'); m['sh18m22']=-m['shift_22_minus_18']; t=m[(m.js22>=44.7)&(m.js22<5293.1)]
a18=V[V.id.isin(pipe_ids)&(V.date==D18)].copy()
a18['xt']=a18.x_raw-np.interp(a18.x_raw,[t.js18.iloc[0],t.js18.iloc[-1]],[t.sh18m22.iloc[0],t.sh18m22.iloc[-1]])-44.7
a18=a18[(a18.xt>=0)&(a18.xt<L_t)]
st22=data.segment_state(V[(V.id==tid)&(V.date==A)],L_t,wall_t); f22=st22.thk_min.to_numpy(); c22=~st22.has_def.to_numpy()
C18=cands(a18.xt,a18.rw,a18.dp,L_t,wall_t)
C18['main (смесь модели и перцентиля)']=z(C18['прошлый ВТД (минимум)'])+z(C18['5-й перцентиль'])   # закона роста 2018->2022 нет: смесь двух базовых
T['целевой 2018→2022, все 525']=score(f22,C18); T['целевой 2018→2022, без чистых 2022']=score(f22,C18,~c22)
# ---- T3: двойник
src=V[V['id'].isin(pipe_ids)&(V['date']==A)].copy(); src['xt']=src['x_raw'].to_numpy()+pl._shift_fn(js)(src['x_raw'])-acc[tid]; src=src[(src['xt']>=0)&(src['xt']<L_t)].reset_index(drop=True); p1=pd.read_csv(ROOT+'/outputs/case1_segments.csv').thickness_pred.values
x,dp,wt,sec=(src[c].to_numpy() for c in ('xt','dp','wt','sec'))
Ct=cands(x,src.rw.to_numpy(),dp,L_t,wall_t); Ct['main (смесь модели и перцентиля)']=p1
def truth(seed,k=0.0,rep=()):
    xs=x+k; keep=(xs>=0)&(xs<L_t)
    return gm.simulate(xs[keep],dp[keep],wt[keep],L_t,wall_t,n_sim=1,seed=seed,pos_sigma=0.0,repair_mask=np.isin(sec,list(rep))[keep],repair_prob=1.0,repair_groups=sec[keep])[0]
e=lambda y:y>=7.99
for name,k,rep,f in (('двойник: без сюрпризов',0,(),lambda y:y),('двойник: 3 трубы, сдвиг 3 м',3,(1370.,1420.,1890.),lambda y:y),('двойник: сдвиг 5 м',5,(),lambda y:y),
                     ('двойник: сдвиг 10 м',10,(),lambda y:y),('двойник: чистые=среднее',0,(),lambda y:np.where(e(y),y[~e(y)].mean(),y)),
                     ('двойник: чистые=среднее, сдвиг 5 м',5,(),lambda y:np.where(e(y),y[~e(y)].mean(),y))):
    ys=[f(truth(9000+i,k,rep)) for i in range(8)]
    T[name]={kk:np.mean([pearson_r2(y,v) for y in ys]) for kk,v in Ct.items()}
R=pd.DataFrame(T); R=R.loc[['main (смесь модели и перцентиля)']+[i for i in R.index if not i.startswith('main')]]
print(R.round(2).T.to_string()); R.round(3).to_csv(OUT+'coarse_features_r2.csv')

# ---- 3. гибрид против main

def hyb(main,pers): 
    h=z(main)+z(kern(pers,[1,2,1])); return main.mean()+main.std()*z(h)
rows=[]
def add(test,y,main,pers,mask=None):
    h=hyb(main,pers)
    if mask is not None: y,main,h=y[mask],main[mask],h[mask]
    rows.append({'проверка':test,'n':len(y),'r2 main':pearson_r2(y,main),'r2 гибрид':pearson_r2(y,h),'MAE main':mae(y,main),'MAE гибрид':mae(y,h)})
    return y,main,h
# 11604: концы
y,m_,h_=add('11604, сдвиг на концах, все',fact,C['main (смесь модели и перцентиля)'],C['прошлый ВТД (минимум)'])
rng=np.random.default_rng(0); dd=[]
for _ in range(4000):
    i=rng.integers(0,50,50); dd.append(pearson_r2(y[i],h_[i])-pearson_r2(y[i],m_[i]))
print('11604 концы: разница r2 (гибрид - main), бутстреп 5/50/95%%: %s'%np.nanpercentile(dd,[5,50,95]).round(3))
add('11604, сдвиг на концах, без чистых',fact,C['main (смесь модели и перцентиля)'],C['прошлый ВТД (минимум)'],~clean)
add('11604, чистые=среднее',fm,C['main (смесь модели и перцентиля)'],C['прошлый ВТД (минимум)'])
# 11604: проверка по блокам из репозитория (точный сдвиг)
val=pd.read_csv(ROOT+'/outputs/validation_case1_points.csv')
add('11604, проверка по блокам (репозиторий)',fact,val.pred_blend.to_numpy(),val.pred_pers.to_numpy())
add('11604, по блокам, без чистых',fact,val.pred_blend.to_numpy(),val.pred_pers.to_numpy(),~clean)
# 11604: «жёсткая»: сдвиг только по швам до участка (экстраполяция)
outer=js[js['sec']<=first]; a['xh']=a['x_raw']+pl._shift_fn(outer)(a['x_raw'])-acc[vid]
Ch=cands(a.xh,a.rw,a.dp,Lv,wall_v); mc=gm.simulate(a['xh'],a['dp'],a['wt'],Lv,wall_v,n_sim=500,seed=42,pos_sigma=1.0,pos_bump=np.ones(len(a))).mean(axis=0)
mh=pl.blend(mc,Ch['5-й перцентиль'],0.5)
add('11604, жёсткая (сдвиг снаружи), все',fact,mh,Ch['прошлый ВТД (минимум)']); add('11604, жёсткая, без чистых',fact,mh,Ch['прошлый ВТД (минимум)'],~clean)
# закон только с 9490
g9=pl._fit([anc[1250009490]]); mc=g9.simulate(a['xe'],a['dp'],a['wt'],Lv,wall_v,n_sim=500,seed=42,pos_sigma=1.0,pos_bump=np.ones(len(a))).mean(axis=0)
add('11604, концы, закон только с 9490',fact,pl.blend(mc,C['5-й перцентиль'],0.5),C['прошлый ВТД (минимум)'])
# 2018 -> 2022
m18=z(C18['прошлый ВТД (минимум)'])+z(C18['5-й перцентиль']); m18=f22.mean()+0.44*z(m18)
add('целевой 2018→2022, все (main = смесь базовых)',f22,m18,C18['прошлый ВТД (минимум)']); add('целевой 2018→2022, без чистых',f22,m18,C18['прошлый ВТД (минимум)'],~c22)
# двойник
pers_t=Ct['прошлый ВТД (минимум)']; hT=hyb(p1,pers_t)
scen=(('двойник: без сюрпризов',0,(),lambda y:y),('двойник: 3 трубы заменены',0,(1370.,1420.,1890.),lambda y:y),('двойник: 3 трубы, сдвиг 3 м',3,(1370.,1420.,1890.),lambda y:y),
      ('двойник: сдвиг 5 м',5,(),lambda y:y),('двойник: сдвиг 10 м',10,(),lambda y:y),('двойник: чистые=среднее',0,(),lambda y:np.where(e(y),y[~e(y)].mean(),y)),
      ('двойник: чистые=среднее, сдвиг 5 м',5,(),lambda y:np.where(e(y),y[~e(y)].mean(),y)),
      ('двойник: заменены все трубы >=40%',0,tuple(np.unique(sec[pd.Series(dp).groupby(sec).transform('max').to_numpy()>=40])),lambda y:y))
for name,k,rep,f in scen:
    ys=[f(truth(9000+i,k,rep)) for i in range(10)]
    rows.append({'проверка':name,'n':525,'r2 main':np.mean([pearson_r2(y,p1) for y in ys]),'r2 гибрид':np.mean([pearson_r2(y,hT) for y in ys]),'MAE main':np.mean([mae(y,p1) for y in ys]),'MAE гибрид':np.mean([mae(y,hT) for y in ys])})
R=pd.DataFrame(rows); R['Δr2']=R['r2 гибрид']-R['r2 main']; R['ΔMAE']=R['MAE гибрид']-R['MAE main']
print(R.round(3).to_string()); R.round(3).to_csv(OUT+'hybrid_vs_main.csv',index=False)
print('гибрид на целевом: mean %.3f sd %.3f min %.2f max %.2f; corr с main %.3f; сегменты 130,131,135,183,184: main %s гибрид %s'%(hT.mean(),hT.std(),hT.min(),hT.max(),np.corrcoef(hT,p1)[0,1],p1[[130,131,135,183,184]].round(2),hT[[130,131,135,183,184]].round(2)))

import numpy as np, pandas as pd
from pathlib import Path
rng=np.random.default_rng(42)
idx=pd.date_range('2026-08-20', periods=30*24, freq='h')
h=np.arange(len(idx)); shift=((idx.hour>=7)&(idx.hour<=22)).astype(float)
production=np.clip(900+260*shift+70*np.sin(h/18)+rng.normal(0,35,len(h)),650,None)
compressor=110+0.045*production+rng.normal(0,6,len(h))
chiller=135+0.035*production+18*np.sin(h/24)+rng.normal(0,5,len(h))
boiler=85+0.025*production+rng.normal(0,4,len(h))
# recurring demo anomalies in final 72h
chiller[-54:-30]+=34
compressor[-28:-12]+=27
other=145+rng.normal(0,8,len(h))
total=compressor+chiller+boiler+other
cop=np.clip(4.6-(chiller-(135+0.035*production))/45+rng.normal(0,.08,len(h)),2.4,5.2)
energy_intensity=total/production
pd.DataFrame({'timestamp':idx,'production_kg_h':production.round(1),'compressor_kw':compressor.round(1),'chiller_kw':chiller.round(1),'boiler_kw':boiler.round(1),'other_kw':other.round(1),'total_kw':total.round(1),'chiller_cop':cop.round(2),'energy_intensity_kwh_kg':energy_intensity.round(3)}).to_csv(Path(__file__).with_name('factory_energy.csv'),index=False)

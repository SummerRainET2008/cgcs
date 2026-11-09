#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import numpy as np
from acbpo.reproducibility import stable_seed
S=(99801,99802,99803);F=("best_of_n","entropy_value","value_value");E=("E0_standard_actor_ratio","E1_trajectory_propensity","E3_step_cumulative_lambda_0.75","E4_extended_candidate_set")
def clusters(root,h,f,e):
 z={i:[] for i in range(248,348)}
 for s in S:
  d=json.loads((root/f"s{s}"/h/f"{f}_{e}.json").read_text())
  for x in d["examples"]:z[int(x["example_id"])].append(x)
 return z
def arrays(z):
 keys=sorted(z);rows=[z[i] for i in keys];return {k:np.array([[float(bool(x["result"]["correct"])) if k=="correct" else float(bool(x["result"]["answered"])) if k=="answered" else len(x["steps"]) if k=="steps" else x["completion_tokens"] for x in q] for q in rows]) for k in ("correct","answered","steps","tokens")}
def stats(a):
 n=a["correct"].size;c=a["correct"].sum();ans=a["answered"].sum()
 return {"examples":n,"accuracy":float(c/n),"answer_rate":float(ans/n),"conditional_accuracy":float(c/ans) if ans else math.nan,"mean_steps":float(a["steps"].mean()),"mean_completion_tokens":float(a["tokens"].mean())}
def boot(L,R,seed):
 rng=np.random.default_rng(seed);idx=rng.integers(0,100,(5000,100));out={}
 for name,key in (("accuracy","correct"),("answer_rate","answered"),("mean_steps","steps"),("mean_completion_tokens","tokens")):
  d=(L[key].sum(1)[idx].sum(1)-R[key].sum(1)[idx].sum(1))/300
  v=float(L[key].mean()-R[key].mean());out[name]={"difference":v,"ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975))}
 lc=L["correct"].sum(1);la=L["answered"].sum(1);rc=R["correct"].sum(1);ra=R["answered"].sum(1)
 d=lc[idx].sum(1)/la[idx].sum(1)-rc[idx].sum(1)/ra[idx].sum(1);v=float(L["correct"].sum()/L["answered"].sum()-R["correct"].sum()/R["answered"].sum());out["conditional_accuracy"]={"difference":v,"ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--s4-root",type=Path,required=True);p.add_argument("--s3-analysis",type=Path,required=True);p.add_argument("--output-json",type=Path,required=True);p.add_argument("--output-report",type=Path,required=True);a=p.parse_args();M={(h,f,e):arrays(clusters(a.s4_root,h,f,e)) for h in ("H8","H16") for f in F for e in E}
 summaries=[{"horizon":h,"family":f,"estimator":e,**stats(M[h,f,e])} for h in ("H8","H16") for f in F for e in E];contr=[]
 for h in ("H8","H16"):
  for f in F:
   for le,re in ((E[1],E[0]),(E[2],E[0]),(E[2],E[1]),(E[3],E[0])):
    L=M[h,f,le];R=M[h,f,re];delta=(L["correct"]-R["correct"]).ravel();contr.append({"horizon":h,"family":f,"left":le,"right":re,"wins":int((delta>0).sum()),"losses":int((delta<0).sum()),"ties":int((delta==0).sum()),"metrics":boot(L,R,stable_seed("v3-bootstrap",h,f,le,re))})
 tr=json.loads(a.s3_analysis.read_text())["training_stability"];g={}
 for f in ("burnin","k1","best_of_n","entropy_value","value_value"):
  v=[x["clip_fraction"] for x in tr if x["family"]==f];g[f]={"rows":len(v),"minimum":min(v),"median":float(np.median(v)),"maximum":max(v),"below_0.20":sum(x<.2 for x in v)}
 out={"schema":"cgcs-v3-offline-diagnostics-v1","s4_h16_estimator_summaries":summaries,"s4_paired_estimator_contrasts":contr,"s3_sequence_clipping":g,"s3_token_level_available":False,"s3_token_level_note":"S3 retained only sequence ratios; S5 records token-level diagnostics prospectively."};a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2)+"\n")
 lines=["# V3 Offline Diagnostics","","## Existing S4 H16 estimator contrast",""]
 for f in F:
  lines+=["### "+f,"","| contrast | d accuracy [95% CI] | d answer rate [95% CI] |","|---|---:|---:|"]
  for r in contr:
   if r["horizon"]=="H16" and r["family"]==f:
    x=r["metrics"]["accuracy"];y=r["metrics"]["answer_rate"];lines.append(f'| {r["left"]} - {r["right"]} | {x["difference"]:.4f} [{x["ci95_low"]:.4f}, {x["ci95_high"]:.4f}] | {y["difference"]:.4f} [{y["ci95_low"]:.4f}, {y["ci95_high"]:.4f}] |')
  lines.append("")
 lines+=["## S3 sequence clipping","","| family | rows | min | median | max | below .20 |","|---|---:|---:|---:|---:|---:|"]
 for f,x in g.items():lines.append(f'| {f} | {x["rows"]} | {x["minimum"]:.4f} | {x["median"]:.4f} | {x["maximum"]:.4f} | {x["below_0.20"]} |')
 lines+=["","S3 fractions are two-sided sequence-ratio diagnostics, not token-drop or actor-saturation fractions. Token ratios were not retained and cannot be recovered post hoc; S5 records them prospectively."];a.output_report.write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()

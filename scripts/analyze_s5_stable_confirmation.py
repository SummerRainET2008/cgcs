#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
import numpy as np
from acbpo.reproducibility import stable_seed
S=(99802,99803);A=("k1","search_e0","search_e1","search_e3")
def load(root):
 m={}
 for seed in S:
  for arm in A:
   b=root/f"s{seed}"/arm
   for rd in (1,2):m[seed,arm,f"T{rd}"]=json.loads((b/f"round{rd}"/"train"/"summary.json").read_text())
   for h in (8,16):
    d=json.loads((b/"eval"/f"H{h}.json").read_text());c=d["config"];assert (c["offset"],c["count"],c["seed"],c["max_steps"],c["step_tokens"],c["batch_size"])==(48,200,100901,h,96,24);assert [x["example_id"] for x in d["examples"]]==list(range(48,248));m[seed,arm,f"H{h}"]=d
 return m
def ar(m,arm,h):
 return np.array([[[float(bool(x["result"]["correct"])),float(bool(x["result"]["answered"])),len(x["steps"]),x["completion_tokens"]] for x in [m[seed,arm,f"H{h}"]["examples"][i] for seed in S]] for i in range(200)])
def sm(x):
 c=x[:,:,0];a=x[:,:,1];return {"examples":int(c.size),"accuracy":float(c.mean()),"answer_rate":float(a.mean()),"conditional_accuracy":float(c.sum()/a.sum()),"mean_steps":float(x[:,:,2].mean()),"mean_completion_tokens":float(x[:,:,3].mean())}
def ct(l,r,seed):
 rng=np.random.default_rng(seed);ix=rng.integers(0,200,(10000,200));o={}
 for n,j in (("accuracy",0),("answer_rate",1),("mean_steps",2),("mean_completion_tokens",3)):
  d=(l[:,:,j].sum(1)[ix].sum(1)-r[:,:,j].sum(1)[ix].sum(1))/400;o[n]={"difference":float(l[:,:,j].mean()-r[:,:,j].mean()),"ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975))}
 lc,la,rc,ra=l[:,:,0].sum(1),l[:,:,1].sum(1),r[:,:,0].sum(1),r[:,:,1].sum(1);d=lc[ix].sum(1)/la[ix].sum(1)-rc[ix].sum(1)/ra[ix].sum(1);o["conditional_accuracy"]={"difference":float(l[:,:,0].sum()/l[:,:,1].sum()-r[:,:,0].sum()/r[:,:,1].sum()),"ci95_low":float(np.quantile(d,.025)),"ci95_high":float(np.quantile(d,.975))};q=(l[:,:,0]-r[:,:,0]).ravel();o.update(wins=int((q>0).sum()),losses=int((q<0).sum()),ties=int((q==0).sum()));return o
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output-json",type=Path,required=True);p.add_argument("--output-report",type=Path,required=True);a=p.parse_args();m=load(a.root);sel=json.loads((a.root.parent/"calibration"/"SELECTION.json").read_text());ev=[{"arm":x,"horizon":f"H{h}",**sm(ar(m,x,h))} for x in A for h in (8,16)];cs=[]
 for h in (8,16):
  for l,r in (("search_e1","search_e0"),("search_e3","search_e0"),("search_e3","search_e1"),("search_e0","k1")):cs.append({"horizon":f"H{h}","left":l,"right":r,**ct(ar(m,l,h),ar(m,r,h),stable_seed("s5-bootstrap",h,l,r))})
 for x in A:cs.append({"horizon":"H16-H8","left":x,"right":x,**ct(ar(m,x,16),ar(m,x,8),stable_seed("s5-horizon-bootstrap",x))})
 tr=[]
 for seed in S:
  for arm in A:
   for rd in (1,2):
    d=m[seed,arm,f"T{rd}"];assert float(d["config"]["learning_rate"])==1e-6;tr.append({"seed":seed,"arm":arm,"round":rd,**{k:d[k] for k in ("states","optimizer_updates","sequence_ratio_clip_fraction","sequence_ratio_saturation_fraction","selected_actor_token_ratio_clip_fraction","selected_actor_token_saturation_fraction","mean_selected_kl_proxy","external_weight_ess","external_weight_maximum","elapsed_seconds")}})
 out={"schema":"cgcs-s5-stable-confirmation-analysis-v1","selection":sel,"evaluation_summaries":ev,"paired_contrasts":cs,"training_stability":tr};a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2)+"\n")
 z=["# S5 Stable-Update Confirmation","","Calibration gate: **"+("PASS" if sel["gate_pass"] else "FAIL")+"**; selected LR: "+str(sel["selected_learning_rate"])+".","","| arm | horizon | accuracy | answer rate | conditional accuracy | mean steps |","|---|---:|---:|---:|---:|---:|"]
 for x in ev:z.append(f'| {x["arm"]} | {x["horizon"]} | {x["accuracy"]:.4f} | {x["answer_rate"]:.4f} | {x["conditional_accuracy"]:.4f} | {x["mean_steps"]:.2f} |')
 z+=["","## Paired contrasts","","| horizon | contrast | d accuracy [95% CI] | d answer [95% CI] |","|---|---|---:|---:|"]
 for x in cs:y=x["accuracy"];w=x["answer_rate"];z.append(f'| {x["horizon"]} | {x["left"]} - {x["right"]} | {y["difference"]:.4f} [{y["ci95_low"]:.4f}, {y["ci95_high"]:.4f}] | {w["difference"]:.4f} [{w["ci95_low"]:.4f}, {w["ci95_high"]:.4f}] |')
 a.output_report.write_text("\n".join(z)+"\n")
if __name__=="__main__":main()

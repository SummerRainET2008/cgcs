import argparse,json,math
from pathlib import Path
A=("k1","search_e0","search_e1","search_e3"); L=("1e-6","2e-6")
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=[];ok={}
for lr in L:
 z=[]
 for arm in A:
  q=a.root/f"lr_{lr}"/arm/"train"/"summary.json";d=json.loads(q.read_text());r={k:d[k] for k in ("states","optimizer_updates","sequence_ratio_clip_fraction","sequence_ratio_saturation_fraction","selected_actor_token_ratio_clip_fraction","selected_actor_token_saturation_fraction","mean_selected_kl_proxy","external_weight_ess","external_weight_maximum","mean_actor_loss","mean_value_loss")};r.update(learning_rate=float(lr),arm=arm,summary=str(q));r["pass"]=r["optimizer_updates"]>0 and all(math.isfinite(float(r[k])) for k in ("mean_actor_loss","mean_value_loss","mean_selected_kl_proxy")) and r["sequence_ratio_clip_fraction"]<.2 and r["selected_actor_token_ratio_clip_fraction"]<.2;rows.append(r);z.append(r)
 ok[lr]=all(x["pass"] for x in z)
v=[float(x) for x in L if ok[x]];out={"schema":"cgcs-s5-stable-lr-selection-v1","gate_pass":bool(v),"per_learning_rate_pass":ok,"selected_learning_rate":max(v) if v else 1e-6,"rows":rows};a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))

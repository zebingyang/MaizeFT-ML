"""Replot archived prediction results; no fitting, network propagation or rescoring."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--qa-tools', type=Path, required=True)
parser.add_argument('--capital-out', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.qa_tools))
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'source_data').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none',
    'text.color':'black','axes.labelcolor':'black','savefig.facecolor':'white'})
MODELS=['L2_LR','ElasticNet_LR','Linear_SVM','Bagging','Shallow_XGBoost','Shallow_NeuralNet']
NAMES=['L2-LR','Elastic Net','Linear SVM','Bagging','XGBoost','NeuralNet']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#B28B20','#707070']
hashes={}
def read_csv(name):
    p=args.root/'FT_six_model_results'/name
    if not p.exists(): p=args.root/'source_data'/name
    hashes[str(p.relative_to(args.root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    d=pd.read_csv(p)
    d.to_csv(OUT/'source_data'/name,index=False)
    return d
oof=read_csv('FT_oof_predictions.csv')
folds=read_csv('FT_outer_fold_metrics.csv')
comp=read_csv('FT_model_comparison.csv')
cal=read_csv('FT_calibration_curve.csv')
cand=read_csv('candidate9_predictions.csv')
allmodels=read_csv('candidate9_all_model_summary.csv')
blocks=read_csv('candidate9_feature_block_contributions.csv')
p=args.root/'Mechanism_results/Mechanism_results_v2/candidate9_mechanism_results_v2.json'
if not p.exists(): p=args.root/'source_data/mechanism_plot_data.json'
hashes[str(p.relative_to(args.root))]=hashlib.sha256(p.read_bytes()).hexdigest()
raw=json.loads(p.read_text(encoding='utf-8'))
(OUT/'source_data/mechanism_plot_data.json').write_text(json.dumps({k:raw[k] for k in ['candidate_summary','l1_scores_long','coverage_qc','top_components_long','tissue_contexts']},ensure_ascii=False,indent=2),encoding='utf-8')
summary=pd.DataFrame(raw['candidate_summary']).set_index('gene_id_v5')
l1=pd.DataFrame(raw['l1_scores_long'])
coverage=pd.DataFrame(raw['coverage_qc']).set_index('gene_id_v5')
for key in ['candidate_summary','l1_scores_long','coverage_qc','top_components_long','tissue_contexts']:
    pd.DataFrame(raw[key]).to_csv(OUT/'source_data'/('mechanism_'+key+'.csv'),index=False)
assert oof.groupby('Model').size().eq(130).all()
assert oof.groupby('Model').Label.sum().eq(65).all()
assert not oof.duplicated(['Model','Gene_ID_v5']).any()
assert len(folds)==30 and len(cand)==9 and len(l1)==54
assert oof.Predicted_probability.between(0,1).all()
genes=cand.sort_values('Prediction_rank').Gene_ID_v5.tolist()
net=['Zm00001eb349860','Zm00001eb325180','Zm00001eb341290','Zm00001eb325140','Zm00001eb325090','Zm00001eb202620']
desc=['Zm00001eb303670','Zm00001eb202630']
mechs=['M1','M2','M3','M4','M5','M6']
blue=LinearSegmentedColormap.from_list('L1',['#E3F5F6','#ABE0E4','#7FC6D3','#779FC6','#6F78B9'])
orange=LinearSegmentedColormap.from_list('L2',['#FFF0DC','#FED297','#FCB461','#F7935A','#F37252'])
for cm in [blue,orange]: cm.set_bad('#D6D6D6')
pdfs=[]
export_sizes={}
def fig(height):
    width_inches=183/25.4
    return plt.figure(figsize=(width_inches,height/25.4))
def axis(f,x,y,w,h):
    H=f.get_figheight()*25.4
    return f.add_axes([x/183,1-(y+h)/H,w/183,h/H])
def heading(ax,letter,title,pad=10):
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-13,pad),textcoords='offset points',fontsize=10,fontweight='bold',va='bottom')
    ax.set_title(title,loc='left',pad=pad)
def save(f,name,axes,groups):
    f.canvas.draw()
    require_matplotlib_panel_alignment(f,axes=axes,panel_ids=list('abcdef')[:len(axes)],
        row_groups=groups,column_groups=[] if groups else [['a','b'],['b','c']],tolerance_pt=1.5,strict=True,
        json_out=str(OUT/name)+'.alignment.json',overlay_svg=str(OUT/name)+'.alignment.svg')
    labels=[next(t for t in ax.texts if t.get_text()==letter)
            for ax,letter in zip(axes,'abcdef')]
    def tight():
        f.canvas.draw()
        pixels=np.asarray(f.canvas.buffer_rgba())
        rows=np.flatnonzero(np.any(pixels[:,:,:3]<255,axis=(1,2)))
        return Bbox.from_extents(0,(pixels.shape[0]-rows[-1]-1)/f.dpi,
                                 f.get_figwidth(),(pixels.shape[0]-rows[0])/f.dpi)
    dpi=f.dpi
    f.set_dpi(300)
    extents=[tight()]
    if args.capital_out:
        for t in labels: t.set_text(t.get_text().upper())
        extents.append(tight())
        for t in labels: t.set_text(t.get_text().lower())
    f.set_dpi(dpi)
    margin=1/25.4
    bottom=min(b.y0 for b in extents)-margin
    top=max(b.y1 for b in extents)+margin
    crop=Bbox.from_extents(0,bottom,f.get_figwidth(),top)
    export_sizes[name]=[183,crop.height*25.4]
    def export(directory):
        for ext in ['pdf','svg','png','tiff']:
            kw={'dpi':600} if ext in ['png','tiff'] else {}
            if ext=='tiff': kw['pil_kwargs']={'compression':'tiff_lzw'}
            f.savefig(directory/(name+'.'+ext),bbox_inches=crop,**kw)
        f.savefig(directory/(name+'_preview.png'),dpi=300,bbox_inches=crop)
    export(OUT)
    if args.capital_out:
        for t in labels: t.set_text(t.get_text().upper())
        export(args.capital_out)
    pdfs.append(name+'.pdf')
    plt.close(f)
def colorbar(f,im,x,y,h,label,ticks):
    ca=axis(f,x,y,2.4,h)
    cb=f.colorbar(im,cax=ca,ticks=ticks)
    cb.set_label(label,fontsize=7,labelpad=5)
    cb.outline.set_linewidth(.5)
def heat(ax,values,rows,cols,cmap,vmin,vmax,fmt='.2f',italic=True):
    values=np.asarray(values,float)
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto',interpolation='none')
    ax.set_xticks(range(len(cols)),cols)
    ax.set_yticks(range(len(rows)),rows)
    ax.tick_params(length=0,pad=5)
    for t in ax.get_yticklabels(): t.set_fontstyle('italic' if italic else 'normal')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks(np.arange(-.5,len(cols),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7)
    ax.tick_params(which='minor',length=0)
    for (i,j),v in np.ndenumerate(values):
        if np.isfinite(v):
            rgba=im.cmap(im.norm(v)); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=6,color='white' if lum<.48 else 'black')
        else: ax.text(j,i,'NA',ha='center',va='center',fontsize=6,color='#444444')
    return im

# SF1: discrimination, calibration and threshold-dependent diagnostics.
f=fig(216)
axes=[axis(f,x,y,65,43) for y in [26,91,156] for x in [20,110]]
handles=[Line2D([0],[0],color=c,lw=1.4,label=n) for c,n in zip(COLORS,NAMES)]
f.legend(handles=handles,loc='center',bbox_to_anchor=(.52,1-81.5/216),ncol=6,frameon=False,columnspacing=1.2,handlelength=1.5)
for m,c in zip(MODELS,COLORS):
    d=oof[oof.Model==m]; y=d.Label.to_numpy(); pr=d.Predicted_probability.to_numpy()
    x,t,_=roc_curve(y,pr); axes[0].plot(x,t,color=c,lw=1)
    precision,recall,_=precision_recall_curve(y,pr); axes[1].step(recall,precision,where='post',color=c,lw=1)
    r=comp.set_index('Model').loc[m]
    assert abs(roc_auc_score(y,pr)-r.Pooled_OOF_ROC_AUC)<1e-7
    assert abs(average_precision_score(y,pr)-r.Pooled_OOF_PR_AUC)<1e-7
    d=cal[cal.Model==m]; axes[2].plot(d.Mean_predicted_probability,d.Observed_positive_fraction,'o-',color=c,lw=.8,ms=2.5)
for a in axes[:3]: a.set(xlim=(0,1),ylim=(0,1)); a.set_xticks([0,.5,1]); a.set_yticks([0,.5,1])
axes[0].plot([0,1],[0,1],'--',color='#999999',lw=.6)
axes[1].axhline(.5,color='#999999',ls='--',lw=.6)
axes[2].plot([0,1],[0,1],'--',color='#999999',lw=.6)
for a,l,t,x,y in zip(axes[:3],'abc',['Pooled OOF ROC','Pooled OOF precision–recall','OOF calibration'],['False-positive rate','Recall','Mean predicted probability'],['True-positive rate','Precision','Observed positive fraction']):
    heading(a,l,t); a.set(xlabel=x,ylabel=y)
for a,l,col,title in zip(axes[3:],'def',['F1_at_0.5','MCC_at_0.5','Brier_score'],['F1 at threshold 0.5','MCC at threshold 0.5','Brier score (lower is better)']):
    for i,(m,c) in enumerate(zip(MODELS,COLORS)):
        v=folds[folds.Model==m][col].to_numpy()
        a.scatter(v,np.full(5,i)+np.linspace(-.14,.14,5),s=10,color=c,alpha=.65,linewidths=0)
        a.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='D',ms=3,color=c,lw=.9,capsize=2)
    a.set_yticks(range(6),NAMES); a.invert_yaxis(); a.set_xlabel('Outer-fold score (mean ± s.d.)')
    a.set_xlim((0,.4) if col=='Brier_score' else (-.2,1) if col.startswith('MCC') else (0,1))
    heading(a,l,title)
save(f,'SuppFig_FT_01_OOF_validation',axes,[['a','b'],['c','d'],['e','f']])

# SF2: retained fold and alternative-model predictions, not a model ensemble.
f=fig(229); axes=[axis(f,43,y,116,47) for y in [20,90,160]]
mat=allmodels.pivot(index='Gene_ID_v5',columns='Model',values='Mean_probability').reindex(index=genes,columns=MODELS)
im=heat(axes[0],mat,genes,NAMES,'Blues',0,1)
heading(axes[0],'a','Candidate probabilities across six models')
colorbar(f,im,166,20,47,'Mean probability',[0,.5,1])
mat=cand.set_index('Gene_ID_v5').loc[genes,[f'Probability_F{i}' for i in range(1,6)]]
im=heat(axes[1],mat,genes,[f'F{i}' for i in range(1,6)],'Blues',0,1)
heading(axes[1],'b','Locked L2-LR: individual fold-model predictions')
colorbar(f,im,166,90,47,'Probability',[0,.5,1])
mat=blocks.pivot(index='Gene_ID_v5',columns='Feature_block',values='Mean_absolute_feature_delta').reindex(index=genes,columns=['PPI','Shortest_distance','Transcriptome','Translatome'])*1000
mat.to_csv(OUT/'source_data/SF2_feature_block_ablation_x1000.csv')
im=heat(axes[2],mat,genes,['PPI','Seed distance','Transcriptome','Translatome'],'Blues',0,50,fmt='.1f')
assert np.nanmax(mat)<=50
heading(axes[2],'c','Mean absolute single-feature ablation effect')
colorbar(f,im,166,160,47,'Probability change per feature (×0.001)',[0,25,50])
save(f,'SuppFig_FT_02_candidate_robustness',axes,[])

# SF3: separate non-comparable mechanism result classes and retain NA.
f=fig(215); axes=[axis(f,45,23,114,42),axis(f,45,92,114,16),axis(f,45,140,114,54)]
for a,gg,cm,letter,title,y,h in [(axes[0],net,blue,'a','Network-supported L1 hypotheses',23,42),(axes[1],desc,orange,'b','Descriptive-only profiles (not integrated scores)',92,16)]:
    mat=l1.pivot(index='gene_id_v5',columns='mechanism_l1_id',values='primary_display_score_0_100').loc[gg,mechs]
    im=heat(a,mat,gg,mechs,cm,0,100,fmt='.1f')
    for i,g in enumerate(gg):
        # Mark all reported ties, not an arbitrary tie-broken mechanism.
        for j,v in enumerate(mat.loc[g]):
            if abs(v-mat.loc[g].max())<.011: a.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,ec='black',lw=1,zorder=4))
    heading(a,letter,title)
    colorbar(f,im,166,y,h,'Integrated score' if letter=='a' else 'Descriptive score',[0,50,100])
order=net+desc+['Zm00001eb349850']
cols=['PPI','RNA','RIBO','CHIA','TF','Tissue','Annotation']
vals=[]
for g in order:
    r=coverage.loc[g]; layers=str(r.network_layers_assessable).split(';')
    vals.append([int(c in layers) for c in cols[:4]]+[int(r.tf_evidence_assessable),int(r.tissue_evidence_assessable),int(r.annotation_evidence_assessable)])
im=heat(axes[2],vals,order,cols,ListedColormap(['#E7E7E7','#4C8EAA']),0,1,fmt='.0f')
heading(axes[2],'c','Assessability of public-data evidence')
cbax=axis(f,166,151,2.4,25); cb=f.colorbar(im,cax=cbax,ticks=[0,1]); cb.ax.set_yticklabels(['No','Yes']); cb.set_label('Assessable')
pd.DataFrame(vals,index=order,columns=cols).to_csv(OUT/'source_data/SF3_evidence_assessability.csv')
save(f,'SuppFig_FT_03_mechanism_coverage',axes,[])

# SF4: existing network-score ranges, not confidence intervals or new simulations.
f=fig(164); axes=[]
for i,g in enumerate(net):
    a=axis(f,[16,76,136][i%3],[25,102][i//3],40,42); axes.append(a)
    dd=l1[l1.gene_id_v5==g].set_index('mechanism_l1_id').loc[mechs]
    for j,(m,r) in enumerate(dd.iterrows()):
        low=r.network_sensitivity_min_0_100; high=r.network_sensitivity_max_0_100; mean=r.network_score_equal_layer_0_100
        assert low-.02<=mean<=high+.02
        c='#D55E00' if m==summary.loc[g,'top_mechanism_l1_id'] else '#4286AA'
        a.plot([low,high],[j,j],color=c,lw=1.4)
        a.plot([low,low],[j-.09,j+.09],color=c,lw=.7)
        a.plot([high,high],[j-.09,j+.09],color=c,lw=.7)
        a.plot(mean,j,'o',ms=3,color=c)
    a.set(yticks=range(6),yticklabels=mechs,xlim=(-2,102),ylim=(5.5,-.5),xlabel='Network score')
    a.set_xticks([0,50,100]); heading(a,'abcdef'[i],g,pad=19)
    a.set_title(g,loc='left',pad=19,fontstyle='italic',fontsize=6.5)
    n=int(summary.loc[g,'network_layers_assessable_n']); stability=round(summary.loc[g,'sensitivity_top_mechanism_stability']*3)
    a.text(0,1.035,f'{n}/4 layers; top retained {stability}/3',transform=a.transAxes,fontsize=6,va='bottom')
f.legend(handles=[Line2D([0],[0],color='#4286AA',marker='o',ms=3,lw=1,label='Equal-weight score and sensitivity range'),Line2D([0],[0],color='#D55E00',marker='o',ms=3,lw=1,label='Primary integrated mechanism')],loc='center',bbox_to_anchor=(.52,1-84/164),ncol=2,frameon=False)
save(f,'SuppFig_FT_04_network_sensitivity',axes,[['a','b','c'],['d','e','f']])

manifest={'source_sha256':hashes,'figures':pdfs,'independent_training_genes':130,'positive':65,'reference_negative':65,
    'models':6,'outer_folds':5,'network_ranked':6,'descriptive_only':2,'unresolved':1,
    'retraining':False,'rescoring':False,'figure_width_mm':183,'font_pt':7,'cell_font_pt':6,
    'numbering':'Provisional module-specific numbering; renumber for final manuscript.'}
manifest.update(export_sizes_mm=export_sizes,vertical_margins_mm=1)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if args.capital_out:
    (args.capital_out/'manifest.json').write_text(json.dumps(dict(manifest,panel_label_case='upper'),indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))

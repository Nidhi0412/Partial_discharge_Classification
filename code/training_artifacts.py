"""Save the reports/figures expected from the historical training workflow."""
import json
from pathlib import Path


def export_figures(out, history=None, predictions=None, scope='development validation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from training_pipeline import binary_metrics
    from training_safety import atomic_json
    from evaluation_tools import roc_points
    out=Path(out)
    if history:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,metric in zip(axes,['accuracy','loss']):
            for key,label in [(metric,'Training'),('val_'+metric,'Validation')]:
                if key in history:ax.plot(range(1,len(history[key])+1),history[key],label=label)
            ax.set(xlabel='Epoch',ylabel=metric.title());ax.legend()
        fig.tight_layout();fig.savefig(out/'training_history.png',dpi=160);plt.close(fig)
    if predictions:
        labels=[int(r['y_true']) for r in predictions]
        scores=[float(r['p_teflon']) for r in predictions]
        metrics=binary_metrics(labels,scores)
        report=dict(metrics,scope=scope,threshold=.5)
        atomic_json(out/'classification_report.json',report)
        lines=[scope,'Class\tPrecision\tRecall\tF1\tSupport']
        for i,name in enumerate(['Nomex','Teflon']):
            row=metrics['per_class'][name]
            lines.append(f"{name}\t{row['precision']:.6f}\t{row['recall']:.6f}\t{row['f1']:.6f}\t{labels.count(i)}")
        (out/'classification_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        fig,axis=plt.subplots(figsize=(5,4));cm=metrics['confusion_matrix']
        axis.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):axis.text(j,i,str(cm[i][j]),ha='center',va='center')
        axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Nomex','Teflon'],
                 yticklabels=['Nomex','Teflon'],xlabel='Predicted',ylabel='True',title=scope)
        fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
        if metrics['auc'] is not None:
            points=roc_points(predictions)
            fig,axis=plt.subplots(figsize=(5,4))
            axis.plot([x for x,y in points],[y for x,y in points],label=f"AUC={metrics['auc']:.4f}")
            axis.plot([0,1],[0,1],'--',color='gray')
            axis.set(xlabel='False positive rate',ylabel='True positive rate',title=scope)
            axis.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=160);plt.close(fig)


def export_model_files(tf, best, out, state, enabled=True):
    """Compatibility H5 files contain genuine HDF5, not renamed .keras archives."""
    from training_safety import snapshot_path
    import shutil
    out=Path(out)
    shutil.copyfile(out/'best.keras',out/'model.keras')
    if enabled:
        best.save(out/'model.h5')
    if state.get('best_accuracy_file'):
        source=snapshot_path(out,state['best_accuracy_file'])
        shutil.copyfile(source,out/'best_val_accuracy.keras')
        if enabled:
            accuracy_model=tf.keras.models.load_model(source,compile=False)
            accuracy_model.save(out/'best_weights.h5')
            del accuracy_model
    manifest=[]
    for metadata in sorted((out/'augmented_samples').glob('*/*.json')):
        manifest.append(json.loads(metadata.read_text(encoding='utf-8')))
    from training_safety import atomic_json
    atomic_json(out/'augmentation_samples_manifest.json',manifest)
    export_augmentation_previews(out,manifest)


def export_augmentation_previews(out,manifest):
    out=Path(out)
    if manifest:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        for material in sorted({r['material'] for r in manifest}):
            example=next(r for r in manifest if r['material']==material)
            panels=[('Original',example['original_preview'])]
            panels.extend((stage['operation'],stage['preview_path']) for stage in example.get('stages',[]))
            panels.append(('Final training input',example['augmented_preview']))
            cols=min(4,len(panels));rows=(len(panels)+cols-1)//cols
            fig,axes=plt.subplots(rows,cols,figsize=(3*cols,3*rows),squeeze=False)
            for ax in axes.flat:ax.axis('off')
            for ax,(title,path) in zip(axes.flat,panels):
                with Image.open(path) as image:ax.imshow(image)
                ax.set_title(title.replace('_',' '),fontsize=9)
            fig.suptitle(f'{material}: generated PNG augmentation example')
            fig.tight_layout();fig.savefig(out/f'augmentation_preview_{material}.png',dpi=140);plt.close(fig)

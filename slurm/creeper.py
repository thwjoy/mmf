import os
import re
import numpy as np

ROOT_DIR = "/fsx-labs/thwjoy/mmf/save"

metrics_dict = {
    "visual_entailment": ["visual_entailment/accuracy"],
    "vqa2": ["vqa2/accuracy", "vqa2/topk_accuracy"],
    "hateful_memes": ["hateful_memes/accuracy", "hateful_memes/roc_auc"]
}

models = {
    'random': "Random Init",
    'contrastive': "Contrastive COCO",
    'xtransform': "XTran COCO (Ours)",
    'unit': "UniT",
    'vilt': "ViLT",
    'uniter': "UNITER",
    'villa': "VILLA",
    'visual_bert': "VisualBERT"
}
            
table_tex_pref = \
"\\begin{tabular}{c|ccccccc} \n \
\t\\toprule \n \
\tInitialisattion & \multicolumn{2}{c}{Hateful Memes} & \multicolumn{2}{c}{VQAv2} & SNLI-VI\\\\ \n \
\t& (Acc) & (AUROC) & (Top-1) & (Top-5) & (Acc) \\\\  \n \
\t\midrule \n "

table_row = "\t{model} & {hateful_memes_accuracy} & {hateful_memes_roc_auc} & {vqa2_accuracy} & {vqa2_topk_accuracy} & {visual_entailment_accuracy} & \\\\ \n"

table_tex_suf = \
"\t\\bottomrule \\\\ \n \
\end{tabular}"   

class Default(dict):
    def __missing__(self, key):
        return '-'

def main():
    # go through each of the logs and 
    results_dict = {}
    for model in models.keys():
        model_results_dict = {}
        for dataset, metrics in metrics_dict.items():
            logfile = os.path.join(ROOT_DIR, model, dataset, "train.log")
            for metric in metrics:
                with open(logfile, 'r') as contenlog:
                    vals = []
                    for line in contenlog:
                        if re.findall(metric, line):
                            match = re.search("%s: (0.\d+)" % metric, line)
                            if match:
                                vals.append(float(match.group(1)))
                if len(vals) > 0:
                    metric = metric.split("/")[-1]
                    model_results_dict['%s_%s' % (dataset, metric)] = np.max(vals)
                # best_metric = np.max(vals)
                # print(best_metric)
        results_dict[model] = model_results_dict

    table = table_tex_pref
    for key, vals in results_dict.items():
        table += table_row.format_map(Default(model=models[key], **vals))

    table += table_tex_suf

    print(table)

                            

if __name__ == "__main__":
    main()






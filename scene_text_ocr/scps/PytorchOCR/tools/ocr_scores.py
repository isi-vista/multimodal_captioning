from wer import compute_scores
from os.path import basename, splitext

rec_hyps = "/nas/home/xpeng/projects/image_captioning/run.41.01/scps/PytorchOCR/output/det_results/icdar2015.results"
rec_refs = "/nas/home/xpeng/projects/image_captioning/data/Incidental-Scene-Text-2015/recognition/test.txt"


refs, hyps = {}, {}
with open(rec_hyps, "r") as fh1, open(rec_refs, "r") as fh2:
    for line in fh1:
        file_path = line.strip().split()[0]
        hyp = line.strip().split()[1]
        fname = splitext(basename(file_path))[0].split("_")
        fname = fname[0] + "_" + fname[1]
        
        if fname in hyps:
            hyps[fname].append(hyp)
        else:
            hyps[fname] = [hyp]

    for line in fh2:
        file_path = line.strip().split()[0]
        ref = line.strip().split()[1]
        fname = splitext(basename(file_path))[0].split("_")
        fname = fname[0] + "_" + fname[1]
        
        if fname in refs:
            refs[fname].append(ref)
        else:
            refs[fname] = [ref]


    total_cer = 0
    total_cnt = 0
    for image in hyps:
        for hyp in hyps[image]:
            best_cer = 100
            if image in refs:
                for ref in refs[image]:
                    trans = {}
                    trans[image] = [ref, hyp]
                    score = compute_scores(trans)
                    if score['CER'] < best_cer:
                        best_cer = score['CER'] 
            if best_cer > 99:
                total_cer += 1.0
            else:
                total_cer += best_cer

            total_cnt += 1

    print(total_cer / total_cnt)

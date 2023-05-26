from utils.parse_utils import parse_aste


def parse_and_score(pred, target, target_format):
    all_labels, all_preds = [], []
    for i in range(len(pred)):
        gold_list = parse_aste(target[i])
        pred_list = parse_aste(pred[i])
        all_labels.append(gold_list)
        all_preds.append(pred_list)
    if target_format == "AO":
        all_preds = distance_aware_merge_preds(all_preds)
        all_labels = all_labels[: int(len(all_labels) / 2)]
    raw_score = score(all_preds, all_labels)
    return raw_score


def distance_aware_merge_preds(preds):
    pred_nums = len(preds) // 2
    merged_preds = []
    for i in range(pred_nums):
        ao_pred, oa_pred = preds[i], preds[i + pred_nums]
        if ao_pred == oa_pred:
            pred = ao_pred
        else:
            ao_pred = list([x for x in ao_pred])
            oa_pred = list([x for x in oa_pred])
            pred = []
            ao_pred_dup, oa_pred_dup = ao_pred.copy(), oa_pred.copy()
            # add the common triplet into ans (intersection set)
            for cur in ao_pred:
                if cur in oa_pred_dup:
                    ao_pred_dup.remove(cur)
                    oa_pred_dup.remove(cur)
                    pred.append(cur)
        merged_preds.append(pred)
    return merged_preds


def score(all_preds, all_labels):
    assert len(all_preds) == len(all_labels)
    n_preds, n_labels, n_common = 0, 0, 0
    for pred, label in zip(all_preds, all_labels):
        n_preds += len(pred)
        n_labels += len(label)
        label_dup = label.copy()
        for p in pred:
            if p in label_dup:
                n_common += 1
                label_dup.remove(p)
    precision = n_common / n_preds
    recall = n_common / n_labels
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'precision': precision, "recall": recall, "f1_score": f1_score,
            "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}


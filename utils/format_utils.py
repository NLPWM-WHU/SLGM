

def ASTE_format(sentence, label, marker_order='aspect'):
    polarity2word = {'POS': "positive", "NEG": "negative", "NEU": "neutral"}
    assert marker_order in ['aspect', 'opinion'] and len(label) != 0
    all_tri = []
    for tri in label:
        if len(tri[0]) == 1:
            a = sentence[tri[0][0]]
        else:
            st, ed = tri[0][0], tri[0][-1]
            a = ' '.join(sentence[st: ed + 1])
        if len(tri[1]) == 1:
            b = sentence[tri[1][0]]
        else:
            st, ed = tri[1][0], tri[1][-1]
            b = ' '.join(sentence[st: ed + 1])
        c = polarity2word[tri[2]]
        all_tri.append((a, b, c))
    if marker_order == 'aspect':
        label_strs = ["aspect: " + x + ", opinion: " + y + ", sentiment: " + z for (x, y, z) in all_tri]
    else:
        label_strs = ["opinion: " + y + ", aspect: " + x + ", sentiment: " + z for (x, y, z) in all_tri]
    return " [SSEP] ".join(label_strs)

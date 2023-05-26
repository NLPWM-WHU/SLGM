from typing import List


def parse_aste_a(seq):
    triplets = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            _, a, b, c = s.split(":")
            a, b, c = a.strip(), b.strip(), c.strip()
            a = a.replace(', opinion', '')
            b = b.replace(', sentiment', '')
        except ValueError:
            a, b, c = '', '', ''
        triplets.append((a, b, c))
    return triplets


def parse_aste_o(seq):
    triplets = []
    sents = [s.strip() for s in seq.split('[SSEP]')]
    for s in sents:
        try:
            _, a, b, c = s.split(":")
            a, b, c = a.strip(), b.strip(), c.strip()
            a = a.replace(', aspect', '')
            b = b.replace(', sentiment', '')
        except ValueError:
            a, b, c = '', '', ''
        triplets.append((b, a, c))
    return triplets


def parse_aste(seq):
    if seq.startswith('aspect'):
        return parse_aste_a(seq)
    else:
        return parse_aste_o(seq)


def match_token_str(source: List, token: str):
    all_match_index, match_index = [], []
    begin_match = False
    for i in range(len(source)):
        cur = source[i]
        if token.startswith(cur) and not begin_match:
            match_token = token
            begin_match = True
            match_index.append(i)
            match_token = match_token.replace(cur, '', 1).strip()
        elif begin_match:
            if match_token[:len(cur)] == cur:
                match_index.append(i)
                match_token = match_token.replace(cur, '', 1).strip()
            else:
                begin_match = False
                match_index = []
                # 在这里匹配失败，但是不能放弃这个token
                if token.startswith(cur) and not begin_match:
                    match_token = token
                    begin_match = True
                    match_index.append(i)
                    match_token = match_token.replace(cur, '', 1).strip()
        if begin_match and match_token == '':
            begin_match = False
            all_match_index.append(match_index)
            match_index = []
    all_match_index = [tuple(x) for x in all_match_index]
    return all_match_index


def match_triplet_str(source, triplet):
    # closest !!!
    a, o = triplet[0], triplet[1]
    a_indexs = match_token_str(source, a)
    o_indexs = match_token_str(source, o)
    ans = []
    for a_index in a_indexs:
        for o_index in o_indexs:
            ans.append((
                abs(a_index[0] - o_index[0]),
                a_index, o_index, triplet[-1]
            ))
    ans.sort(key=lambda x: x[0])
    return ans[0]

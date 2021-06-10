def scorer(doc):
    tags = ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$', 'IN', 'EX', 'RB', 'RBR', 'RBS', 'WRB', 'CC', 'DT', 'UH', 'NN', 'NNS', 'WP', 'CD', 'POS', 'RP', 'TO', 'PRP', 'NNP', 'NNPS', '-LRB-', '-RRB-', ',', ':', '.', "''", '""', '``', 'HYPH', 'LS', 'NFP', '_SP', '#', '$', 'SYM', 'BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'ADD', 'FW', 'GW', 'XX', 'NIL']
    counter = 0
    for tag in tags:
        for token in doc:
            if token.tag_ == tag:
                counter+=1
                break
    score = max(counter*3 - len(doc),counter)
    return f'Unique tags: {counter}\nTokens used: {len(doc)}\nSCORE: {score}\nCONGRATULATIONS!'
a = {'basc' : 1,"dadf":2, "Fdfsg": 322}
b = {'sfgdcvd' : 1,"dgddfadf":16, "Fdfsg": 7,'SFGS' : 1,'bassdc' : 1,}
c = {'sdjgf' : 5,"dfgd":8, "vcxvb": 133}

cols = [a,b,c]
for col in cols:
    max_value = max([len(str(i)) for i in col.values()])
    col_size = max([len(str(i)) for i in col.keys()]) + 2 + max_value
    print(f"|{'Col name':-^{col_size}}|")
    for k,v in col.items():
        print(f"{f'|{k}:':<{col_size-max_value}} {v:<{max_value}}|")
    print()
def parse(filename, floats=[0,1,2,5,6,7,9], 
        ):
    import csv
    lines = [];
    with open(filename) as f:
        reader=csv.reader(f, delimiter=',', quotechar='"')
        keys = reader.__next__()
        for row in reader:
            thisline = {}
            for k in range(len(keys)):
                if k in floats:
                    if row[k] == '':
                        thisline[keys[k]] = -1
                        continue
                    thisline[keys[k]] = float(row[k])
                else:
                    thisline[keys[k]] = row[k]
            lines.append(thisline)
    return lines


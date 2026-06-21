import csv

with open('output/submission.csv') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i > 10:
            break
        print(row[0], '|', row[1], '|', row[2])
        if i > 0:
            print(' ', row[3])
        print()
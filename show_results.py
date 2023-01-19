import csv
with open('output.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    i = 0
    for row in spamreader:
        print(len(row))
        print(row[34533:34544])
        if i == 1:
            print(len(row))
            print(row[0:20])
            break
            # print(', '.join(row))
        i = i + 1
    print(i)

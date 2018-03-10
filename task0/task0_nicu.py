import csv


csv_file = open('output.csv', 'w')
csv_writer = csv.writer(csv_file, delimiter=',')

with open('test.csv', 'r') as f:
    mycsv = csv.reader(f)
    next(mycsv)  # Skip header
    csv_writer.writerow(["Id", "y"])
    for row in mycsv:
        id = row[0]
        sum = 0
        for i in range(1, len(row)):
            sum += float(row[i])
        avg = sum / (len(row) -1)
        csv_writer.writerow([id, avg])




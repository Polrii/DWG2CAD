import csv

def convert_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = [[col.replace('.', ',') for col in row] for row in reader]
    
    with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
        for row in rows:
            outfile.write('\t'.join(row) + '\n')

if __name__ == "__main__":
    file_path = "C:/python_work/AI/DWG2CAD/runs/detect/train3/results.csv"  # Change this to your file
    convert_csv(file_path)
    print(f"Modified file saved as {file_path}")

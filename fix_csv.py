import csv

def fix_csv(input_file, output_file):
    fixed_rows = []

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # First line is header
    header = lines[0].strip()
    fixed_rows.append(header.split(','))

    # Now process every two lines at a time
    i = 1
    while i < len(lines):
        # Join line i and line i+1 (assuming they form one row)
        if i + 1 < len(lines):
            combined = lines[i].strip() + ' ' + lines[i+1].strip()
            i += 2
        else:
            # In case there's an odd line left at the end
            combined = lines[i].strip()
            i += 1

        # Use CSV reader to parse the line correctly
        for row in csv.reader([combined]):
            fixed_rows.append(row)

    # Write to new clean CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)

    print(f"✅ Fixed CSV saved as: {output_file}")

if __name__ == "__main__":
    fix_csv('Data/USA_Housing.csv', 'Data/FIXED_USA_Housing.csv')

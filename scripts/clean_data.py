# AUXILLARY FILE IN CASE ERRORS WITH GESTURE DATA CSV 

input_file = "data/gesture_data.csv"
output_file = "data/cleaned_gesture_data.csv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for i, line in enumerate(infile):
        columns = line.strip().split(",")
        if len(columns) == 64:  # Replace 64 with the expected number of columns
            outfile.write(line)
        else:
            print(f"Row {i + 1} has an inconsistent column count: {len(columns)} columns.")

print("Data cleaning complete. Check 'cleaned_gesture_data.csv'.")
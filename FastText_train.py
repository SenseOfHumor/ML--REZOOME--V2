## Open file for reading and writing using 'with' statement
with open("linkedin skill", "r") as file, open("fast_train.txt", "w") as new_file:
    ## Parse the file and add __label__skill at the end of each line and write to a new file
    for line in file:
        new_file.write(line.strip() + " __label__skill\n")

print("File 'fast_train.txt' has been created.")
import argparse

def main():
    """
    usage: 
    python label_parser.py -i "path_to_input_txt_file" "path_to_output_lst_file"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="path of input file to parse")
    parser.add_argument("-o", help="path of output file .lst")

    args = parser.parse_args()

    with open(args.i, "r") as f:
        lines = f.readlines()
    
    file_names = []

    # we get the name of the file
    for line in lines:
        tokens = line.split()
        file_names.append(tokens[1] + "\n")

    with open(args.o, "w") as f:
        f.writelines(file_names)

if __name__ == "__main__":
    main()
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input file name")
    parser.add_argument("-o", help="output file name")
    args = parser.parse_args()


    with open(args.i, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        l = line.split(" ")
        # l.pop(3)
        l.pop(2)
        lines[i] = " ".join(l)
        # print(lines[i])

    # print(lines)
    with open(args.o, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    main()